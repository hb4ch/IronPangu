//! Hygienic, bounded elaboration. Modules and folds disappear before graph typing.
use crate::syntax::{Document, Expr, Member, Module, SpannedMember};
use inferfabric_authoring::execution::{ExpansionFrame, SourceOrigin};
use inferfabric_model::{Result, invalid};
use std::collections::{BTreeMap, BTreeSet};
const PRIMITIVES: &[&str] = &[
    "embedding",
    "cast",
    "quantize",
    "repeat",
    "mhc_pre",
    "mhc_merge",
    "mhc_input",
    "mhc_post",
    "norm",
    "optional_norm",
    "linear",
    "gather",
    "allreduce",
    "mome",
    "slice",
    "rope",
    "concat",
    "mla_query",
    "mla_value",
    "sink_normalize",
    "cache_append",
    "cache_norm_rope_append",
    "latent_attention",
    "swiglu",
    "router",
    "expert_linear",
    "expert_swiglu_down",
    "expert_reduce",
    "add",
];
const STATIC: &[&str] = &["weights", "child", "choose", "div", "mod", "eq", "lt"];
#[derive(Clone)]
enum Value {
    Tensor(String),
    Scalar(Expr),
    Scope(String),
}
type Env = BTreeMap<String, Value>;
struct Expander<'a> {
    modules: BTreeMap<String, &'a Module>,
    members: Vec<SpannedMember>,
    origins: BTreeMap<String, SourceOrigin>,
    frames: Vec<ExpansionFrame>,
    steps: usize,
}
fn error(site: &SpannedMember, message: impl AsRef<str>) -> inferfabric_model::Error {
    invalid(format!("byte {}: {}", site.start, message.as_ref()))
}
fn literal(v: Value, site: &SpannedMember) -> Result<Expr> {
    match v {
        Value::Scalar(e) => Ok(e),
        Value::Scope(s) => Ok(Expr::String(s)),
        Value::Tensor(_) => Err(error(
            site,
            "tensor supplied where a compile-time attribute is required",
        )),
    }
}
fn integer(v: Value, site: &SpannedMember) -> Result<usize> {
    match v {
        Value::Scalar(Expr::Number(s)) => s
            .parse()
            .map_err(|_| error(site, "expected nonnegative compile-time integer")),
        _ => Err(error(site, "expected compile-time integer")),
    }
}
fn tensor(v: Value, site: &SpannedMember) -> Result<String> {
    match v {
        Value::Tensor(s) => Ok(s),
        _ => Err(error(site, "expected tensor/graph handle")),
    }
}
fn scoped(base: &str, component: &str, site: &SpannedMember) -> Result<String> {
    if component.is_empty()
        || component.split('.').any(|s| {
            s.is_empty()
                || (!s.bytes().all(|c| c.is_ascii_alphanumeric() || c == b'_') && s != "{expert}")
        })
    {
        return Err(error(site, "invalid weight scope component"));
    }
    let path = if base.is_empty() {
        component.into()
    } else {
        format!("{base}.{component}")
    };
    if path.len() > 1024 {
        return Err(error(site, "weight path limit"));
    }
    Ok(path)
}
impl Expander<'_> {
    fn tick(&mut self, site: &SpannedMember) -> Result<()> {
        self.steps += 1;
        if self.steps > 100000 {
            return Err(error(site, "compile-time expansion work limit"));
        }
        Ok(())
    }
    fn resolve(&self, name: &str, env: &Env, site: &SpannedMember) -> Result<Value> {
        if let Some(v) = env.get(name) {
            return Ok(v.clone());
        }
        if let Some((root, tail)) = name.split_once('.')
            && let Some(Value::Scope(base)) = env.get(root)
        {
            return Ok(Value::Scope(scoped(base, tail, site)?));
        }
        if ["true", "false"].contains(&name) {
            return Ok(Value::Scalar(Expr::Name(name.into())));
        }
        Err(error(site, format!("unknown binding {name}")))
    }
    fn eval(
        &mut self,
        e: &Expr,
        env: &Env,
        path: &str,
        site: &SpannedMember,
        depth: usize,
    ) -> Result<Value> {
        self.tick(site)?;
        if depth > 64 || path.len() > 1024 {
            return Err(error(
                site,
                "module/expression expansion depth or path limit",
            ));
        }
        match e {
            Expr::Name(n) => self.resolve(n, env, site),
            Expr::String(_) | Expr::Number(_) => Ok(Value::Scalar(e.clone())),
            Expr::List(_) => Err(error(site, "list is not a token-graph value")),
            Expr::Fold {
                index,
                start,
                end,
                carry,
                initial,
                body,
            } => {
                if index == carry {
                    return Err(error(site, "loop index and carry must differ"));
                }
                let begin = integer(self.eval(start, env, path, site, depth + 1)?, site)?;
                let end = integer(self.eval(end, env, path, site, depth + 1)?, site)?;
                if begin > end || end - begin > 4096 {
                    return Err(error(
                        site,
                        "fold range must be ascending with at most 4096 iterations",
                    ));
                }
                let mut value = self.eval(initial, env, path, site, depth + 1)?;
                tensor(value.clone(), site)?;
                for i in begin..end {
                    let mut local = env.clone();
                    local.insert(index.clone(), Value::Scalar(Expr::Number(i.to_string())));
                    local.insert(carry.clone(), value);
                    let instance = format!("{path}.i{i}");
                    self.frames.push(ExpansionFrame {
                        module: "fold".into(),
                        instance: instance.clone(),
                        start: site.start,
                        end: site.end,
                    });
                    value = self.body(body, local, &instance, depth + 1)?;
                    self.frames.pop();
                    tensor(value.clone(), site)?;
                }
                Ok(value)
            }
            Expr::Call(op, args) => {
                if op == "choose" {
                    self.positional(args, 3, site)?;
                    let cond = self.eval(&args[0].1, env, path, site, depth + 1)?;
                    let Value::Scalar(Expr::Name(cond)) = cond else {
                        return Err(error(
                            site,
                            "choose condition must be a compile-time boolean",
                        ));
                    };
                    let selected = match cond.as_str() {
                        "true" => 1,
                        "false" => 2,
                        _ => return Err(error(site, "choose condition must be boolean")),
                    };
                    return self.eval(&args[selected].1, env, path, site, depth + 1);
                }
                if op == "weights" || op == "child" {
                    if op == "weights" {
                        self.positional(args, 1, site)?;
                    } else if args.len() < 2 || args.iter().any(|a| a.0.is_some()) {
                        return Err(error(
                            site,
                            "child requires scope and positional components",
                        ));
                    }
                    let first = self.eval(&args[0].1, env, path, site, depth + 1)?;
                    let mut scope = match (op.as_str(), first) {
                        ("weights", Value::Scalar(Expr::String(s))) => scoped("", &s, site)?,
                        ("child", Value::Scope(s)) => s,
                        _ => {
                            return Err(error(
                                site,
                                "weights requires a string; child requires a weight scope",
                            ));
                        }
                    };
                    for (_, e) in &args[1..] {
                        let part = match self.eval(e, env, path, site, depth + 1)? {
                            Value::Scalar(Expr::String(s)) => s,
                            Value::Scalar(Expr::Number(s)) => {
                                let n: usize = s.parse().map_err(|_| {
                                    error(site, "weight index must be nonnegative integer")
                                })?;
                                n.to_string()
                            }
                            _ => return Err(error(site, "invalid scope component")),
                        };
                        scope = scoped(&scope, &part, site)?;
                    }
                    return Ok(Value::Scope(scope));
                }
                if ["div", "mod", "eq", "lt"].contains(&op.as_str()) {
                    self.positional(args, 2, site)?;
                    let a = integer(self.eval(&args[0].1, env, path, site, depth + 1)?, site)?;
                    let b = integer(self.eval(&args[1].1, env, path, site, depth + 1)?, site)?;
                    let e = match op.as_str() {
                        "div" => {
                            if b == 0 || !a.is_multiple_of(b) {
                                return Err(error(
                                    site,
                                    "div requires nonzero divisor and exact division",
                                ));
                            }
                            Expr::Number((a / b).to_string())
                        }
                        "mod" => {
                            if b == 0 {
                                return Err(error(site, "modulo by zero"));
                            }
                            Expr::Number((a % b).to_string())
                        }
                        "eq" => Expr::Name((a == b).to_string()),
                        _ => Expr::Name((a < b).to_string()),
                    };
                    return Ok(Value::Scalar(e));
                }
                if let Some(module) = self.modules.get(op).copied() {
                    let mut local = Env::new();
                    let mut position = 0;
                    let mut named = false;
                    for (k, e) in args {
                        let key = if let Some(k) = k {
                            named = true;
                            if !module.parameters.contains(k) {
                                return Err(error(site, format!("unknown parameter {k} for {op}")));
                            }
                            k.clone()
                        } else {
                            if named {
                                return Err(error(
                                    site,
                                    "positional argument after named argument",
                                ));
                            }
                            let k = module
                                .parameters
                                .get(position)
                                .ok_or_else(|| error(site, "too many module arguments"))?
                                .clone();
                            position += 1;
                            k
                        };
                        let arg_path = format!("{path}.arg_{key}");
                        let v = self.eval(e, env, &arg_path, site, depth + 1)?;
                        if local.insert(key.clone(), v).is_some() {
                            return Err(error(site, format!("duplicate module argument {key}")));
                        }
                    }
                    if local.len() != module.parameters.len() {
                        return Err(error(site, format!("missing arguments for {op}")));
                    }
                    self.frames.push(ExpansionFrame {
                        module: op.clone(),
                        instance: path.into(),
                        start: site.start,
                        end: site.end,
                    });
                    let value = self.body(&module.body, local, path, depth + 1)?;
                    self.frames.pop();
                    return Ok(value);
                }
                if !PRIMITIVES.contains(&op.as_str()) {
                    return Err(error(site, format!("unknown module/operator {op}")));
                }
                let mut expanded = vec![];
                for (i, (k, e)) in args.iter().enumerate() {
                    let v = if k.is_some() {
                        match e {
                            Expr::Name(n) if !env.contains_key(n) && !n.contains('.') => {
                                Value::Scalar(e.clone())
                            }
                            _ => self.eval(e, env, &format!("{path}.arg{i}"), site, depth + 1)?,
                        }
                    } else {
                        self.eval(e, env, &format!("{path}.arg{i}"), site, depth + 1)?
                    };
                    expanded.push((
                        k.clone(),
                        if k.is_some() {
                            literal(v, site)?
                        } else {
                            Expr::Name(tensor(v, site)?)
                        },
                    ));
                }
                if self.origins.len() >= 10000 {
                    return Err(error(site, "expanded node limit"));
                }
                let origin = SourceOrigin {
                    start: site.start,
                    end: site.end,
                    frames: self.frames.clone(),
                };
                if self.origins.insert(path.into(), origin).is_some() {
                    return Err(error(site, "expanded instance name collision"));
                }
                self.members.push(SpannedMember {
                    start: site.start,
                    end: site.end,
                    member: Member::Let(path.into(), Expr::Call(op.clone(), expanded)),
                });
                Ok(Value::Tensor(path.into()))
            }
        }
    }
    fn positional(
        &self,
        args: &[(Option<String>, Expr)],
        count: usize,
        site: &SpannedMember,
    ) -> Result<()> {
        if args.len() != count || args.iter().any(|a| a.0.is_some()) {
            Err(error(
                site,
                format!("expected {count} positional arguments"),
            ))
        } else {
            Ok(())
        }
    }
    fn body(
        &mut self,
        body: &[SpannedMember],
        mut env: Env,
        prefix: &str,
        depth: usize,
    ) -> Result<Value> {
        let mut result = None;
        for m in body {
            self.tick(m)?;
            if result.is_some() {
                return Err(error(m, "declaration after yield"));
            }
            match &m.member {
                Member::Let(n, e) => {
                    if env.contains_key(n) {
                        return Err(error(m, format!("duplicate local binding {n}")));
                    }
                    let v = self.eval(e, &env, &format!("{prefix}.{n}"), m, depth + 1)?;
                    env.insert(n.clone(), v);
                }
                Member::Yield(n) => result = Some(self.resolve(n, &env, m)?),
                _ => return Err(error(m, "module/fold bodies contain only let and yield")),
            }
        }
        result.ok_or_else(|| invalid(format!("{prefix}: module/fold missing yield")))
    }
}
fn dependencies(e: &Expr, called: &mut BTreeSet<String>) {
    match e {
        Expr::Call(n, args) => {
            called.insert(n.clone());
            for (_, e) in args {
                dependencies(e, called);
            }
        }
        Expr::Fold {
            start,
            end,
            initial,
            body,
            ..
        } => {
            for e in [start, end, initial] {
                dependencies(e, called);
            }
            for m in body {
                if let Member::Let(_, e) = &m.member {
                    dependencies(e, called);
                }
            }
        }
        _ => {}
    }
}
fn visit(
    name: &str,
    deps: &BTreeMap<String, BTreeSet<String>>,
    active: &mut BTreeSet<String>,
    done: &mut BTreeSet<String>,
) -> Result<()> {
    if done.contains(name) {
        return Ok(());
    }
    if !active.insert(name.into()) {
        return Err(invalid(format!("recursive DSL module {name}")));
    }
    for next in &deps[name] {
        if deps.contains_key(next) {
            visit(next, deps, active, done)?;
        }
    }
    active.remove(name);
    done.insert(name.into());
    Ok(())
}
pub fn expand(ast: &Document) -> Result<(Document, BTreeMap<String, SourceOrigin>)> {
    let mut modules = BTreeMap::new();
    let mut deps = BTreeMap::new();
    for m in &ast.modules {
        if PRIMITIVES.contains(&m.name.as_str())
            || STATIC.contains(&m.name.as_str())
            || modules.insert(m.name.clone(), m).is_some()
        {
            return Err(invalid(format!("duplicate/reserved module {}", m.name)));
        }
        if m.parameters.len() > 64
            || m.parameters.iter().collect::<BTreeSet<_>>().len() != m.parameters.len()
        {
            return Err(invalid("duplicate/excessive module parameters"));
        }
        let mut called = BTreeSet::new();
        for member in &m.body {
            if let Member::Let(_, e) = &member.member {
                dependencies(e, &mut called);
            }
        }
        deps.insert(m.name.clone(), called);
    }
    for called in deps.values().flatten() {
        if !modules.contains_key(called)
            && !PRIMITIVES.contains(&called.as_str())
            && !STATIC.contains(&called.as_str())
        {
            return Err(invalid(format!("unknown module/operator {called}")));
        }
    }
    let mut done = BTreeSet::new();
    for name in deps.keys() {
        visit(name, &deps, &mut BTreeSet::new(), &mut done)?;
    }
    let mut x = Expander {
        modules,
        members: vec![],
        origins: BTreeMap::new(),
        frames: vec![],
        steps: 0,
    };
    let mut env = Env::new();
    let mut yielded = false;
    for m in &ast.model {
        if yielded {
            return Err(error(m, "declaration after model yield"));
        }
        match &m.member {
            Member::Let(n, e) => {
                if env.contains_key(n) {
                    return Err(error(m, "duplicate model binding"));
                }
                let v = x.eval(e, &env, n, m, 0)?;
                env.insert(n.clone(), v);
            }
            Member::Yield(n) => {
                let result = tensor(x.resolve(n, &env, m)?, m)?;
                x.members.push(SpannedMember {
                    start: m.start,
                    end: m.end,
                    member: Member::Yield(result),
                });
                yielded = true;
            }
            Member::Field(_, _) => x.members.push(m.clone()),
            _ => return Err(error(m, "unsupported token graph declaration")),
        }
    }
    let mut expanded = ast.clone();
    expanded.modules.clear();
    expanded.model = x.members;
    Ok((expanded, x.origins))
}
