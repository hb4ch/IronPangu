//! Bounded v1 lexer and span-bearing AST. Legacy key/value files are rejected.
use inferfabric_model::{Result, invalid};
#[derive(Debug, Clone, PartialEq)]
pub enum Expr {
    Name(String),
    String(String),
    Number(String),
    List(Vec<Expr>),
    Call(String, Vec<(Option<String>, Expr)>),
}
#[derive(Debug, Clone, PartialEq)]
pub enum Member {
    Field(String, Expr),
    Input(String, Vec<usize>),
    Constant(String, Vec<usize>, Vec<f32>),
    State(String, Vec<usize>, Vec<f32>),
    Let(String, Expr),
    Yield(String),
    Update(String, String),
}
#[derive(Debug, Clone, PartialEq)]
pub struct SpannedMember {
    pub start: usize,
    pub end: usize,
    pub member: Member,
}
#[derive(Debug, Clone, PartialEq)]
pub struct Document {
    pub name: String,
    pub model: Vec<SpannedMember>,
    pub plan_name: String,
    pub plan: Vec<SpannedMember>,
}
#[derive(Debug, Clone)]
struct Token {
    text: String,
    start: usize,
    end: usize,
}
fn lex(source: &str) -> Result<Vec<Token>> {
    if source.len() > 1024 * 1024 {
        return Err(invalid("DSL source exceeds 1 MiB"));
    }
    let b = source.as_bytes();
    let mut i = 0;
    let mut tokens = Vec::new();
    while i < b.len() {
        if b[i].is_ascii_whitespace() {
            i += 1;
            continue;
        }
        if b[i] == b'#' || (b[i] == b'/' && b.get(i + 1) == Some(&b'/')) {
            while i < b.len() && b[i] != b'\n' {
                i += 1;
            }
            continue;
        }
        let start = i;
        if b[i] == b'"' {
            i += 1;
            let mut closed = false;
            while i < b.len() {
                if b[i] == b'\\' {
                    i += 2;
                    continue;
                }
                if b[i] == b'"' {
                    i += 1;
                    closed = true;
                    break;
                }
                i += 1;
            }
            if !closed {
                return Err(invalid(format!("byte {start}: unterminated string")));
            }
        } else if b[i].is_ascii_alphabetic() || b[i] == b'_' {
            i += 1;
            while i < b.len() && (b[i].is_ascii_alphanumeric() || b[i] == b'_' || b[i] == b'.') {
                i += 1;
            }
        } else if b[i].is_ascii_digit() || b[i] == b'-' {
            i += 1;
            while i < b.len() && (b[i].is_ascii_digit() || b".eE+-".contains(&b[i])) {
                i += 1;
            }
        } else if b"{}[]()<>=,:;".contains(&b[i]) {
            i += 1;
        } else {
            return Err(invalid(format!("byte {i}: unsupported character")));
        }
        tokens.push(Token {
            text: source[start..i].into(),
            start,
            end: i,
        });
        if tokens.len() > 100000 {
            return Err(invalid("DSL token limit"));
        }
    }
    Ok(tokens)
}
struct Parser {
    tokens: Vec<Token>,
    position: usize,
}
impl Parser {
    fn error(&self, message: &str) -> inferfabric_model::Error {
        invalid(format!(
            "byte {}: {message}",
            self.tokens
                .get(self.position)
                .map_or_else(|| self.tokens.last().map_or(0, |t| t.end), |t| t.start)
        ))
    }
    fn peek(&self) -> &str {
        self.tokens
            .get(self.position)
            .map_or("", |t| t.text.as_str())
    }
    fn take(&mut self) -> Result<String> {
        let value = self
            .tokens
            .get(self.position)
            .ok_or_else(|| self.error("unexpected end of source"))?
            .text
            .clone();
        self.position += 1;
        Ok(value)
    }
    fn expect(&mut self, text: &str) -> Result<()> {
        if self.peek() != text {
            return Err(self.error(&format!("expected '{text}'")));
        }
        self.position += 1;
        Ok(())
    }
    fn name(&mut self) -> Result<String> {
        let value = self.take()?;
        if !value
            .as_bytes()
            .first()
            .is_some_and(|c| c.is_ascii_alphabetic() || *c == b'_')
            || !value
                .bytes()
                .all(|c| c.is_ascii_alphanumeric() || c == b'_' || c == b'.')
        {
            return Err(self.error("expected identifier"));
        }
        Ok(value)
    }
    fn expr(&mut self, depth: usize) -> Result<Expr> {
        if depth > 64 {
            return Err(self.error("expression nesting limit"));
        }
        if self.peek() == "[" {
            self.take()?;
            let mut values = vec![];
            while self.peek() != "]" {
                values.push(self.expr(depth + 1)?);
                if self.peek() != "," {
                    break;
                }
                self.take()?;
            }
            self.expect("]")?;
            return Ok(Expr::List(values));
        }
        let value = self.take()?;
        if value.starts_with('"') {
            return Ok(Expr::String(
                serde_json::from_str(&value).map_err(|_| self.error("invalid string literal"))?,
            ));
        }
        if value.starts_with('-') || value.as_bytes()[0].is_ascii_digit() {
            return Ok(Expr::Number(value));
        }
        if !value.as_bytes()[0].is_ascii_alphabetic() && !value.starts_with('_') {
            return Err(self.error("expected expression"));
        }
        if self.peek() != "(" {
            return Ok(Expr::Name(value));
        }
        self.take()?;
        let mut args = vec![];
        while self.peek() != ")" {
            let key = if self
                .tokens
                .get(self.position + 1)
                .is_some_and(|t| t.text == "=")
            {
                let k = self.name()?;
                self.expect("=")?;
                Some(k)
            } else {
                None
            };
            args.push((key, self.expr(depth + 1)?));
            if self.peek() != "," {
                break;
            }
            self.take()?;
        }
        self.expect(")")?;
        Ok(Expr::Call(value, args))
    }
    fn shape(&mut self) -> Result<Vec<usize>> {
        self.expect(":")?;
        self.expect("tensor")?;
        self.expect("<")?;
        self.expect("f32")?;
        self.expect(">")?;
        let Expr::List(values) = self.expr(0)? else {
            return Err(self.error("expected shape list"));
        };
        values
            .iter()
            .map(|v| match v {
                Expr::Number(n) => n.parse().map_err(|_| self.error("invalid dimension")),
                _ => Err(self.error("dimension must be integer")),
            })
            .collect()
    }
    fn members(&mut self, graph_allowed: bool) -> Result<Vec<SpannedMember>> {
        self.expect("{")?;
        let mut members = vec![];
        while self.peek() != "}" {
            let start = self
                .tokens
                .get(self.position)
                .ok_or_else(|| self.error("unclosed declaration"))?
                .start;
            let key = self.name()?;
            let member = match key.as_str() {
                "input" if graph_allowed => {
                    let n = self.name()?;
                    Member::Input(n, self.shape()?)
                }
                "const" | "state" if graph_allowed => {
                    let n = self.name()?;
                    let shape = self.shape()?;
                    self.expect("=")?;
                    let Expr::List(values) = self.expr(0)? else {
                        return Err(self.error("expected initializer list"));
                    };
                    let values = values
                        .iter()
                        .map(|v| match v {
                            Expr::Number(n) => {
                                n.parse::<f32>().map_err(|_| self.error("invalid float"))
                            }
                            _ => Err(self.error("initializer must be numeric")),
                        })
                        .collect::<Result<Vec<_>>>()?;
                    if key == "const" {
                        Member::Constant(n, shape, values)
                    } else {
                        Member::State(n, shape, values)
                    }
                }
                "let" if graph_allowed => {
                    let n = self.name()?;
                    self.expect("=")?;
                    Member::Let(n, self.expr(0)?)
                }
                "yield" if graph_allowed => Member::Yield(self.name()?),
                "update" if graph_allowed => {
                    let n = self.name()?;
                    self.expect("=")?;
                    Member::Update(n, self.name()?)
                }
                _ => {
                    self.expect("=")?;
                    Member::Field(key, self.expr(0)?)
                }
            };
            self.expect(";")?;
            members.push(SpannedMember {
                start,
                end: self.tokens[self.position - 1].end,
                member,
            });
        }
        self.expect("}")?;
        Ok(members)
    }
}
pub fn parse(source: &str) -> Result<Document> {
    let mut p = Parser {
        tokens: lex(source)?,
        position: 0,
    };
    if p.peek() != "language" {
        return Err(invalid(
            "legacy DSL removed: use 'language inferfabric 1;' and model/plan declarations",
        ));
    }
    p.expect("language")?;
    p.expect("inferfabric")?;
    p.expect("1")?;
    p.expect(";")?;
    p.expect("model")?;
    let name = p.name()?;
    let model = p.members(true)?;
    p.expect("plan")?;
    let plan_name = p.name()?;
    p.expect("for")?;
    if p.name()? != name {
        return Err(p.error("plan references a different model"));
    }
    let plan = p.members(false)?;
    if !p.peek().is_empty() {
        return Err(
            p.error("unexpected declaration; imports/modules are not supported by this v1 subset")
        );
    }
    Ok(Document {
        name,
        model,
        plan_name,
        plan,
    })
}
