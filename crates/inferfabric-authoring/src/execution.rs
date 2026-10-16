//! Typed, ordered token dataflow. Both authoring frontends use this IR.
//! Stateful kernels own caches; ordering and all data dependencies belong here.
use inferfabric_model::{Result, invalid};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct ExpansionFrame {
    pub module: String,
    pub instance: String,
    pub start: usize,
    pub end: usize,
}
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct SourceOrigin {
    pub start: usize,
    pub end: usize,
    pub frames: Vec<ExpansionFrame>,
}
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct Node {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source: Option<SourceOrigin>,
    pub name: String,
    pub op: String,
    pub inputs: Vec<usize>,
    pub attrs: BTreeMap<String, String>,
}
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct Graph {
    pub name: String,
    pub max_context: usize,
    pub eos: usize,
    pub nodes: Vec<Node>,
    pub output: usize,
}
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Type {
    Tensor(usize),
    Quantized(usize),
    HyperConnection,
    Routes,
    Experts(usize),
    Cache,
}
fn fail<T>(n: &Node, message: &str) -> Result<T> {
    Err(invalid(format!("{} ({}): {message}", n.name, n.op)))
}
impl Node {
    fn cache_attributes(&self, mut attrs: Vec<&'static str>) -> Result<Vec<&'static str>> {
        if let Some(dtype)=self.attrs.get("cache_dtype") {
            attrs.push("cache_dtype");
            if !["bf16","int8","int4","int8_ds_mla"].contains(&dtype.as_str()) {return fail(self,"unsupported cache dtype");}
            if dtype=="int8_ds_mla" {
                attrs.extend(["cache_scale_dtype","cache_quantization","cache_rope_dtype"]);
                if self.op!="cache_norm_rope_append" || self.attrs.get("cache_scale_dtype").map(String::as_str)!=Some("f32") || self.attrs.get("cache_quantization").map(String::as_str)!=Some("pertile128") || self.attrs.get("cache_rope_dtype").map(String::as_str)!=Some("bf16") {
                    return fail(self,"packed MLA C8 requires fused normalization, pertile128 FP32 scales and BF16 rotary");
                }
            } else if dtype!="bf16" {
                attrs.extend(["cache_scale_dtype","cache_quantization"]);
                if self.attrs.get("cache_scale_dtype").map(String::as_str)!=Some("f32") || self.attrs.get("cache_quantization").map(String::as_str)!=Some("per_token_head_symmetric") {return fail(self,"quantized cache requires FP32 per-token/head symmetric scales");}
            }
        }
        Ok(attrs)
    }
    pub fn integer(&self, key: &str) -> Result<usize> {
        self.attrs
            .get(key)
            .and_then(|s| s.parse().ok())
            .ok_or_else(|| invalid(format!("{}: invalid {key}", self.name)))
    }
    fn schema(&self, inputs: &[Type], expected: &[Type], attrs: &[&str]) -> Result<()> {
        if inputs != expected {
            return fail(self, "input type/shape mismatch");
        }
        if self
            .attrs
            .keys()
            .map(String::as_str)
            .collect::<BTreeSet<_>>()
            != attrs.iter().copied().collect()
        {
            return fail(self, "missing or unknown attributes");
        }
        for (k, v) in &self.attrs {
            if v.is_empty() || v.len() > 1024 || v.contains(['\0', '\n']) {
                return fail(self, "invalid attribute string");
            }
            if k == "weight"
                && (!v
                    .bytes()
                    .all(|c| c.is_ascii_alphanumeric() || b"._{}".contains(&c))
                    || v.contains(".."))
            {
                return fail(self, "invalid weight binding");
            }
        }
        Ok(())
    }
    fn infer(&self, inputs: &[Type]) -> Result<Type> {
        use Type::*;
        let tensor = |i| match inputs.get(i) {
            Some(Tensor(n)) => Ok(*n),
            _ => fail(self, "expected tensor"),
        };
        let width = || {
            let w = self.integer("width")?;
            if w == 0 || w > 200000 {
                return fail(self, "width out of range");
            }
            Ok(w)
        };
        Ok(match self.op.as_str() {
            "cast" => {
                let w = tensor(0)?;
                self.schema(inputs, &[Tensor(w)], &["dtype"])?;
                if !["bf16", "f32"].contains(&self.attrs["dtype"].as_str()) { return fail(self,"cast requires bf16 or f32; use quantize for integer storage"); }
                Tensor(w)
            }
            "quantize" => {
                let w = tensor(0)?;
                self.schema(inputs, &[Tensor(w)], &["dtype", "scheme", "scale_dtype"])?;
                if self.attrs["dtype"] != "int8" || self.attrs["scheme"] != "per_token_symmetric" || self.attrs["scale_dtype"] != "f32" { return fail(self,"unsupported runtime quantization contract"); }
                Quantized(w)
            }
            "embedding" => {
                self.schema(inputs, &[], &["weight", "width", "vocab"])?;
                if self.integer("vocab")? != 151552 || width()? != 2560 {
                    return fail(self, "unsupported embedding specialization");
                }
                Tensor(width()?)
            }
            "repeat" => {
                let w = tensor(0)?;
                self.schema(inputs, &[Tensor(w)], &["count"])?;
                let c = self.integer("count")?;
                if c == 0 || c > 16 {
                    return fail(self, "repeat count");
                }
                Tensor(w * c)
            }
            "mhc_pre" | "mhc_merge" => {
                self.schema(
                    inputs,
                    &[Tensor(10240)],
                    &["weight", "streams", "sinkhorn", "eps"],
                )?;
                if self.integer("streams")? != 4
                    || self.integer("sinkhorn")? != 20
                    || self.attrs["eps"] != "1e-6"
                {
                    return fail(self, "unsupported mHC specialization");
                }
                HyperConnection
            }
            "mhc_input" => {
                self.schema(inputs, &[HyperConnection], &[])?;
                Tensor(2560)
            }
            "mhc_post" => {
                self.schema(inputs, &[HyperConnection, Tensor(2560)], &[])?;
                Tensor(10240)
            }
            "norm" | "optional_norm" => {
                let w = tensor(0)?;
                self.schema(inputs, &[Tensor(w)], &["weight", "eps"])?;
                if self.attrs["eps"] != "1e-5" {
                    return fail(self, "unsupported RMSNorm epsilon");
                }
                Tensor(w)
            }
            "linear" => {
                let w = match inputs.first() { Some(Quantized(w))=>*w, _=>tensor(0)? };
                let mut attrs = vec!["weight", "width", "placement"];
                let mut output = width()?;
                if self.attrs.contains_key("paired_weight") {
                    if matches!(inputs.first(),Some(Quantized(_))) { return fail(self,"paired BF16 projection cannot consume quantized input"); }
                    attrs.extend(["paired_weight", "paired_width"]);
                    let paired = self.integer("paired_width")?;
                    if paired == 0 || paired > 200000 - output || self.attrs.get("placement").map(String::as_str) != Some("replicated") {
                        return fail(self, "paired projection requires bounded widths and replicated placement");
                    }
                    output += paired;
                }
                let expected = if matches!(inputs.first(),Some(Quantized(_))) { Quantized(w) } else { Tensor(w) };
                self.schema(inputs, &[expected], &attrs)?;
                if !["tp_rows", "tp_columns", "replicated"]
                    .contains(&self.attrs["placement"].as_str())
                {
                    return fail(self, "unknown linear placement");
                }
                Tensor(output)
            }
            "gather" => {
                let w = tensor(0)?;
                self.schema(inputs, &[Tensor(w)], &[])?;
                Tensor(w * 8)
            }
            "allreduce" => {
                let w = tensor(0)?;
                let mut attrs = vec![];
                if let Some(dtype) = self.attrs.get("dtype") {
                    attrs.push("dtype");
                    if !["f32", "bf16"].contains(&dtype.as_str()) {
                        return fail(self, "unsupported collective dtype");
                    }
                }
                self.schema(inputs, &[Tensor(w)], &attrs)?;
                Tensor(w)
            }
            "mome" => {
                let w = tensor(0)?;
                let mut attrs = vec!["weight", "taps"];
                if let Some(placement) = self.attrs.get("placement") {
                    attrs.push("placement");
                    if !["replicated", "tp_channels"].contains(&placement.as_str()) {
                        return fail(self, "unsupported convolution placement");
                    }
                }
                self.schema(inputs, &[Tensor(w)], &attrs)?;
                if self.integer("taps")? != 3 {
                    return fail(self, "unsupported convolution taps");
                }
                Tensor(w)
            }
            "slice" => {
                let w = tensor(0)?;
                self.schema(inputs, &[Tensor(w)], &["start", "width"])?;
                let a = self.integer("start")?;
                let b = width()?;
                if a.checked_add(b).is_none_or(|n| n > w) {
                    return fail(self, "slice bounds");
                }
                Tensor(b)
            }
            "rope" => {
                self.schema(inputs, &[Tensor(64)], &["theta"])?;
                if self.attrs["theta"] != "6400000" {
                    return fail(self, "unsupported RoPE specialization");
                }
                Tensor(64)
            }
            "concat" => {
                let a = tensor(0)?;
                let b = tensor(1)?;
                self.schema(inputs, &[Tensor(a), Tensor(b)], &[])?;
                Tensor(a + b)
            }
            "mla_query" => {
                let mut attrs = vec!["weight", "theta", "heads", "nope", "rope", "latent"];
                let width = if let Some(layout) = self.attrs.get("input_layout") {
                    attrs.push("input_layout");
                    match layout.as_str() {
                        "rank_local" => 1152,
                        "replicated" => 9216,
                        _ => return fail(self, "unsupported query input layout"),
                    }
                } else {
                    9216
                };
                self.schema(inputs, &[Tensor(width)], &attrs)?;
                for (k, v) in [
                    ("theta", 6400000),
                    ("heads", 48),
                    ("nope", 128),
                    ("rope", 64),
                    ("latent", 512),
                ] {
                    if self.integer(k)? != v {
                        return fail(self, "unsupported query absorption specialization");
                    }
                }
                Tensor(3456)
            }
            "sink_normalize" => {
                self.schema(
                    inputs,
                    &[],
                    &["weight", "norm", "count", "latent", "rope", "eps"],
                )?;
                if self.integer("count")? != 128
                    || self.integer("latent")? != 512
                    || self.integer("rope")? != 64
                    || self.attrs["eps"] != "1e-5"
                {
                    return fail(self, "unsupported sink specialization");
                }
                Tensor(73728)
            }
            "cache_append" => {
                self.schema(inputs, &[Tensor(576), Tensor(73728)], &self.cache_attributes(vec![])?)?;
                Cache
            }
            "cache_norm_rope_append" => {
                self.schema(
                    inputs,
                    &[Tensor(512), Tensor(64), Tensor(73728)],
                    &self.cache_attributes(vec!["weight", "eps", "theta"] )?,
                )?;
                if self.attrs["eps"] != "1e-5" || self.integer("theta")? != 6400000 {
                    return fail(
                        self,
                        "unsupported fused cache normalization/rotary contract",
                    );
                }
                Cache
            }
            "latent_attention" => {
                self.schema(
                    inputs,
                    &[Tensor(3456), Cache],
                    &["scale_dim", "window", "selection"],
                )?;
                if self.integer("scale_dim")? != 192
                    || ![512, 2048].contains(&self.integer("window")?)
                    || self.attrs["selection"] != "all_causal_within_window"
                {
                    return fail(self, "unsupported attention semantics");
                }
                Tensor(3072)
            }
            "mla_value" => {
                self.schema(
                    inputs,
                    &[Tensor(3072)],
                    &["weight", "heads", "latent", "value"],
                )?;
                if self.integer("heads")? != 48
                    || self.integer("latent")? != 512
                    || self.integer("value")? != 128
                {
                    return fail(self, "unsupported value expansion specialization");
                }
                Tensor(768)
            }
            "swiglu" => {
                let w = tensor(0)?;
                if inputs.len() == 1 {
                    self.schema(inputs, &[Tensor(w)], &[])?;
                    if w == 0 || w % 2 != 0 { return fail(self, "packed SwiGLU needs equal halves"); }
                    Tensor(w / 2)
                } else {
                    self.schema(inputs, &[Tensor(w), Tensor(w)], &[])?;
                    Tensor(w)
                }
            }
            "router" => {
                self.schema(
                    inputs,
                    &[Tensor(2560)],
                    &["weight", "experts", "topk", "scoring", "scale"],
                )?;
                if self.integer("experts")? != 256
                    || self.integer("topk")? != 8
                    || self.attrs["scoring"] != "sigmoid_bias_group_topk"
                    || self.attrs["scale"] != "2.5"
                {
                    return fail(self, "unsupported router specialization");
                }
                Routes
            }
            "expert_linear" => {
                let mut attrs = vec!["weight", "width"];
                let paired = self.attrs.get("paired_weight");
                if let Some(weight) = paired {
                    attrs.push("paired_weight");
                    if !weight.contains("{expert}") {
                        return fail(self, "paired expert binding");
                    }
                }
                self.schema(inputs, &[Tensor(2560), Routes], &attrs)?;
                if width()? != 1024 || !self.attrs["weight"].contains("{expert}") {
                    return fail(self, "expert projection specialization/binding");
                }
                Experts(if paired.is_some() { 2048 } else { 1024 })
            }
            "expert_swiglu_down" => {
                self.schema(
                    inputs,
                    if inputs.len() == 2 {
                        &[Experts(2048), Routes]
                    } else {
                        &[Experts(1024), Experts(1024), Routes]
                    },
                    &["weight", "width"],
                )?;
                if width()? != 2560 || !self.attrs["weight"].contains("{expert}") {
                    return fail(self, "expert down specialization/binding");
                }
                Experts(2560)
            }
            "expert_reduce" => {
                self.schema(inputs, &[Experts(2560), Routes], &[])?;
                Tensor(2560)
            }
            "add" => {
                let w = tensor(0)?;
                self.schema(inputs, &[Tensor(w), Tensor(w)], &[])?;
                Tensor(w)
            }
            _ => return fail(self, "unknown operation"),
        })
    }
}
impl Graph {
    pub fn validate(&self) -> Result<Vec<Type>> {
        if self.nodes.is_empty()
            || self.nodes.len() > 10000
            || self.max_context == 0
            || self.max_context > 512
            || self.eos >= 151552
            || self.name.is_empty()
            || self.name.len() > 1024
        {
            return Err(invalid("token graph bounds"));
        }
        let mut types: Vec<Type> = vec![];
        let mut names = BTreeSet::new();
        for (i, n) in self.nodes.iter().enumerate() {
            if n.name.is_empty()
                || n.name.len() > 1024
                || !names.insert(&n.name)
                || n.inputs.iter().any(|&j| j >= i)
            {
                return fail(n, "duplicate name or non-topological dependency");
            }
            let inputs = n
                .inputs
                .iter()
                .map(|&j| types[j].clone())
                .collect::<Vec<Type>>();
            let t = n.infer(&inputs)?;
            if let Type::Tensor(w) = t
                && w > 200000
            {
                return fail(n, "tensor width bound");
            }
            if n.op == "latent_attention" && self.max_context > n.integer("window")? {
                return fail(n, "context exceeds exact causal attention window");
            }
            if n.op == "expert_swiglu_down" {
                let routes = *n.inputs.last().unwrap();
                for &j in &n.inputs[..n.inputs.len() - 1] {
                    let source = &self.nodes[j];
                    if source.op != "expert_linear" || source.inputs[1] != routes {
                        return fail(n, "expert route provenance mismatch");
                    }
                }
            }
            if n.op == "expert_reduce" {
                let source = &self.nodes[n.inputs[0]];
                if source.op != "expert_swiglu_down" || source.inputs.last() != Some(&n.inputs[1]) {
                    return fail(n, "expert route provenance mismatch");
                }
            }
            types.push(t);
        }
        if types.get(self.output) != Some(&Type::Tensor(151552)) {
            return Err(invalid("output must be vocabulary logits"));
        }
        let mut live = BTreeSet::from([self.output]);
        for i in (0..self.nodes.len()).rev() {
            if live.contains(&i) {
                live.extend(self.nodes[i].inputs.iter());
            }
        }
        if live.len() != self.nodes.len() {
            return Err(invalid("unreachable node/state effect in token graph"));
        }
        // Every mutable mHC/cache handle has one consumer. No hidden ordering or aliasing.
        for (i, t) in types.iter().enumerate() {
            let uses = self
                .nodes
                .iter()
                .filter(|n| n.inputs.contains(&i))
                .collect::<Vec<_>>();
            if *t == Type::Cache && (uses.len() != 1 || uses[0].op != "latent_attention") {
                return Err(invalid("cache handle must feed exactly one attention"));
            }
            if *t == Type::HyperConnection {
                let pre = self.nodes[i].op == "mhc_pre";
                if uses.iter().filter(|n| n.op == "mhc_input").count() != 1
                    || uses.iter().filter(|n| n.op == "mhc_post").count() != usize::from(pre)
                {
                    return Err(invalid("unpaired hyper-connection state"));
                }
            }
        }
        Ok(types)
    }
    /// Stable binary consumed without a source parser or model-layer loop.
    pub fn encode(&self) -> Result<Vec<u8>> {
        let types = self.validate()?;
        fn u32_(b: &mut Vec<u8>, n: usize) {
            b.extend_from_slice(&(n as u32).to_le_bytes());
        }
        fn str_(b: &mut Vec<u8>, s: &str) {
            u32_(b, s.len());
            b.extend_from_slice(s.as_bytes());
        }
        let mut b = b"IFTOKN01".to_vec();
        u32_(&mut b, self.max_context);
        u32_(&mut b, self.eos);
        u32_(&mut b, self.output);
        u32_(&mut b, self.nodes.len());
        for (n, t) in self.nodes.iter().zip(types) {
            let (tag, width) = match t {
                Type::Tensor(w) => (0, w),
                Type::HyperConnection => (1, 0),
                Type::Routes => (2, 0),
                Type::Experts(w) => (3, w),
                Type::Cache => (4, 0),
                Type::Quantized(_) => return Err(invalid("quantized values require physical program encoding")),
            };
            u32_(&mut b, tag);
            u32_(&mut b, width);
            str_(&mut b, &n.name);
            str_(&mut b, &n.op);
            u32_(&mut b, n.inputs.len());
            for &i in &n.inputs {
                u32_(&mut b, i);
            }
            u32_(&mut b, n.attrs.len());
            for (k, v) in &n.attrs {
                str_(&mut b, k);
                str_(&mut b, v);
            }
        }
        Ok(b)
    }
    pub fn dot(&self) -> Result<String> {
        self.validate()?;
        let mut s = String::from("digraph token_plan {\nrankdir=LR;\n");
        for (i, n) in self.nodes.iter().enumerate() {
            s.push_str(&format!(
                "n{i} [label={:?}];\n",
                format!("{}\\n{}", n.name, n.op)
            ));
            for j in &n.inputs {
                s.push_str(&format!("n{j} -> n{i};\n"));
            }
        }
        s.push_str("}\n");
        Ok(s)
    }
}
