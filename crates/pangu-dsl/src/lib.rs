//! Small line-oriented DSL. Every field is required; no expressions or implicit defaults.
use pangu_model::{Layer, Mesh, Model, Result, Spec, invalid};
use std::collections::BTreeMap;

pub fn parse(source: &str) -> Result<Spec> {
    let mut fields = BTreeMap::new();
    for (line, raw) in source.lines().enumerate() {
        let raw = raw.split('#').next().unwrap_or("").trim();
        if raw.is_empty() {
            continue;
        }
        let (key, value) = raw
            .split_once('=')
            .ok_or_else(|| invalid(format!("line {}: expected key = value", line + 1)))?;
        if fields
            .insert(key.trim().to_string(), value.trim().to_string())
            .is_some()
        {
            return Err(invalid(format!("duplicate field {key}")));
        }
    }
    fn take(f: &mut BTreeMap<String, String>, key: &str) -> Result<String> {
        f.remove(key)
            .ok_or_else(|| invalid(format!("missing {key}")))
    }
    fn num(f: &mut BTreeMap<String, String>, key: &str) -> Result<usize> {
        take(f, key)?
            .parse()
            .map_err(|_| invalid(format!("invalid integer {key}")))
    }
    fn mesh(f: &mut BTreeMap<String, String>, key: &str) -> Result<Mesh> {
        let raw = take(f, key)?;
        let parts: Vec<_> = raw.split_whitespace().collect();
        if parts.len() != 3 {
            return Err(invalid("mesh syntax: tp cp sp(true|false)"));
        }
        Ok(Mesh {
            tp: parts[0].parse().map_err(|_| invalid("TP"))?,
            cp: parts[1].parse().map_err(|_| invalid("CP"))?,
            sp: parts[2].parse().map_err(|_| invalid("SP"))?,
        })
    }
    fn list(f: &mut BTreeMap<String, String>, key: &str) -> Result<Vec<usize>> {
        take(f, key)?
            .split_whitespace()
            .map(|s| {
                s.parse()
                    .map_err(|_| invalid(format!("invalid bucket {s}")))
            })
            .collect()
    }
    let model = Model {
        name: take(&mut fields, "model")?,
        hidden: num(&mut fields, "hidden")?,
        intermediate: num(&mut fields, "intermediate")?,
        vocab: num(&mut fields, "vocab")?,
        query_heads: num(&mut fields, "query_heads")?,
        kv_heads: num(&mut fields, "kv_heads")?,
        head_dim: num(&mut fields, "head_dim")?,
        delta_heads: num(&mut fields, "delta_heads")?,
        delta_dim: num(&mut fields, "delta_dim")?,
        conv_width: num(&mut fields, "conv_width")?,
        layers: take(&mut fields, "layers")?
            .split_whitespace()
            .map(|s| match s {
                "full" => Ok(Layer::Full),
                "delta" => Ok(Layer::Delta),
                _ => Err(invalid(format!("unknown layer {s}"))),
            })
            .collect::<Result<_>>()?,
    };
    let spec = Spec {
        model,
        prefill: mesh(&mut fields, "prefill")?,
        decode: mesh(&mut fields, "decode")?,
        page_tokens: num(&mut fields, "page_tokens")?,
        token_budget: num(&mut fields, "token_budget")?,
        batch_buckets: list(&mut fields, "batch_buckets")?,
        context_buckets: list(&mut fields, "context_buckets")?,
    };
    if !fields.is_empty() {
        return Err(invalid(format!("unsupported fields: {:?}", fields.keys())));
    }
    spec.validate()?;
    Ok(spec)
}
#[cfg(test)]
mod tests {
    use super::*;
    const EXAMPLE: &str = include_str!("../../../examples/qwen35-2b.pangu");
    #[test]
    fn parses_and_rejects() {
        assert_eq!(parse(EXAMPLE).unwrap().model.layers.len(), 24);
        for source in [
            format!("{EXAMPLE}\nloop = 2"),
            format!("{EXAMPLE}\nhidden = 1"),
            EXAMPLE.replace("hidden = 2048", "hidden = 0"),
            EXAMPLE.replace("decode = 1 1 false", "decode = 1 2 false"),
            EXAMPLE
                .replace("prefill = 1 1 false", "prefill = 2 1 false")
                .replace("decode = 1 1 false", "decode = 2 1 false")
                .replace("kv_heads = 2", "kv_heads = 1"),
        ] {
            assert!(parse(&source).is_err());
        }
    }
}
