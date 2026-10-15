//! Bounded, duplicate-rejecting JSON and confined checkpoint paths.
use super::MAX_METADATA;
use inferfabric_model::{Result, invalid};
use serde::{
    Deserialize, Deserializer,
    de::{self, MapAccess, SeqAccess, Visitor},
};
use serde_json::Value;
use std::{
    fmt,
    fs::File,
    io::Read,
    path::{Path, PathBuf},
};
pub fn read(path: &Path) -> Result<Vec<u8>> {
    read_bounded(path, MAX_METADATA)
}
pub fn read_bounded(path: &Path, limit: u64) -> Result<Vec<u8>> {
    let mut b = vec![];
    File::open(path)?.take(limit + 1).read_to_end(&mut b)?;
    if b.len() as u64 > limit {
        return Err(invalid(format!(
            "{} exceeds metadata limit {limit} bytes",
            path.display()
        )));
    }
    Ok(b)
}
pub fn local_file(root: &Path, name: &str) -> Result<PathBuf> {
    if name.is_empty() || name == "." || name == ".." || name.contains(['/', '\\', ':']) {
        return Err(invalid("checkpoint filenames must be local basenames"));
    }
    let path = root.join(name).canonicalize()?;
    if path.parent() != Some(root) {
        return Err(invalid("checkpoint symlink escapes root"));
    }
    Ok(path)
}
struct Unique(Value);
impl<'de> Deserialize<'de> for Unique {
    fn deserialize<D: Deserializer<'de>>(d: D) -> std::result::Result<Self, D::Error> {
        struct V;
        impl<'de> Visitor<'de> for V {
            type Value = Unique;
            fn expecting(&self, f: &mut fmt::Formatter) -> fmt::Result {
                f.write_str("JSON with unique object keys")
            }
            fn visit_map<M: MapAccess<'de>>(
                self,
                mut m: M,
            ) -> std::result::Result<Unique, M::Error> {
                let mut out = serde_json::Map::new();
                while let Some((k, v)) = m.next_entry::<String, Unique>()? {
                    if out.insert(k, v.0).is_some() {
                        return Err(de::Error::custom("duplicate JSON key"));
                    }
                }
                Ok(Unique(Value::Object(out)))
            }
            fn visit_seq<S: SeqAccess<'de>>(
                self,
                mut s: S,
            ) -> std::result::Result<Unique, S::Error> {
                let mut out = vec![];
                while let Some(v) = s.next_element::<Unique>()? {
                    out.push(v.0)
                }
                Ok(Unique(Value::Array(out)))
            }
            fn visit_str<E: de::Error>(self, v: &str) -> std::result::Result<Unique, E> {
                Ok(Unique(Value::String(v.into())))
            }
            fn visit_bool<E: de::Error>(self, v: bool) -> std::result::Result<Unique, E> {
                Ok(Unique(Value::Bool(v)))
            }
            fn visit_i64<E: de::Error>(self, v: i64) -> std::result::Result<Unique, E> {
                Ok(Unique(v.into()))
            }
            fn visit_u64<E: de::Error>(self, v: u64) -> std::result::Result<Unique, E> {
                Ok(Unique(v.into()))
            }
            fn visit_f64<E: de::Error>(self, v: f64) -> std::result::Result<Unique, E> {
                serde_json::Number::from_f64(v)
                    .map(|n| Unique(Value::Number(n)))
                    .ok_or_else(|| de::Error::custom("nonfinite JSON"))
            }
            fn visit_unit<E: de::Error>(self) -> std::result::Result<Unique, E> {
                Ok(Unique(Value::Null))
            }
        }
        d.deserialize_any(V)
    }
}
pub fn parse(b: &[u8]) -> Result<Value> {
    if b.len() as u64 > MAX_METADATA {
        return Err(invalid("metadata exceeds 16 MiB"));
    }
    serde_json::from_slice::<Unique>(b)
        .map(|v| v.0)
        .map_err(|e| invalid(e.to_string()))
}
