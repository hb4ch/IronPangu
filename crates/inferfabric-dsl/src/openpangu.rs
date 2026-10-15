//! Explicit topology DSL; model dimensions still come from the validated checkpoint.
use inferfabric_model::{OpenPanguParallel, Result, invalid};
use std::collections::BTreeMap;
pub fn parse(source: &str) -> Result<OpenPanguParallel> {
    let mut fields = BTreeMap::new();
    for raw in source.lines() {
        let line = raw.split('#').next().unwrap().trim();
        if line.is_empty() {
            continue;
        }
        let (key, value) = line
            .split_once('=')
            .ok_or_else(|| invalid("expected key = value"))?;
        if fields.insert(key.trim(), value.trim()).is_some() {
            return Err(invalid("duplicate OpenPangu DSL field"));
        }
    }
    if fields.remove("contract") != Some("openpangu_v2_tp_ep_v1") {
        return Err(invalid("expected openpangu_v2_tp_ep_v1 contract"));
    }
    fn take<'a>(f: &mut BTreeMap<&str, &'a str>, k: &str) -> Result<&'a str> {
        f.remove(k).ok_or_else(|| invalid(format!("missing {k}")))
    }
    fn n(f: &mut BTreeMap<&str, &str>, k: &str) -> Result<usize> {
        take(f, k)?
            .parse()
            .map_err(|_| invalid(format!("invalid {k}")))
    }
    let tp = n(&mut fields, "tp")?;
    let ep = n(&mut fields, "ep")?;
    let devices = take(&mut fields, "devices")?
        .split_whitespace()
        .map(|v| v.parse().map_err(|_| invalid("invalid device")))
        .collect::<Result<Vec<u32>>>()?;
    let s = OpenPanguParallel {
        tp,
        ep,
        devices,
        max_num_seqs: n(&mut fields, "max_num_seqs")?,
        max_num_batched_tokens: n(&mut fields, "max_num_batched_tokens")?,
        max_model_len: n(&mut fields, "max_model_len")?,
    };
    if !fields.is_empty() {
        return Err(invalid("unknown OpenPangu DSL field"));
    }
    s.validate()?;
    Ok(s)
}
#[cfg(test)]
mod tests {
    use super::*;
    const SOURCE: &str = "contract = openpangu_v2_tp_ep_v1\ntp = 8\nep = 8\ndevices = 0 1 2 3 4 5 6 7\nmax_num_seqs = 4\nmax_num_batched_tokens = 16\nmax_model_len = 2048\n";
    #[test]
    fn topology_and_budget_contract() {
        let s = parse(SOURCE).unwrap();
        assert_eq!(s.devices.len(), 8);
        for bad in [
            SOURCE.replace("ep = 8", "ep = 4"),
            SOURCE.replace("6 7", "6 6"),
            SOURCE.replace("seqs = 4", "seqs = 0"),
            format!("{SOURCE}tp = 8\n"),
            format!("{SOURCE}pp = 2\n"),
        ] {
            assert!(parse(&bad).is_err());
        }
    }
}
