#[derive(Clone, Debug)]
pub struct ProgramArgs {
    pub port: u16,
    pub max_model_len: Option<usize>,
    pub max_num_seqs: Option<usize>,
    pub max_num_batched_tokens: Option<usize>,
    pub prefix_cache_slots: Option<usize>,
    pub prefix_cache_namespace: Option<String>,
}
impl ProgramArgs {
    pub fn parse(args: &[String]) -> anyhow::Result<Self> {
        let mut result = Self {
            port: 18081,
            max_model_len: None,
            max_num_seqs: None,
            max_num_batched_tokens: None,
            prefix_cache_slots: None,
            prefix_cache_namespace: None,
        };
        let mut args = args.iter().peekable();
        if args.peek().is_some_and(|a| !a.starts_with("--")) {
            result.port = args.next().unwrap().parse()?;
        }
        let mut used = std::collections::BTreeSet::new();
        while let Some(option) = args.next() {
            let (key, inline) = option
                .split_once('=')
                .map_or((option.as_str(), None), |(k, v)| (k, Some(v)));
            anyhow::ensure!(used.insert(key), "duplicate option {key}");
            if key=="--prefix-cache-namespace" {
                let value=inline.or_else(||args.next().map(String::as_str)).ok_or_else(||anyhow::anyhow!("missing cache namespace"))?;
                anyhow::ensure!(!value.is_empty() && value.len()<=256,"cache namespace must contain 1-256 bytes");
                result.prefix_cache_namespace=Some(value.to_owned());
                continue;
            }
            let target = match key {
                "--max-model-len" => &mut result.max_model_len,
                "--max-num-seqs" => &mut result.max_num_seqs,
                "--max-num-batched-tokens" => &mut result.max_num_batched_tokens,
                "--prefix-cache-slots" => &mut result.prefix_cache_slots,
                _ => anyhow::bail!("unknown program option {key}"),
            };
            let value = inline
                .or_else(|| args.next().map(String::as_str))
                .ok_or_else(|| anyhow::anyhow!("missing value for {key}"))?;
            let count: usize = value.parse()?;
            anyhow::ensure!(count > 0, "{key} must be positive");
            *target = Some(count);
        }
        Ok(result)
    }
    pub fn resolve(
        &self,
        resident_capacity: usize,
        compiled_context: usize,
    ) -> anyhow::Result<(usize, usize, usize)> {
        let residents = self.max_num_seqs.unwrap_or(resident_capacity);
        let context = self.max_model_len.unwrap_or(compiled_context);
        let budget = self.max_num_batched_tokens.unwrap_or(residents);
        let snapshots=self.prefix_cache_slots.unwrap_or(0);
        anyhow::ensure!(self.prefix_cache_slots.is_some()==self.prefix_cache_namespace.is_some(),
            "prefix cache requires both --prefix-cache-slots and --prefix-cache-namespace");
        anyhow::ensure!(snapshots==0 || self.max_num_seqs.is_some(),"prefix cache requires explicit --max-num-seqs");
        anyhow::ensure!(residents.checked_add(snapshots).is_some_and(|n|n<=resident_capacity),
            "active plus snapshot residents exceed compiled capacity");
        anyhow::ensure!(
            residents > 0 && residents <= resident_capacity,
            "max-num-seqs exceeds compiled resident capacity"
        );
        anyhow::ensure!(
            context > 0 && context <= compiled_context,
            "max-model-len exceeds compiled context"
        );
        anyhow::ensure!(budget > 0, "max-num-batched-tokens must be positive");
        Ok((residents, context, budget))
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    fn parse(s: &str) -> anyhow::Result<ProgramArgs> {
        ProgramArgs::parse(&s.split_whitespace().map(str::to_owned).collect::<Vec<_>>())
    }
    #[test]
    fn prefix_capacity_is_additional_and_sharing_is_explicit() {
        assert_eq!(parse("--max-num-seqs 1 --prefix-cache-slots 1 --prefix-cache-namespace test").unwrap().resolve(2,4096).unwrap(),(1,4096,1));
        for input in ["--prefix-cache-slots 1 --prefix-cache-namespace test",
            "--max-num-seqs 2 --prefix-cache-slots 1 --prefix-cache-namespace test",
            "--max-num-seqs 1 --prefix-cache-slots 1","--prefix-cache-namespace test"] {
            assert!(parse(input).unwrap().resolve(2,4096).is_err());
        }
    }
    #[test]
    fn program_limits_are_bounded_by_compiled_resources() {
        assert_eq!(parse("").unwrap().resolve(2, 512).unwrap(), (2, 512, 2));
        assert_eq!(
            parse("18085 --max-num-seqs=1 --max-model-len 128 --max-num-batched-tokens 8")
                .unwrap()
                .resolve(2, 512)
                .unwrap(),
            (1, 128, 8)
        );
        for input in [
            "--max-num-seqs 0",
            "--max-num-batched-tokens",
            "--max-model-len -1",
            "--unknown 1",
            "--max-num-seqs 1 --max-num-seqs=2",
        ] {
            assert!(parse(input).is_err());
        }
        assert!(parse("--max-num-seqs 3").unwrap().resolve(2, 512).is_err());
        assert!(
            parse("--max-model-len 65536")
                .unwrap()
                .resolve(2, 512)
                .is_err()
        );
    }
}
