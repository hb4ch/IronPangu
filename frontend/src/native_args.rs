use std::path::PathBuf;
pub struct NativeArgs {
    pub startup: pangu_native::StartupOptions,
    pub port: u16,
    pub profile_only: bool,
    pub max_num_batched_tokens: usize,
}
impl NativeArgs {
    pub fn parse(args: &[String], default_context: usize, device: i32) -> anyhow::Result<Self> {
        let mut options = Self {
            startup: pangu_native::StartupOptions {
                max_model_len: default_context,
                max_num_seqs: 4,
                memory_utilization: 0.9,
                profile_path: Some(PathBuf::from(format!(
                    "native-memory-profile-device{device}.json"
                ))),
            },
            port: 8000,
            profile_only: false,
            max_num_batched_tokens: 1024,
        };
        let mut used = std::collections::BTreeSet::new();
        let mut index = 0;
        if args.first().is_some_and(|a| !a.starts_with("--")) {
            options.port = args[0].parse()?;
            index += 1;
        }
        while index < args.len() {
            let current = &args[index];
            index += 1;
            let (key, inline) = current
                .split_once('=')
                .map_or((current.as_str(), None), |(k, v)| (k, Some(v)));
            anyhow::ensure!(used.insert(key), "duplicate option {key}");
            if key == "--profile-only" {
                anyhow::ensure!(inline.is_none(), "profile-only takes no value");
                options.profile_only = true;
                continue;
            }
            anyhow::ensure!(
                [
                    "--max-model-len",
                    "--gpu-memory-utilization",
                    "--memory-profile",
                    "--max-num-seqs",
                    "--max-num-batched-tokens"
                ]
                .contains(&key),
                "unknown native option {key}"
            );
            let value = if let Some(value) = inline {
                value
            } else {
                let value = args
                    .get(index)
                    .ok_or_else(|| anyhow::anyhow!("missing value for {key}"))?;
                index += 1;
                value
            };
            match key {
                "--max-num-seqs" => options.startup.max_num_seqs = value.parse()?,
                "--max-num-batched-tokens" => options.max_num_batched_tokens = value.parse()?,
                "--max-model-len" => options.startup.max_model_len = value.parse()?,
                "--gpu-memory-utilization" => options.startup.memory_utilization = value.parse()?,
                "--memory-profile" => {
                    anyhow::ensure!(!value.is_empty(), "empty memory profile path");
                    options.startup.profile_path = Some(value.into());
                }
                _ => unreachable!(),
            }
        }
        options.startup.validate()?;
        anyhow::ensure!(
            options.max_num_batched_tokens > 0,
            "max-num-batched-tokens must be positive"
        );
        Ok(options)
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    fn parse(s: &str) -> anyhow::Result<NativeArgs> {
        NativeArgs::parse(
            &s.split_whitespace().map(str::to_owned).collect::<Vec<_>>(),
            8192,
            0,
        )
    }
    #[test]
    fn context_and_profile_cli() {
        let o = parse("18081 --max-model-len 2050 --gpu-memory-utilization=0.75 --profile-only")
            .unwrap();
        assert_eq!(o.startup.max_model_len, 2050);
        assert_eq!(o.port, 18081);
        assert!(o.profile_only);
        assert_eq!(parse("").unwrap().startup.max_model_len, 8192);
        for bad in [
            "--max-model-len 0",
            "--max-num-seqs 0",
            "--max-num-seqs 65",
            "--max-num-batched-tokens 0",
            "--max-model-len 262145",
            "--gpu-memory-utilization NaN",
            "--profile-only=true",
            "--unknown 1",
            "--max-model-len 12 --max-model-len 13",
        ] {
            assert!(parse(bad).is_err(), "{bad}");
        }
    }
}
