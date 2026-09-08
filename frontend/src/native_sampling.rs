use vllm_engine_core_client::protocol::sampling::EngineCoreSamplingParams;
use vllm_llm::{Error, GenerateRequest, Result};
pub fn settings(p: &EngineCoreSamplingParams) -> inferfabric_native::Sampling {
    inferfabric_native::Sampling {
        temperature: p.temperature,
        top_k: p.top_k,
        top_p: p.top_p,
        min_p: p.min_p,
        repetition_penalty: p.repetition_penalty,
        frequency_penalty: p.frequency_penalty,
        presence_penalty: p.presence_penalty,
        min_tokens: p.min_tokens,
    }
}
pub fn defaults(dir: &std::path::Path) -> anyhow::Result<EngineCoreSamplingParams> {
    // Supplied Qwen3.5-2B model card: non-thinking, text-only recommendations.
    let mut value = serde_json::json!({"temperature":1.0,"top_k":20,"top_p":1.0,"min_p":0.0,"presence_penalty":2.0,"frequency_penalty":0.0,"repetition_penalty":1.0});
    let path = dir.join("generation_config.json");
    if path.exists() {
        let checkpoint: serde_json::Value = serde_json::from_slice(&std::fs::read(path)?)?;
        for key in [
            "temperature",
            "top_k",
            "top_p",
            "min_p",
            "presence_penalty",
            "frequency_penalty",
            "repetition_penalty",
        ] {
            if let Some(v) = checkpoint.get(key).filter(|v| !v.is_null()) {
                value[key] = v.clone();
            }
        }
    }
    let result: EngineCoreSamplingParams = serde_json::from_value(value)?;
    settings(&result).validate()?;
    Ok(result)
}
pub fn validate(req: &GenerateRequest) -> Result<()> {
    let p = &req.sampling_params;
    settings(p).validate().map_err(|e| Error::Unsupported {
        message: e.to_string(),
    })?;
    if req.mm_features.is_some()
        || req.lora_request.is_some()
        || req.priority != 0
        || req.data_parallel_rank.is_some()
        || req.cache_salt.is_some()
        || req.reasoning_parser_kwargs.is_some()
        || p.thinking_token_budget.is_some()
        || p.logprobs.is_some()
        || p.prompt_logprobs.is_some()
        || p.repetition_detection.is_some()
        || p.logit_bias.is_some()
        || p.allowed_token_ids.is_some()
        || p.bad_words_token_ids.is_some()
        || p.structured_outputs.is_some()
        || p.logprob_token_ids.is_some()
        || p.extra_args.is_some()
    {
        return Err(Error::Unsupported { message:"native sampling supports text, temperature, top-k/top-p/min-p, seed, repetition/presence/frequency penalties and min_tokens; advanced constraints/logprobs are not implemented".into() });
    }
    if p.min_tokens > p.max_tokens
        || p.stop_token_ids
            .iter()
            .chain(p.eos_token_id.iter())
            .any(|&id| id >= 248320)
    {
        return Err(Error::Unsupported {
            message: "invalid stop token or min_tokens".into(),
        });
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn model_card_defaults_are_supported() {
        let p = defaults(std::path::Path::new("fixtures/mock-tokenizer")).unwrap();
        assert_eq!(
            (p.temperature, p.top_k, p.top_p, p.presence_penalty),
            (1., 20, 1., 2.)
        );
        assert!(settings(&p).validate().is_ok());
    }
}
