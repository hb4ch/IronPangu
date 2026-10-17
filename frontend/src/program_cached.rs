use super::program_worker::ResidentEngine;
use inferfabric_ir::{
    arithmetic_history::ArithmeticHistory,
    serving::{Manifest, PrefixIdentity, Variant},
};
use inferfabric_model::{Result, invalid};
use inferfabric_native::{prefix_cache::PrefixCache, program::Engine};
use vllm_llm::GenerateRequest;

pub(super) struct CachedProgram {
    engine: Engine,
    cache: Option<PrefixCache>,
    identity: PrefixIdentity,
    variants: Vec<Variant>,
    alignment: u32,
    budget: u32,
    hits: u64,
    restored: u64,
    published: u64,
}
impl CachedProgram {
    pub fn new(
        engine: Engine,
        manifest: &Manifest,
        options: &crate::program_args::ProgramArgs,
        residents: usize,
        context: usize,
        budget: usize,
        tokenizer_hash: String,
    ) -> Result<Self> {
        let identity = PrefixIdentity {
            engine: manifest.identity.clone(),
            numerical_class: "completed-invocation-history-v1".into(),
            tokenizer_template_sha256: tokenizer_hash,
            tenant_namespace: options.prefix_cache_namespace.clone().unwrap_or_default(),
            adapter_sha256: None,
        };
        let cache = if let Some(slots) = options.prefix_cache_slots {
            if !manifest.shared_pages || manifest.states.is_empty()
                || manifest.states.iter().any(|state| !state.checkpoint_supported)
            {
                return Err(invalid("prefix cache requires complete executable checkpoint capabilities"));
            }
            Some(PrefixCache::new(
                &engine,
                identity.clone(),
                residents as u32,
                slots as u32,
                context as u32,
            )?)
        } else {
            None
        };
        let alignment = engine.checkpoint_alignment()?;
        Ok(Self {
            engine,
            cache,
            identity,
            variants: manifest.variants.clone(),
            alignment,
            budget: budget.min(u32::MAX as usize) as u32,
            hits: 0,
            restored: 0,
            published: 0,
        })
    }
    fn boundary(&self, tokens: u32) -> u32 {
        tokens.saturating_sub(1) / self.alignment * self.alignment
    }
    fn namespace(&self, request: &GenerateRequest) -> Result<PrefixIdentity> {
        if request.lora_request.is_some() || request.mm_features.is_some() {
            return Err(invalid(
                "prefix cache requires a supported text-only model identity",
            ));
        }
        let mut identity = self.identity.clone();
        let bytes = serde_json::to_vec(&(&identity.tenant_namespace, &request.cache_salt))
            .map_err(|e| invalid(e.to_string()))?;
        identity.tenant_namespace = inferfabric_compiler::checkpoint::sha(&bytes);
        Ok(identity)
    }
}
impl ResidentEngine for CachedProgram {
    fn prefill_capacity(&self) -> u32 {
        self.engine.prefill_capacity()
    }
    fn prefill_chunk(&self, limit: u32) -> u32 {
        self.engine.prefill_chunk(limit)
    }
    fn prefill(&mut self, tokens: &[u32], position: u32, sample: bool) -> Result<Option<u32>> {
        self.engine.prefill(tokens, position, sample)
    }
    fn token(&mut self, token: u32, position: u32, sample: bool) -> Result<Option<u32>> {
        self.engine.token(token, position, sample)
    }
    fn reset(&mut self, slot: u32) -> Result<()> {
        self.engine.reset_resident(slot)
    }
    fn admit(&mut self, slot: u32, tokens: u32) -> Result<()> {
        self.engine.prepare_resident(slot, tokens)
    }
    fn release(&mut self, slot: u32) -> Result<()> {
        self.engine.release_resident(slot)
    }
    fn select(&mut self, slot: u32) -> Result<()> {
        self.engine.select_resident(slot)
    }
    fn prefill_limit(&self, position: u32, prompt: u32) -> u32 {
        let boundary = self.boundary(prompt);
        if self.cache.is_some() && position < boundary {
            boundary - position
        } else {
            prompt - position
        }
    }
    fn restore_prefix(&mut self, slot: u32, request: &GenerateRequest) -> Result<u32> {
        if self.cache.is_none() {
            return Ok(0);
        }
        let identity = self.namespace(request)?;
        let boundary = self.boundary(request.prompt_token_ids.len() as u32);
        // Exact prospective arithmetic; dynamic token-budget differences may
        // reduce hits, but never grant compatibility from final shape alone.
        let mut history = ArithmeticHistory::default();
        while history.tokens() < boundary {
            let count = self
                .engine
                .prefill_chunk((boundary - history.tokens()).min(self.budget));
            let variant = self
                .variants
                .iter()
                .find(|v| v.max_query_tokens == count && v.max_sequences == 1)
                .ok_or_else(|| invalid("prefix arithmetic variant missing"))?;
            history.append_completed(variant, history.tokens(), count)?;
        }
        let restored = self.cache.as_mut().unwrap().restore(
            &mut self.engine,
            &identity,
            slot,
            &request.prompt_token_ids,
            &history,
        )?;
        if restored > 0 {
            self.hits += 1;
            self.restored += u64::from(restored);
        }
        Ok(restored)
    }
    fn checkpoint(&mut self, slot: u32, request: &GenerateRequest, position: u32) -> Result<()> {
        if self.cache.is_none()
            || position == 0
            || position != self.boundary(request.prompt_token_ids.len() as u32)
        {
            return Ok(());
        }
        let identity = self.namespace(request)?;
        if self.cache.as_mut().unwrap().publish(
            &mut self.engine,
            &identity,
            slot,
            &request.prompt_token_ids[..position as usize],
        )? {
            self.published += 1;
        }
        Ok(())
    }
    fn clear_prefix_cache(&mut self) -> Result<()> {
        if let Some(cache) = &mut self.cache {
            cache.clear(&mut self.engine)?;
            eprintln!(
                "prefix cache stopped hits={} restored_tokens={} published={} remaining=0",
                self.hits, self.restored, self.published
            );
        }
        Ok(())
    }
}
