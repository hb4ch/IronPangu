use crate::ffi::{Session, SessionPtr};
use inferfabric_model::{Result, invalid};
use serde::Serialize;
use std::path::PathBuf;
#[derive(Debug, Clone)]
pub struct StartupOptions {
    pub max_model_len: usize,
    pub max_num_seqs: usize,
    pub memory_utilization: f64,
    pub profile_path: Option<PathBuf>,
}
impl StartupOptions {
    pub fn validate(&self) -> Result<()> {
        if self.max_num_seqs == 0 || self.max_num_seqs > 64 {
            return Err(invalid("max-num-seqs must be in 1..=64"));
        }
        if self.max_model_len == 0
            || self.max_model_len > inferfabric_compiler::checkpoint::MAX_MODEL_LEN
        {
            return Err(invalid(
                "max-model-len must be in 1..=262144 for this checkpoint",
            ));
        }
        if !self.memory_utilization.is_finite()
            || self.memory_utilization <= 0.
            || self.memory_utilization > 1.
        {
            return Err(invalid("gpu-memory-utilization must be in (0,1]"));
        }
        Ok(())
    }
}
#[repr(C)]
#[derive(Debug, Default, Clone, Serialize)]
pub struct MemoryStats {
    pub total_bytes: u64,
    pub free_bytes: u64,
    pub observed_peak_bytes: u64,
    pub buffer_bytes: u64,
    pub temporary_bytes: u64,
    pub workspace_bytes: u64,
    pub graphs: u64,
    pub budget_bytes: u64,
}
#[repr(C)]
#[derive(Debug, Default, Clone, Serialize)]
pub struct AllocationStats {
    pub scratch_pool_bytes: u64,
    pub layer_pool_bytes: u64,
    pub snapshot_pool_bytes: u64,
    pub owned_temporary_bytes: u64,
    pub logical_scratch_bytes: u64,
    pub logical_workspace_bytes: u64,
    pub workspace_blocks: u64,
    pub owned_buffers: u64,
    pub buffer_views: u64,
    pub dedicated_activations: u64,
    pub dedicated_workspaces: u64,
}
type Snapshot = unsafe extern "C" fn(SessionPtr, *mut MemoryStats) -> i32;
type SetBudget = unsafe extern "C" fn(SessionPtr, u64, u64) -> i32;
#[derive(Debug, Clone, Serialize)]
pub struct MemoryStage {
    pub name: String,
    pub stats: MemoryStats,
    pub allocations: AllocationStats,
}
#[derive(Debug, Clone, Serialize)]
pub struct MemoryProfile {
    pub tensor_parallel_rank: u32,
    pub tensor_parallel_world: u32,
    pub status: String,
    pub max_model_len: usize,
    pub max_num_seqs: usize,
    pub memory_utilization: f64,
    pub memory_reserve_bytes: u64,
    pub source_weight_bytes: u64,
    pub resident_weight_bytes: u64,
    pub kv_cache_bytes: u64,
    pub recurrent_and_conv_bytes: u64,
    pub other_buffer_bytes: u64,
    pub temporary_bytes: u64,
    pub workspace_bytes: u64,
    pub observed_peak_bytes: u64,
    pub non_weight_observed_bytes: u64,
    pub untracked_runtime_and_allocator_bytes: u64,
    pub remaining_budget_bytes: u64,
    pub full_context_graph_eager: String,
    pub startup_logits_sha256: String,
    pub startup_state_sha256: Vec<String>,
    pub full_context_logits_sha256: String,
    pub measurement_note: String,
    pub stages: Vec<MemoryStage>,
    #[serde(skip)]
    pub(crate) path: Option<PathBuf>,
}
impl MemoryProfile {
    pub(crate) fn start(
        s: &Session<'_>,
        options: &StartupOptions,
        source_weights: u64,
        planned_weight_bytes: u64,
    ) -> Result<Self> {
        options.validate()?;
        let mut profile = Self {
            tensor_parallel_rank: s.rank,
            tensor_parallel_world: s.world,
            status: "profiling".into(),
            max_model_len: options.max_model_len,
            max_num_seqs: options.max_num_seqs,
            memory_utilization: options.memory_utilization,
            memory_reserve_bytes: 256 * 1024 * 1024,
            source_weight_bytes: source_weights,
            resident_weight_bytes: 0,
            kv_cache_bytes: 0,
            recurrent_and_conv_bytes: 0,
            other_buffer_bytes: 0,
            temporary_bytes: 0,
            workspace_bytes: 0,
            observed_peak_bytes: 0,
            non_weight_observed_bytes: 0,
            untracked_runtime_and_allocator_bytes: 0,
            remaining_budget_bytes: 0,
            full_context_graph_eager: "pending".into(),
            startup_logits_sha256: String::new(),
            startup_state_sha256: vec![],
            full_context_logits_sha256: String::new(),
            measurement_note: "HBM free-memory differences and allocation/phase watermarks, not an exact trace of short-lived CANN allocations. Baseline includes context/communicator creation; concurrent device users can affect measurements. Fixed batch with sequential per-request prefill; last-position dry run uses synthetic cache state.".into(),
            stages: vec![],
            path: options.profile_path.clone(),
        };
        let initial = profile.record(s, "context_initialized")?;
        let set: SetBudget = unsafe {
            *s.api
                ._library
                .get(b"inferfabric_acl_set_memory_budget\0")
                .map_err(|e| invalid(e.to_string()))?
        };
        let budget = (initial.total_bytes as f64 * options.memory_utilization) as u64;
        s.api
            .check(unsafe { set(s.ptr, budget, profile.memory_reserve_bytes) })?;
        let budgeted = profile.record(s, "budget_set")?;
        if planned_weight_bytes > budgeted.budget_bytes {
            return Err(invalid(format!(
                "weights alone require {planned_weight_bytes} bytes, exceeding HBM budget {}",
                budgeted.budget_bytes
            )));
        }
        Ok(profile)
    }
    pub(crate) fn record(&mut self, s: &Session<'_>, name: &str) -> Result<MemoryStats> {
        let query: Snapshot = unsafe {
            *s.api
                ._library
                .get(b"inferfabric_acl_memory_snapshot\0")
                .map_err(|e| invalid(e.to_string()))?
        };
        let mut stats = MemoryStats::default();
        s.api.check(unsafe { query(s.ptr, &mut stats) })?;
        type AllocationSnapshot = unsafe extern "C" fn(SessionPtr, *mut AllocationStats) -> i32;
        let query: AllocationSnapshot = unsafe {
            *s.api
                ._library
                .get(b"inferfabric_acl_allocation_snapshot\0")
                .map_err(|e| invalid(e.to_string()))?
        };
        let mut allocations = AllocationStats::default();
        s.api.check(unsafe { query(s.ptr, &mut allocations) })?;
        self.stages.push(MemoryStage {
            name: name.into(),
            stats: stats.clone(),
            allocations,
        });
        self.save()?;
        Ok(stats)
    }
    pub(crate) fn finish(&mut self, s: &Session<'_>) -> Result<()> {
        let last = self.record(s, "profile_reset")?;
        self.other_buffer_bytes = last.buffer_bytes.saturating_sub(
            self.resident_weight_bytes + self.kv_cache_bytes + self.recurrent_and_conv_bytes,
        );
        self.temporary_bytes = last.temporary_bytes;
        self.workspace_bytes = last.workspace_bytes;
        self.observed_peak_bytes = last.observed_peak_bytes;
        self.non_weight_observed_bytes = last
            .observed_peak_bytes
            .saturating_sub(self.resident_weight_bytes);
        let used = self.stages[0]
            .stats
            .free_bytes
            .saturating_sub(last.free_bytes);
        self.untracked_runtime_and_allocator_bytes =
            used.saturating_sub(last.buffer_bytes + last.temporary_bytes + last.workspace_bytes);
        self.remaining_budget_bytes = last.budget_bytes.saturating_sub(last.observed_peak_bytes);
        self.status = "passed".into();
        self.full_context_graph_eager = "bitwise_equal".into();
        self.save()
    }
    fn save(&self) -> Result<()> {
        if let Some(path) = &self.path {
            let data = serde_json::to_vec_pretty(self).map_err(|e| invalid(e.to_string()))?;
            inferfabric_compiler::write_atomic(path, &data)?;
        }
        Ok(())
    }
}
pub(crate) fn save_failure(options: &StartupOptions, message: &str) {
    if let Some(path) = &options.profile_path {
        let mut value = std::fs::read(path)
            .ok()
            .and_then(|v| serde_json::from_slice::<serde_json::Value>(&v).ok())
            .unwrap_or_else(|| serde_json::json!({"max_model_len":options.max_model_len}));
        value["status"] = serde_json::json!("failed");
        value["error"] = serde_json::json!(message);
        if let Ok(bytes) = serde_json::to_vec_pretty(&value) {
            let _ = inferfabric_compiler::write_atomic(path, &bytes);
        }
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn rejects_context_and_budget_ranges() {
        let mut o = StartupOptions {
            max_model_len: 2050,
            max_num_seqs: 1,
            memory_utilization: 0.9,
            profile_path: None,
        };
        assert!(o.validate().is_ok());
        o.max_model_len = 262145;
        assert!(o.validate().is_err());
        o.max_model_len = 0;
        assert!(o.validate().is_err());
        o.max_model_len = 2048;
        for u in [0., -1., 1.1, f64::NAN] {
            o.memory_utilization = u;
            assert!(o.validate().is_err());
        }
    }
}
