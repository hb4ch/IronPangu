//! Physical planning and source-independent CPU execution. No device dependency.
mod binary;
mod execute;
pub mod ir;
mod planner;
mod verify;
pub mod visualize;
pub use binary::{decode, encode, manifest};
pub use execute::{CompiledKernel, Executor, RunReport};
use inferfabric_model::{Result, invalid};
pub use planner::{Compilation, CompileOptions, compile, compile_with_options};
pub use verify::verify;
pub const VERSION: u32 = 1;
pub const TARGET: &str = "cpu-f32-v1";
pub const ALIGNMENT: usize = 64;
pub const MAX_BYTES: usize = 64 * 1024 * 1024;
pub const MAX_NODES: usize = 512;
pub(crate) fn elements(shape: &[usize]) -> Result<usize> {
    if shape.is_empty() || shape.len() > 8 || shape.contains(&0) {
        return Err(invalid("shape must have 1..8 positive dimensions"));
    }
    let n = shape
        .iter()
        .try_fold(1usize, |a, b| a.checked_mul(*b))
        .ok_or_else(|| invalid("shape overflow"))?;
    if n > MAX_BYTES / 4 {
        return Err(invalid("tensor exceeds CPU plan limit"));
    }
    Ok(n)
}
pub(crate) fn json<T: serde::Serialize>(value: &T) -> Result<Vec<u8>> {
    serde_json::to_vec_pretty(value).map_err(|e| invalid(e.to_string()))
}

/// Tune an already optimized plan without bypassing semantic or memory checks.
/// Validation is transactional; a failed override never mutates the caller's plan.
pub fn override_kernels(
    plan: &ir::PhysicalPlan,
    overrides: &std::collections::BTreeMap<String, ir::Kernel>,
) -> Result<ir::PhysicalPlan> {
    verify(plan)?;
    let mut result = plan.clone();
    for (name, kernel) in overrides {
        let step = result
            .steps
            .iter_mut()
            .find(|s| &s.name == name)
            .ok_or_else(|| invalid(format!("unknown physical step {name}")))?;
        if step.kernel.op() != kernel.op() {
            return Err(invalid("kernel override changes operation semantics"));
        }
        step.kernel = *kernel;
        step.selection_reason = "explicit checked physical-plan override".into();
    }
    verify(&result)?;
    Ok(result)
}
