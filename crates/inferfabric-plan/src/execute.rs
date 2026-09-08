use crate::{elements, ir::*, verify};
use inferfabric_model::{Result, invalid};
use serde::Serialize;
use std::collections::BTreeMap;
#[derive(Debug, Serialize)]
pub struct RunReport {
    pub outputs: BTreeMap<String, Vec<f32>>,
    pub state: BTreeMap<String, Vec<f32>>,
    pub executed_steps: Vec<String>,
    pub arena_bytes: usize,
}
#[derive(Clone)]
#[repr(C, align(64))]
struct Line([f32; 16]);
struct Arena(Vec<Line>);
impl std::ops::Index<usize> for Arena {
    type Output = f32;
    fn index(&self, index: usize) -> &f32 {
        &self.0[index / 16].0[index % 16]
    }
}
impl std::ops::IndexMut<usize> for Arena {
    fn index_mut(&mut self, index: usize) -> &mut f32 {
        &mut self.0[index / 16].0[index % 16]
    }
}
/// Owns stable CPU storage across invocations; consumes only a verified physical plan.
pub struct Executor {
    plan: PhysicalPlan,
    arena: Arena,
    owned: BTreeMap<usize, Vec<f32>>,
}
impl Executor {
    pub fn new(plan: PhysicalPlan) -> Result<Self> {
        verify(&plan)?;
        let owned = plan
            .buffers
            .iter()
            .enumerate()
            .filter_map(|(i, b)| b.value.initial.clone().map(|v| (i, v)))
            .collect();
        let arena = Arena(vec![Line([0.; 16]); plan.arena_bytes / 64]);
        Ok(Self { plan, arena, owned })
    }
    fn get(&self, id: usize, index: usize, inputs: &BTreeMap<String, Vec<f32>>) -> f32 {
        let b = &self.plan.buffers[id];
        match b.value.storage {
            Storage::Arena => self.arena[b.offset.unwrap() / 4 + index],
            Storage::Input => inputs[&b.value.name][index],
            Storage::State | Storage::Constant => self.owned[&id][index],
        }
    }
    fn collect(&self, id: usize, inputs: &BTreeMap<String, Vec<f32>>) -> Vec<f32> {
        (0..elements(&self.plan.buffers[id].value.shape).unwrap())
            .map(|i| self.get(id, i, inputs))
            .collect()
    }
    pub fn run(&mut self, inputs: &BTreeMap<String, Vec<f32>>) -> Result<RunReport> {
        let expected: Vec<_> = self
            .plan
            .buffers
            .iter()
            .filter(|b| b.value.storage == Storage::Input)
            .collect();
        if inputs.len() != expected.len()
            || expected.iter().any(|b| {
                inputs.get(&b.value.name).is_none_or(|v| {
                    v.len() != elements(&b.value.shape).unwrap() || v.iter().any(|x| !x.is_finite())
                })
            })
        {
            return Err(invalid(
                "input names, sizes or finite values do not match binary bindings",
            ));
        }
        // Schedule and kernel choices are consumed verbatim; no compiler or DSL access.
        for step in &self.plan.steps {
            let dst = self.plan.buffers[step.output].offset.unwrap() / 4;
            let a = step.inputs[0];
            let b = step.inputs.get(1).copied();
            let count = elements(&self.plan.buffers[step.output].value.shape)?;
            match step.kernel {
                Kernel::AddF32V1 | Kernel::MulF32V1 | Kernel::ReluF32V1 => {
                    for i in 0..count {
                        let x = self.get(a, i, inputs);
                        let value = match step.kernel {
                            Kernel::AddF32V1 => x + self.get(b.unwrap(), i, inputs),
                            Kernel::MulF32V1 => x * self.get(b.unwrap(), i, inputs),
                            _ => x.max(0.),
                        };
                        self.arena[dst + i] = value;
                    }
                }
                Kernel::MatmulF32V1 | Kernel::MatmulBlockedF32V1 => {
                    let dims = &self.plan.buffers[step.output].value.shape;
                    let (m, n, k) = (dims[0], dims[1], self.plan.buffers[a].value.shape[1]);
                    let tile = if step.kernel == Kernel::MatmulBlockedF32V1 {
                        16
                    } else {
                        1
                    };
                    for row in (0..m).step_by(tile) {
                        for col in (0..n).step_by(tile) {
                            for i in row..(row + tile).min(m) {
                                for j in col..(col + tile).min(n) {
                                    let mut sum = 0.;
                                    for z in 0..k {
                                        sum += self.get(a, i * k + z, inputs)
                                            * self.get(b.unwrap(), z * n + j, inputs);
                                    }
                                    self.arena[dst + i * n + j] = sum;
                                }
                            }
                        }
                    }
                }
            }
            if (dst..dst + count).any(|i| !self.arena[i].is_finite()) {
                return Err(invalid(format!(
                    "nonfinite result at {}; state not committed",
                    step.name
                )));
            }
        }
        let outputs = self
            .plan
            .outputs
            .iter()
            .map(|&id| {
                (
                    self.plan.buffers[id].value.name.clone(),
                    self.collect(id, inputs),
                )
            })
            .collect();
        // Gather all new states before changing any: swaps and self-updates are simultaneous.
        let updates: Vec<_> = self
            .plan
            .updates
            .iter()
            .map(|(&dst, &src)| (dst, self.collect(src, inputs)))
            .collect();
        for (dst, data) in updates {
            self.owned.get_mut(&dst).unwrap().copy_from_slice(&data);
        }
        let state = self
            .plan
            .buffers
            .iter()
            .enumerate()
            .filter(|(_, b)| b.value.storage == Storage::State)
            .map(|(id, b)| (b.value.name.clone(), self.owned[&id].clone()))
            .collect();
        Ok(RunReport {
            outputs,
            state,
            executed_steps: self.plan.steps.iter().map(|s| s.name.clone()).collect(),
            arena_bytes: self.plan.arena_bytes,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn arena_base_and_slots_meet_physical_alignment() {
        let arena = Arena(vec![Line([0.; 16]); 3]);
        assert_eq!(arena.0.as_ptr() as usize % crate::ALIGNMENT, 0);
        assert_eq!(std::mem::size_of::<Line>(), crate::ALIGNMENT);
        assert_eq!(
            &arena[16] as *const f32 as usize - &arena[0] as *const f32 as usize,
            64
        );
    }
}
