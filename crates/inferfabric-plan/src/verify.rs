use crate::{ALIGNMENT, MAX_BYTES, MAX_NODES, TARGET, VERSION, elements, ir::*, planner::shape};
use inferfabric_model::{Result, invalid};
use std::collections::{BTreeMap, BTreeSet};
pub fn verify(p: &PhysicalPlan) -> Result<()> {
    if p.version != VERSION
        || p.target != TARGET
        || p.policy != "deterministic-cpu-v1"
        || p.alignment != ALIGNMENT
    {
        return Err(invalid(
            "unsupported physical plan version/target/policy/alignment",
        ));
    }
    if p.steps.len() > MAX_NODES
        || p.buffers.len() > MAX_NODES * 4
        || p.arena_bytes > MAX_BYTES
        || !p.arena_bytes.is_multiple_of(ALIGNMENT)
        || p.outputs.is_empty()
    {
        return Err(invalid("physical plan resource limit or empty outputs"));
    }
    let mut names = BTreeSet::new();
    let mut total = 0usize;
    for b in &p.buffers {
        let bytes = elements(&b.value.shape)? * 4;
        total = total
            .checked_add(bytes)
            .ok_or_else(|| invalid("plan byte count overflow"))?;
        if total > MAX_BYTES
            || b.value.name.is_empty()
            || b.value.name.len() > 256
            || !names.insert(&b.value.name)
        {
            return Err(invalid("plan byte limit or duplicate/invalid buffer name"));
        }
        match b.value.storage {
            Storage::Arena => {
                let offset = b.offset.ok_or_else(|| invalid("missing arena offset"))?;
                if b.value.initial.is_some()
                    || b.lifetime.is_none()
                    || !offset.is_multiple_of(ALIGNMENT)
                    || offset
                        .checked_add(bytes)
                        .is_none_or(|end| end > p.arena_bytes)
                {
                    return Err(invalid("invalid arena bounds, initializer or alignment"));
                }
            }
            Storage::Input | Storage::State | Storage::Constant => {
                if b.offset.is_some() || b.lifetime.is_some() {
                    return Err(invalid("owned storage cannot alias arena"));
                }
                if b.value.storage == Storage::Input {
                    if b.value.initial.is_some() {
                        return Err(invalid("input cannot have initializer"));
                    }
                } else if b
                    .value
                    .initial
                    .as_ref()
                    .is_none_or(|v| v.len() != bytes / 4 || v.iter().any(|x| !x.is_finite()))
                {
                    return Err(invalid("invalid owned initializer"));
                }
            }
        }
    }
    let mut producer = BTreeMap::new();
    let mut lives: Vec<Option<[usize; 2]>> = vec![None; p.buffers.len()];
    for (i, step) in p.steps.iter().enumerate() {
        if step.rank != 0 || step.stream != 0 {
            return Err(invalid("CPU executor supports rank 0 / stream 0 only"));
        }
        let output = p
            .buffers
            .get(step.output)
            .ok_or_else(|| invalid("output buffer ID out of range"))?;
        if output.value.storage != Storage::Arena
            || output.value.name != step.name
            || producer.contains_key(&step.output)
        {
            return Err(invalid("invalid or duplicate step output"));
        }
        for deps in [&step.dependencies, &step.reuse_dependencies] {
            if deps.iter().any(|&d| d >= i)
                || deps.iter().collect::<BTreeSet<_>>().len() != deps.len()
            {
                return Err(invalid("dependencies must be unique prior steps"));
            }
        }
        let mut shapes = vec![];
        let mut required = BTreeSet::new();
        for &id in &step.inputs {
            let input = p
                .buffers
                .get(id)
                .ok_or_else(|| invalid("input buffer ID out of range"))?;
            shapes.push(input.value.shape.as_slice());
            if input.value.storage == Storage::Arena {
                let src = producer
                    .get(&id)
                    .ok_or_else(|| invalid("read before production"))?;
                required.insert(*src);
                lives[id].as_mut().unwrap()[1] = i;
            }
        }
        if !required.iter().all(|d| step.dependencies.contains(d)) {
            return Err(invalid("missing data dependency"));
        }
        if shape(step.kernel.op(), &shapes)? != output.value.shape {
            return Err(invalid("kernel output shape mismatch"));
        }
        producer.insert(step.output, i);
        lives[step.output] = Some([i, i]);
    }
    if p.outputs.iter().collect::<BTreeSet<_>>().len() != p.outputs.len() {
        return Err(invalid("duplicate result ID"));
    }
    for (&target, &source) in &p.updates {
        let t = p
            .buffers
            .get(target)
            .ok_or_else(|| invalid("state target out of bounds"))?;
        let s = p
            .buffers
            .get(source)
            .ok_or_else(|| invalid("state source out of bounds"))?;
        if t.value.storage != Storage::State || t.value.shape != s.value.shape {
            return Err(invalid("invalid state commit binding"));
        }
    }
    for &id in p.outputs.iter().chain(p.updates.values()) {
        let b = p
            .buffers
            .get(id)
            .ok_or_else(|| invalid("result ID out of bounds"))?;
        if b.value.storage == Storage::Arena {
            let life = lives[id]
                .as_mut()
                .ok_or_else(|| invalid("unproduced result"))?;
            life[1] = p.steps.len();
        }
    }
    for (id, b) in p.buffers.iter().enumerate() {
        if b.lifetime != lives[id] || (b.value.storage == Storage::Arena && lives[id].is_none()) {
            return Err(invalid("forged or missing lifetime"));
        }
        let Some([first, last]) = b.lifetime else {
            continue;
        };
        for other in &p.buffers[..id] {
            let Some([ofirst, olast]) = other.lifetime else {
                continue;
            };
            let (a, c) = (b.offset.unwrap(), other.offset.unwrap());
            let overlaps =
                a < c + elements(&other.value.shape)? * 4 && c < a + elements(&b.value.shape)? * 4;
            if !overlaps {
                continue;
            }
            if first <= olast && ofirst <= last {
                return Err(invalid("simultaneously live buffers alias"));
            }
            let (before, after) = if last < ofirst {
                (last, ofirst)
            } else {
                (olast, first)
            };
            if !p.steps[after].reuse_dependencies.contains(&before) {
                return Err(invalid("missing memory reuse dependency"));
            }
        }
    }
    Ok(())
}
