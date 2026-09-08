use crate::{ALIGNMENT, MAX_BYTES, MAX_NODES, TARGET, VERSION, elements, ir::*, verify};
use inferfabric_model::{Result, invalid};
use std::collections::{BTreeMap, BTreeSet};
#[derive(Debug)]
pub struct Compilation {
    pub logical: LogicalGraph,
    pub typed: TypedGraph,
    pub optimized: TypedGraph,
    pub physical: PhysicalPlan,
}
pub(crate) fn shape(op: Op, inputs: &[&[usize]]) -> Result<Vec<usize>> {
    let arity = if op == Op::Relu { 1 } else { 2 };
    if inputs.len() != arity {
        return Err(invalid("operation arity mismatch"));
    }
    if op == Op::Matmul {
        if inputs[0].len() != 2 || inputs[1].len() != 2 || inputs[0][1] != inputs[1][0] {
            return Err(invalid("matmul requires [M,K] and [K,N]"));
        }
        let out = vec![inputs[0][0], inputs[1][1]];
        let work = elements(&out)?
            .checked_mul(inputs[0][1])
            .ok_or_else(|| invalid("matmul work overflow"))?;
        if work > 100_000_000 {
            return Err(invalid("matmul exceeds CPU invocation work limit"));
        }
        Ok(out)
    } else {
        if arity == 2 && inputs[0] != inputs[1] {
            return Err(invalid("elementwise shape mismatch; no implicit broadcast"));
        }
        Ok(inputs[0].to_vec())
    }
}
fn typed(graph: &LogicalGraph) -> Result<TypedGraph> {
    if graph.version != VERSION || graph.nodes.len() > MAX_NODES || graph.outputs.is_empty() {
        return Err(invalid(
            "invalid logical graph version, node count or outputs",
        ));
    }
    if graph.inputs.len() + graph.constants.len() + graph.states.len() + graph.nodes.len()
        > MAX_NODES * 4
    {
        return Err(invalid("too many logical values"));
    }
    let mut names = BTreeSet::new();
    for name in graph
        .inputs
        .iter()
        .map(|v| &v.name)
        .chain(graph.constants.iter().map(|v| &v.name))
        .chain(graph.states.iter().map(|v| &v.name))
        .chain(graph.nodes.iter().map(|v| &v.name))
    {
        if name.is_empty() || name.len() > 256 || !names.insert(name.clone()) {
            return Err(invalid(format!("invalid or duplicate value name: {name}")));
        }
    }
    let mut values = vec![];
    let mut ids = BTreeMap::new();
    for input in &graph.inputs {
        elements(&input.shape)?;
        ids.insert(input.name.clone(), values.len());
        values.push(Value {
            name: input.name.clone(),
            shape: input.shape.clone(),
            storage: Storage::Input,
            initial: None,
        });
    }
    for (storage, list) in [
        (Storage::Constant, &graph.constants),
        (Storage::State, &graph.states),
    ] {
        for v in list {
            if elements(&v.shape)? != v.data.len() || v.data.iter().any(|x| !x.is_finite()) {
                return Err(invalid(
                    "constant/state initializer shape or value mismatch",
                ));
            }
            ids.insert(v.name.clone(), values.len());
            values.push(Value {
                name: v.name.clone(),
                shape: v.shape.clone(),
                storage,
                initial: Some(v.data.clone()),
            });
        }
    }
    for node in &graph.nodes {
        if node.inputs.iter().any(|name| !names.contains(name)) {
            return Err(invalid(format!("unknown operand in {}", node.name)));
        }
    }
    let mut pending: BTreeMap<_, _> = graph
        .nodes
        .iter()
        .map(|node| (node.name.clone(), node))
        .collect();
    let mut nodes = vec![];
    while !pending.is_empty() {
        let name = pending
            .iter()
            .find(|(_, node)| node.inputs.iter().all(|name| ids.contains_key(name)))
            .map(|(name, _)| name.clone())
            .ok_or_else(|| invalid("cycle in logical graph"))?;
        let node = pending.remove(&name).unwrap();
        let inputs: Vec<_> = node.inputs.iter().map(|name| ids[name]).collect();
        let dims = shape(
            node.op,
            &inputs
                .iter()
                .map(|&i| values[i].shape.as_slice())
                .collect::<Vec<_>>(),
        )?;
        elements(&dims)?;
        let output = values.len();
        ids.insert(name.clone(), output);
        values.push(Value {
            name: name.clone(),
            shape: dims,
            storage: Storage::Arena,
            initial: None,
        });
        nodes.push(TypedNode {
            name,
            op: node.op,
            inputs,
            output,
        });
    }
    let resolve = |name: &String| {
        ids.get(name)
            .copied()
            .ok_or_else(|| invalid(format!("unknown output/update {name}")))
    };
    let outputs = graph
        .outputs
        .iter()
        .map(resolve)
        .collect::<Result<Vec<_>>>()?;
    if outputs.iter().collect::<BTreeSet<_>>().len() != outputs.len() {
        return Err(invalid("duplicate output"));
    }
    let mut updates = BTreeMap::new();
    for (target, source) in &graph.updates {
        let (target, source) = (resolve(target)?, resolve(source)?);
        if values[target].storage != Storage::State || values[target].shape != values[source].shape
        {
            return Err(invalid(
                "state update requires matching state target and value",
            ));
        }
        updates.insert(target, source);
    }
    let bytes = values.iter().try_fold(0usize, |total, value| {
        total
            .checked_add(elements(&value.shape)? * 4)
            .ok_or_else(|| invalid("logical storage overflow"))
    })?;
    if bytes > MAX_BYTES {
        return Err(invalid("logical graph exceeds CPU storage limit"));
    }
    Ok(TypedGraph {
        version: VERSION,
        name: graph.name.clone(),
        values,
        nodes,
        outputs,
        updates,
    })
}
fn fold(node: &TypedNode, values: &[Value]) -> Vec<f32> {
    let a = values[node.inputs[0]].initial.as_ref().unwrap();
    let b = node
        .inputs
        .get(1)
        .map(|&i| values[i].initial.as_ref().unwrap());
    (0..elements(&values[node.output].shape).unwrap())
        .map(|i| match node.op {
            Op::Add => a[i] + b.unwrap()[i],
            Op::Mul => a[i] * b.unwrap()[i],
            Op::Relu => a[i].max(0.),
            Op::Matmul => {
                let k = values[node.inputs[0]].shape[1];
                let n = values[node.output].shape[1];
                let mut sum = 0.;
                for j in 0..k {
                    sum += a[i / n * k + j] * b.unwrap()[j * n + i % n];
                }
                sum
            }
        })
        .collect()
}
pub fn compile(logical: LogicalGraph) -> Result<Compilation> {
    let typed = typed(&logical)?;
    let mut needed: BTreeSet<_> = typed
        .outputs
        .iter()
        .chain(typed.updates.values())
        .copied()
        .collect();
    let mut removed_nodes = vec![];
    for node in typed.nodes.iter().rev() {
        if needed.contains(&node.output) {
            needed.extend(&node.inputs);
        } else {
            removed_nodes.push(node.name.clone());
        }
    }
    let mut optimized = typed.clone();
    optimized.nodes.retain(|node| needed.contains(&node.output));
    let mut folded_nodes = vec![];
    let mut retained = vec![];
    for node in &optimized.nodes {
        let work = elements(&optimized.values[node.output].shape)?
            * if node.op == Op::Matmul {
                optimized.values[node.inputs[0]].shape[1]
            } else {
                1
            };
        if work <= 4096
            && node
                .inputs
                .iter()
                .all(|&i| optimized.values[i].storage == Storage::Constant)
        {
            let data = fold(node, &optimized.values);
            if data.iter().any(|v| !v.is_finite()) {
                return Err(invalid("nonfinite constant-folded result"));
            }
            optimized.values[node.output].storage = Storage::Constant;
            optimized.values[node.output].initial = Some(data);
            folded_nodes.push(node.name.clone());
        } else {
            retained.push(node.clone());
        }
    }
    optimized.nodes = retained;
    let runtime_values: BTreeSet<_> = optimized
        .outputs
        .iter()
        .chain(optimized.updates.values())
        .copied()
        .chain(
            optimized
                .nodes
                .iter()
                .flat_map(|n| n.inputs.iter().copied().chain(std::iter::once(n.output))),
        )
        .collect();
    // Compact value IDs so eliminated nodes have no physical storage or binding.
    let mut remap = BTreeMap::new();
    let mut compact = vec![];
    for (i, value) in optimized.values.iter().enumerate() {
        if matches!(value.storage, Storage::Input | Storage::State) || runtime_values.contains(&i) {
            remap.insert(i, compact.len());
            compact.push(value.clone());
        }
    }
    optimized.values = compact;
    for node in &mut optimized.nodes {
        node.inputs = node.inputs.iter().map(|i| remap[i]).collect();
        node.output = remap[&node.output];
    }
    optimized.outputs = optimized.outputs.iter().map(|i| remap[i]).collect();
    optimized.updates = optimized
        .updates
        .iter()
        .map(|(a, b)| (remap[a], remap[b]))
        .collect();
    let count = optimized.nodes.len();
    let mut lives = BTreeMap::new();
    let mut producers = BTreeMap::new();
    for (i, node) in optimized.nodes.iter().enumerate() {
        lives.insert(node.output, [i, i]);
        producers.insert(node.output, i);
        for input in &node.inputs {
            if let Some(life) = lives.get_mut(input) {
                life[1] = i;
            }
        }
    }
    for id in optimized.outputs.iter().chain(optimized.updates.values()) {
        if let Some(life) = lives.get_mut(id) {
            life[1] = count;
        }
    }
    let mut buffers: Vec<Buffer> = optimized
        .values
        .iter()
        .map(|value| Buffer {
            value: value.clone(),
            offset: None,
            lifetime: None,
        })
        .collect();
    let mut placed: Vec<usize> = vec![];
    let mut arena_bytes = 0;
    for node in &optimized.nodes {
        let id = node.output;
        let size = elements(&buffers[id].value.shape)? * 4;
        let aligned = size.div_ceil(ALIGNMENT) * ALIGNMENT;
        let life = lives[&id];
        let mut conflicts: Vec<_> = placed
            .iter()
            .copied()
            .filter(|j| lives[j][1] >= life[0])
            .collect();
        conflicts.sort_by_key(|&j| buffers[j].offset.unwrap());
        let mut offset = 0;
        for j in conflicts {
            let start = buffers[j].offset.unwrap();
            if offset + aligned <= start {
                break;
            }
            offset = offset.max(
                start + (elements(&buffers[j].value.shape)? * 4).div_ceil(ALIGNMENT) * ALIGNMENT,
            );
        }
        arena_bytes = arena_bytes.max(offset + aligned);
        buffers[id].offset = Some(offset);
        buffers[id].lifetime = Some(life);
        placed.push(id);
    }
    let mut steps = vec![];
    for (i, node) in optimized.nodes.iter().enumerate() {
        let deps: BTreeSet<_> = node
            .inputs
            .iter()
            .filter_map(|id| producers.get(id).copied())
            .collect();
        let out = &buffers[node.output];
        let begin = out.offset.unwrap();
        let end = begin + elements(&out.value.shape)? * 4;
        let reuse: BTreeSet<_> = buffers
            .iter()
            .filter_map(|old| {
                let [_, last] = old.lifetime?;
                let offset = old.offset?;
                (last < i
                    && offset < end
                    && begin < offset + elements(&old.value.shape).unwrap() * 4)
                    .then_some(last)
            })
            .collect();
        let kernel = match node.op {
            Op::Add => Kernel::AddF32V1,
            Op::Mul => Kernel::MulF32V1,
            Op::Relu => Kernel::ReluF32V1,
            Op::Matmul
                if elements(&out.value.shape)? * optimized.values[node.inputs[0]].shape[1]
                    >= 4096 =>
            {
                Kernel::MatmulBlockedF32V1
            }
            Op::Matmul => Kernel::MatmulF32V1,
        };
        steps.push(Step {
            name: node.name.clone(),
            kernel,
            inputs: node.inputs.clone(),
            output: node.output,
            rank: 0,
            stream: 0,
            dependencies: deps.into_iter().collect(),
            reuse_dependencies: reuse.into_iter().collect(),
            selection_reason: if kernel == Kernel::MatmulBlockedF32V1 {
                "static MAC count >= 4096: 16x16 output tiles"
            } else {
                "qualified deterministic CPU implementation; no timing-based cost model"
            }
            .into(),
        });
    }
    let physical = PhysicalPlan {
        version: VERSION,
        name: optimized.name.clone(),
        target: TARGET.into(),
        policy: "deterministic-cpu-v1".into(),
        alignment: ALIGNMENT,
        arena_bytes,
        buffers,
        steps,
        outputs: optimized.outputs.clone(),
        updates: optimized.updates.clone(),
        removed_nodes,
        folded_nodes,
    };
    verify(&physical)?;
    Ok(Compilation {
        logical,
        typed,
        optimized,
        physical,
    })
}
