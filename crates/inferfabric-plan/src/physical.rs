//! Backend-neutral physical explain IR. CPU v1 execution remains unchanged.
//! Incomplete contracts are explicit; this version never claims native execution.
use inferfabric_model::{Result, invalid};
use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;

pub const FORMAT: &str = "inferfabric.physical.v1";

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CostSource {
    Unknown,
    Analytical,
    Measured,
    Interpolated,
}
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Cost {
    pub source: CostSource,
    pub estimate_ns: Option<u64>,
    pub upper_ns: Option<u64>,
    /// Exact snapshot/sample identity, or analytical model revision. Never wall-clock time.
    pub evidence: Option<String>,
}
impl Default for Cost {
    fn default() -> Self {
        Self {
            source: CostSource::Unknown,
            estimate_ns: None,
            upper_ns: None,
            evidence: None,
        }
    }
}
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Location {
    Host,
    Device,
    Opaque,
}
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BoundaryKind {
    HostToDevice,
    DeviceToHost,
    Synchronize,
    DynamicAllocation,
    InternalLaunches,
    PersistentState,
}
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Boundary {
    pub kind: BoundaryKind,
    pub detail: String,
}
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Candidate {
    pub id: String,
    /// Immutable contract/source identity. This is not a compiled binary identity.
    pub contract_revision: String,
    pub numerical_contract: String,
    pub location: Location,
    pub eligible: bool,
    pub rejection: Option<String>,
    /// None means unresolved, never zero.
    pub workspace_bound_bytes: Option<u64>,
    pub capture_qualified: bool,
    pub boundaries: Vec<Boundary>,
    pub unresolved: Vec<String>,
    pub cost: Cost,
}
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Value {
    pub logical_node: usize,
    pub dtype: Option<String>,
    pub shape: Vec<u64>,
    pub layout: String,
    pub location: Location,
    /// Payload only; allocation metadata and persistent state are excluded.
    pub payload_bytes: Option<u64>,
    pub alignment: Option<u64>,
    pub arena_offset: Option<u64>,
    pub lifetime: [usize; 2],
}
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Collective {
    pub group: String,
    pub operation: String,
    pub dtype: String,
    pub elements_per_rank: u64,
    pub sequence: usize,
}
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Operation {
    pub name: String,
    pub logical_nodes: Vec<usize>,
    pub inputs: Vec<usize>,
    pub candidates: Vec<Candidate>,
    pub collective: Option<Collective>,
}
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Step {
    pub id: usize,
    pub operation: Operation,
    pub selected: usize,
    pub selection_reason: String,
    /// CPU calls may enqueue work to this stream, then synchronize internally.
    pub stream: usize,
    pub data_dependencies: Vec<usize>,
    pub order_dependencies: Vec<usize>,
    pub reuse_dependencies: Vec<usize>,
}
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Schedule {
    pub id: usize,
    pub steps: Vec<Step>,
    pub values: Vec<Value>,
    pub outputs: Vec<usize>,
    pub known_host_payload_peak_bytes: u64,
    pub memory_complete: bool,
    pub activation_policy: String,
    pub cost_complete: bool,
    pub estimated_serial_ns: Option<u64>,
}
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Rank {
    pub rank: usize,
    pub device: u32,
    pub schedule: usize,
}
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Group {
    pub id: String,
    pub ranks: Vec<usize>,
}
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Plan {
    pub format: String,
    pub name: String,
    pub target: String,
    pub logical_digest: String,
    pub policy: String,
    pub executable: bool,
    pub ranks: Vec<Rank>,
    pub groups: Vec<Group>,
    /// Identical rank schedules share a template; rank bindings remain explicit.
    pub schedules: Vec<Schedule>,
    pub limitations: Vec<String>,
}

fn check_cost(c: &Cost) -> Result<()> {
    if c.source == CostSource::Unknown {
        if c.estimate_ns.is_some() || c.upper_ns.is_some() {
            return Err(invalid("unknown cost has a numeric estimate"));
        }
    } else if c.estimate_ns.is_none()
        || c.upper_ns.is_none()
        || c.evidence.as_ref().is_none_or(|s| s.is_empty())
        || c.upper_ns < c.estimate_ns
    {
        return Err(invalid("incomplete or invalid cost provenance/bound"));
    }
    Ok(())
}
/// Compare only eligible candidates with complete estimates. Missing estimates
/// retain the stable baseline rather than rewarding unknown work as zero cost.
pub fn choose(candidates: &[Candidate]) -> Result<(usize, String)> {
    if candidates.is_empty() {
        return Err(invalid("no implementation candidates"));
    }
    let mut ids = BTreeSet::new();
    for c in candidates {
        check_cost(&c.cost)?;
        if c.id.is_empty()
            || !ids.insert(&c.id)
            || c.contract_revision.is_empty()
            || c.numerical_contract.is_empty()
            || (!c.eligible && c.rejection.as_ref().is_none_or(|s| s.is_empty()))
            || (c.eligible && c.rejection.is_some())
        {
            return Err(invalid("invalid candidate contract/eligibility"));
        }
    }
    let mut legal: Vec<_> = candidates
        .iter()
        .enumerate()
        .filter(|(_, c)| c.eligible)
        .collect();
    if legal.is_empty() {
        return Err(invalid("no legal implementation"));
    }
    if legal
        .iter()
        .any(|(_, c)| c.numerical_contract != legal[0].1.numerical_contract)
    {
        return Err(invalid("candidate changes numerical contract"));
    }
    let complete = legal.iter().all(|(_, c)| c.cost.upper_ns.is_some());
    legal.sort_by(|a, b| {
        if complete {
            a.1.cost
                .upper_ns
                .cmp(&b.1.cost.upper_ns)
                .then(a.1.id.cmp(&b.1.id))
        } else {
            a.1.id.cmp(&b.1.id)
        }
    });
    Ok((
        legal[0].0,
        if complete {
            "minimum conservative serial cost; stable implementation-ID tie break"
        } else {
            "deterministic legal baseline; costs unavailable, no latency comparison claimed"
        }
        .into(),
    ))
}

fn host_peak(values: &[Value], n: usize) -> Result<u64> {
    let mut total = 0u64;
    let mut peak = 0;
    let mut ends = vec![0u64; n + 1];
    for (i, v) in values.iter().enumerate() {
        let bytes = if v.location == Location::Host {
            v.payload_bytes.unwrap_or(0)
        } else {
            0
        };
        total = total
            .checked_add(bytes)
            .ok_or_else(|| invalid("host payload overflow"))?;
        peak = peak.max(total);
        ends[v.lifetime[1]] = ends[v.lifetime[1]]
            .checked_add(bytes)
            .ok_or_else(|| invalid("host payload overflow"))?;
        total = total
            .checked_sub(ends[i])
            .ok_or_else(|| invalid("invalid host lifetimes"))?;
    }
    Ok(peak)
}
/// Conservative serial planning: one output per operation, no aliasing or arena
/// reuse, stream zero. Other frontends/backends supply their own typed contracts.
pub fn serial_plan(
    name: String,
    target: String,
    logical_digest: String,
    devices: &[u32],
    operations: Vec<Operation>,
    mut values: Vec<Value>,
    outputs: Vec<usize>,
) -> Result<Plan> {
    let n = operations.len();
    if n == 0
        || n > 10000
        || values.len() != n
        || devices.is_empty()
        || devices.len() > 256
        || outputs.is_empty()
        || outputs.iter().any(|&i| i >= n)
    {
        return Err(invalid("physical graph bounds"));
    }
    for (i, v) in values.iter_mut().enumerate() {
        v.lifetime = [i, i];
        v.arena_offset = None;
    }
    let mut steps = Vec::new();
    let mut sum = Some(0u64);
    for (i, op) in operations.into_iter().enumerate() {
        if op.inputs.iter().any(|&j| j >= i) {
            return Err(invalid("non-topological physical dependency"));
        }
        for &j in &op.inputs {
            values[j].lifetime[1] = i;
        }
        let (selected, reason) = choose(&op.candidates)?;
        sum = match (sum, op.candidates[selected].cost.estimate_ns) {
            (Some(a), Some(b)) => Some(
                a.checked_add(b)
                    .ok_or_else(|| invalid("schedule cost overflow"))?,
            ),
            _ => None,
        };
        steps.push(Step {
            id: i,
            data_dependencies: op.inputs.clone(),
            order_dependencies: if i > 0 { vec![i - 1] } else { vec![] },
            reuse_dependencies: vec![],
            stream: 0,
            operation: op,
            selected,
            selection_reason: reason,
        });
    }
    for &i in &outputs {
        values[i].lifetime[1] = n;
    }
    let peak = host_peak(&values, n)?;
    let p=Plan{format:FORMAT.into(),name,target,logical_digest,policy:"serial-conservative-v1".into(),executable:false,ranks:devices.iter().enumerate().map(|(rank,&device)|Rank{rank,device,schedule:0}).collect(),groups:vec![Group{id:"all_ranks".into(),ranks:(0..devices.len()).collect()}],schedules:vec![Schedule{id:0,steps,values,outputs,known_host_payload_peak_bytes:peak,memory_complete:false,activation_policy:"existing ownership; no arena reuse or device memory budget claimed".into(),cost_complete:sum.is_some(),estimated_serial_ns:sum}],limitations:vec!["Inspection IR only: native executor still consumes the token program, not this physical plan".into(),"Backend calls retain opaque internals; boundary annotations are not independently executable launches".into(),"Device workspace, internal temporaries, persistent-state sizes and binary identities require backend preparation/probes".into(),"Known host payload peak excludes opaque handles, container capacity, allocator overhead and device memory".into(),"No measured hardware costs, overlap, reuse or capture qualification is implied".into()]};
    verify(&p)?;
    Ok(p)
}

pub fn verify(p: &Plan) -> Result<()> {
    if p.format != FORMAT
        || p.executable
        || p.policy != "serial-conservative-v1"
        || p.schedules.len() != 1
        || p.ranks.is_empty()
        || p.ranks.len() > 256
        || p.logical_digest.is_empty()
    {
        return Err(invalid("unsupported physical inspection plan"));
    }
    let mut devices = BTreeSet::new();
    for (i, r) in p.ranks.iter().enumerate() {
        if r.rank != i || r.schedule != 0 || !devices.insert(r.device) {
            return Err(invalid("rank binding mismatch"));
        }
    }
    if p.groups
        != vec![Group {
            id: "all_ranks".into(),
            ranks: (0..p.ranks.len()).collect(),
        }]
    {
        return Err(invalid("collective participants mismatch"));
    }
    let s = &p.schedules[0];
    let n = s.steps.len();
    if s.id != 0
        || n == 0
        || n > 10000
        || s.values.len() != n
        || s.outputs.is_empty()
        || s.outputs.iter().any(|&i| i >= n)
        || s.memory_complete
    {
        return Err(invalid("physical schedule bounds"));
    }
    let mut ends: Vec<_> = (0..n).collect();
    let mut sequence = 0;
    let mut sum = Some(0u64);
    let mut origins = BTreeSet::new();
    for (i, step) in s.steps.iter().enumerate() {
        let op = &step.operation;
        if step.id != i
            || step.stream != 0
            || step.data_dependencies != op.inputs
            || step.order_dependencies != if i > 0 { vec![i - 1] } else { vec![] }
            || !step.reuse_dependencies.is_empty()
            || op.inputs.iter().any(|&j| j >= i)
            || op.logical_nodes.is_empty()
            || op.logical_nodes.iter().any(|j| !origins.insert(*j))
        {
            return Err(invalid("invalid physical dependencies/provenance"));
        }
        let choice = choose(&op.candidates)?;
        if choice != (step.selected, step.selection_reason.clone()) {
            return Err(invalid("physical selection changed"));
        }
        for &j in &op.inputs {
            ends[j] = i;
        }
        let c = &op.candidates[step.selected];
        sum = match (sum, c.cost.estimate_ns) {
            (Some(a), Some(b)) => Some(
                a.checked_add(b)
                    .ok_or_else(|| invalid("schedule cost overflow"))?,
            ),
            _ => None,
        };
        if let Some(c) = &op.collective {
            if c.group != "all_ranks"
                || c.sequence != sequence
                || c.operation.is_empty()
                || c.dtype.is_empty()
                || c.elements_per_rank == 0
            {
                return Err(invalid("collective order/signature mismatch"));
            }
            sequence += 1;
        }
    }
    for &i in &s.outputs {
        ends[i] = n;
    }
    for (i, v) in s.values.iter().enumerate() {
        if v.lifetime != [i, ends[i]]
            || v.arena_offset.is_some()
            || !s.steps[i].operation.logical_nodes.contains(&v.logical_node)
            || v.alignment.is_some_and(|a| a == 0 || !a.is_power_of_two())
        {
            return Err(invalid("unproven storage lifetime/reuse"));
        }
    }
    for v in &s.values {
        if let Some(bytes) = v.payload_bytes {
            let unit = match v.dtype.as_deref() {
                Some("i8" | "u8") => 1u64,
                Some("bf16" | "f16") => 2,
                Some("f32" | "i32") => 4,
                Some("f64" | "i64") => 8,
                _ => return Err(invalid("known payload requires concrete dtype")),
            };
            let size = v.shape.iter().try_fold(unit, |a, &b| a.checked_mul(b));
            if v.shape.is_empty()
                || v.shape.len() > 8
                || v.shape.contains(&0)
                || size != Some(bytes)
            {
                return Err(invalid("payload size/shape mismatch"));
            }
        }
    }
    if s.known_host_payload_peak_bytes != host_peak(&s.values, n)?
        || s.estimated_serial_ns != sum
        || s.cost_complete != sum.is_some()
    {
        return Err(invalid("physical summary mismatch"));
    }
    Ok(())
}
