//! Export verified CPU plan semantics for the explicit Ascend qualification probe.
//! This is a test adapter, not an Ascend executable bundle or serving backend.
use inferfabric_plan::{
    Executor, decode,
    ir::{Op, Storage},
};
use std::{collections::BTreeMap, fmt::Write, fs};
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let a: Vec<String> = std::env::args().collect();
    if a.len() != 4 {
        return Err("usage: export_npu_fixture PLAN.ifplan INPUTS.json FIXTURE.txt".into());
    }
    let p = decode(&fs::read(&a[1])?)?;
    if p.outputs.iter().any(|id| p.updates.contains_key(id)) {
        return Err("qualification adapter cannot observe a state output before its commit".into());
    }
    let cases: Vec<BTreeMap<String, Vec<f32>>> = serde_json::from_slice(&fs::read(&a[2])?)?;
    if cases.is_empty() || cases.len() > 1024 {
        return Err("expected 1..1024 cases".into());
    }
    let mut cpu = Executor::new(p.clone())?;
    let mut s = format!("IF_NPU_QUALIFY_1\n{} {}\n", p.arena_bytes, p.buffers.len());
    for b in &p.buffers {
        let kind = match b.value.storage {
            Storage::Input => 0,
            Storage::Constant => 1,
            Storage::State => 2,
            Storage::Arena => 3,
        };
        write!(
            s,
            "{kind} {} {} ",
            b.offset.map(|n| n as i64).unwrap_or(-1),
            b.value.shape.len()
        )?;
        for d in &b.value.shape {
            write!(s, "{d} ")?
        }
        let v = b.value.initial.as_deref().unwrap_or(&[]);
        write!(s, "{} ", v.len())?;
        for x in v {
            write!(s, "{x:?} ")?
        }
        s.push('\n');
    }
    writeln!(s, "{}", p.steps.len())?;
    for step in &p.steps {
        let op = match step.kernel.op() {
            Op::Add => 1,
            Op::Mul => 2,
            Op::Relu => 3,
            Op::Matmul => 4,
        };
        writeln!(
            s,
            "{op} {} {} {}",
            step.inputs[0],
            step.inputs.get(1).copied().unwrap_or(0),
            step.output
        )?;
    }
    writeln!(s, "{}", p.updates.len())?;
    for (d, v) in &p.updates {
        writeln!(s, "{d} {v}")?
    }
    writeln!(s, "{}", cases.len())?;
    for case in cases {
        let report = cpu.run(&case)?;
        writeln!(s, "{}", case.len())?;
        for (name, v) in &case {
            let i = p
                .buffers
                .iter()
                .position(|b| b.value.name == *name)
                .unwrap();
            write!(s, "{i} {} ", v.len())?;
            for x in v {
                write!(s, "{x:?} ")?
            }
            s.push('\n');
        }
        writeln!(s, "{}", report.outputs.len() + report.state.len())?;
        for (name, v) in report.outputs.iter().chain(report.state.iter()) {
            let i = p
                .buffers
                .iter()
                .position(|b| b.value.name == *name)
                .unwrap();
            write!(s, "{i} {} ", v.len())?;
            for x in v {
                write!(s, "{x:?} ")?
            }
            s.push('\n');
        }
    }
    fs::write(&a[3], s)?;
    Ok(())
}
