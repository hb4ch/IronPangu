use inferfabric_authoring::execution::Graph;
use inferfabric_compiler::{token_physical::lower, token_view};
use inferfabric_plan::physical::{self, Cost, CostSource};
fn fixture() -> Graph {
    serde_json::from_str(include_str!("fixtures/viewer-scopes.json")).unwrap()
}
#[test]
fn deterministic_baseline_preserves_logical_program_and_explicit_unknowns() {
    let graph = fixture();
    let binary = graph.encode().unwrap();
    let a = lower(&graph, &[7, 6, 5, 4, 3, 2, 1, 0]).unwrap();
    assert_eq!(a, lower(&graph, &[7, 6, 5, 4, 3, 2, 1, 0]).unwrap());
    assert!(!a.executable);
    assert_eq!(a.ranks[0].device, 7);
    let s = &a.schedules[0];
    assert_eq!(s.steps.len(), 7);
    assert_eq!(s.values[6].lifetime, [6, 7]);
    assert!(!s.memory_complete);
    assert!(!s.cost_complete);
    assert_eq!(s.estimated_serial_ns, None);
    assert!(s.known_host_payload_peak_bytes >= 606208);
    assert_eq!(
        s.steps[6]
            .operation
            .collective
            .as_ref()
            .unwrap()
            .elements_per_rank,
        18944
    );
    assert!(
        token_view::render_with_physical(&graph, &a)
            .unwrap()
            .contains("physical")
    );
    assert_eq!(binary, graph.encode().unwrap());
    let mut other = graph;
    other.name = "other".into();
    assert!(token_view::render_with_physical(&other, &a).is_err());
}
#[test]
fn reject_corrupted_schedule_memory_collectives_and_choices() {
    let baseline = lower(&fixture(), &[0, 1, 2, 3, 4, 5, 6, 7]).unwrap();
    for case in 0..11 {
        let mut p = baseline.clone();
        match case {
            0 => p.executable = true,
            1 => p.ranks[1].device = 0,
            2 => p.groups[0].ranks.pop().map(|_| ()).unwrap(),
            3 => p.schedules[0].steps[2].data_dependencies = vec![2],
            4 => p.schedules[0].values[0].lifetime[1] = 0,
            5 => p.schedules[0].values[0].arena_offset = Some(0),
            6 => {
                p.schedules[0].steps[6]
                    .operation
                    .collective
                    .as_mut()
                    .unwrap()
                    .sequence = 99
            }
            7 => p.schedules[0].steps[0].selected = 9,
            8 => p.schedules[0].memory_complete = true,
            9 => p.schedules[0].known_host_payload_peak_bytes = 0,
            _ => p.schedules[0].steps[0].operation.candidates[0].eligible = false,
        };
        assert!(physical::verify(&p).is_err(), "mutation {case}");
    }
}
#[test]
fn cost_guided_choice_is_legal_conservative_and_provenanced() {
    let p = lower(&fixture(), &[0, 1, 2, 3, 4, 5, 6, 7]).unwrap();
    let mut a = p.schedules[0].steps[0].operation.candidates[0].clone();
    a.id = "a".into();
    a.cost = Cost {
        source: CostSource::Measured,
        estimate_ns: Some(10),
        upper_ns: Some(30),
        evidence: Some("synthetic-test-sample-not-hardware".into()),
    };
    let mut b = a.clone();
    b.id = "b".into();
    b.cost.estimate_ns = Some(15);
    b.cost.upper_ns = Some(20);
    assert_eq!(physical::choose(&[a.clone(), b.clone()]).unwrap().0, 1);
    b.cost = Cost::default();
    assert_eq!(physical::choose(&[b.clone(), a.clone()]).unwrap().0, 1);
    b.eligible = false;
    b.rejection = Some("unsupported layout".into());
    assert_eq!(physical::choose(&[a.clone(), b]).unwrap().0, 0);
    a.cost.evidence = None;
    assert!(physical::choose(&[a]).is_err());
}
