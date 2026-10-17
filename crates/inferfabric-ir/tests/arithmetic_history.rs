use inferfabric_ir::{
    arithmetic_history::ArithmeticHistory,
    serving::{Phase, Variant},
};

fn variant(count: u32) -> Variant {
    Variant {
        id: format!("shape{count}"),
        phase: Phase::Extend,
        max_sequences: 1,
        max_tokens: count,
        max_query_tokens: count,
        max_context: 65536,
        numerical_class: format!("exact{count}"),
        graph_capable: true,
        packed: count > 1,
    }
}
fn append(history: &mut ArithmeticHistory, count: u32, times: usize) {
    for _ in 0..times {
        history
            .append_completed(&variant(count), history.tokens(), count)
            .unwrap();
    }
}

#[test]
fn equal_final_variant_does_not_hide_different_prefix_arithmetic() {
    let mut a = ArithmeticHistory::default();
    append(&mut a, 16, 4);
    let mut b = ArithmeticHistory::default();
    append(&mut b, 8, 4);
    append(&mut b, 16, 2);
    assert_eq!(a.tokens(), b.tokens());
    assert!(!a.same_prefix(&b, 64));
    let mut c = ArithmeticHistory::default();
    append(&mut c, 16, 2);
    append(&mut c, 8, 4);
    assert!(a.same_prefix(&c, 32));
    assert!(!a.same_prefix(&c, 64));
    assert!(!a.same_prefix(&c, 40)); // would split a T=16 invocation
    let prefix = a.prefix(32).unwrap();
    assert_eq!(prefix.tokens(), 32);
    assert!(prefix.same_prefix(&c, 32));
    assert_eq!(a.tokens(), 64); // lookup does not mutate the completed record
}

#[test]
fn long_context_coalesces_without_losing_invocation_boundaries() {
    let mut a = ArithmeticHistory::default();
    append(&mut a, 16, 4096);
    assert_eq!(a.tokens(), 65536);
    assert_eq!(a.spans().len(), 1);
    for boundary in [16, 64, 512, 32768, 65536] {
        assert_eq!(a.prefix(boundary).unwrap().tokens(), boundary);
    }
    for boundary in [1, 15, 65, 65535, 65537] {
        assert!(a.prefix(boundary).is_err());
    }
    let before = a.clone();
    assert!(a.append_completed(&variant(16), 65536, 16).is_err());
    assert_eq!(a, before);
    let encoded = serde_json::to_value(a).unwrap();
    assert_eq!(encoded["version"], 1);
    assert_eq!(encoded["spans"][0]["end"], 65536);
}

#[test]
fn failed_and_noncontiguous_invocations_cannot_advance_history() {
    let mut a = ArithmeticHistory::default();
    append(&mut a, 4, 2);
    let before = a.clone();
    for (position, count) in [(0, 4), (7, 4), (9, 4), (8, 0), (8, 3), (u32::MAX, 4)] {
        assert!(a.append_completed(&variant(4), position, count).is_err());
        assert_eq!(a, before);
    }
    a.check_append(&variant(4), 8, 4).unwrap();
    assert_eq!(a, before); // preflight is not completion
    let mut bad = variant(4);
    bad.numerical_class.clear();
    assert!(a.append_completed(&bad, 8, 4).is_err());
    assert_eq!(a, before);
    append(&mut a, 1, 1);
    assert_eq!(a.tokens(), 9);
    assert_eq!(a.spans().len(), 2);
    assert!(a.prefix(9).unwrap().same_prefix(&a, 9));
}
