use inferfabric_ir::{
    arithmetic_history::ArithmeticHistory,
    serving::{Identity, Phase, PrefixIdentity, Variant},
};
use inferfabric_runtime::prefix_index::PrefixIndex;

fn identity() -> PrefixIdentity {
    PrefixIdentity {
        engine: Identity {
            program_sha256: "1".repeat(64),
            checkpoint_metadata_sha256: "2".repeat(64),
            backend_revision: "3".repeat(64),
            devices: vec![0, 1],
        },
        numerical_class: "exact".into(),
        tokenizer_template_sha256: "4".repeat(64),
        tenant_namespace: "tenant-a".into(),
        adapter_sha256: None,
    }
}
fn history(tokens: u32, width: u32) -> ArithmeticHistory {
    let variant = Variant {
        id: format!("t{width}"),
        phase: Phase::Prefill,
        max_sequences: 1,
        max_tokens: width,
        max_query_tokens: width,
        max_context: 65536,
        numerical_class: format!("exact{width}"),
        packed: width > 1,
        graph_capable: true,
    };
    let mut result = ArithmeticHistory::default();
    for position in (0..tokens).step_by(width as usize) {
        result.append_completed(&variant, position, width).unwrap();
    }
    result
}
#[test]
fn split_edges_do_not_invent_checkpoints_or_skip_final_logit_recomputation() {
    let ns = identity();
    let mut index = PrefixIndex::new(ns.clone(), 16, 8, 1024).unwrap();
    let tokens = (0..64).collect::<Vec<_>>();
    let parent = index.insert(&ns, &tokens[..32], history(32, 4)).unwrap();
    let full = index.insert(&ns, &tokens, history(64, 4)).unwrap();
    let mut branch = tokens.clone();
    branch[40] = 900;
    let fork = index.insert(&ns, &branch, history(64, 4)).unwrap();
    let found = index.lookup(&ns, &tokens[..48], &history(48, 4), 47);
    assert_eq!(found.matched_tokens, 48);
    assert_eq!(found.checkpoint, Some((parent, 32)));
    assert_eq!(
        index.lookup(&ns, &tokens, &history(64, 4), 63).checkpoint,
        Some((parent, 32))
    );
    assert_eq!(
        index.lookup(&ns, &tokens, &history(64, 4), 64).checkpoint,
        Some((full, 64))
    );
    assert_eq!(
        index.lookup(&ns, &branch, &history(64, 4), 64).checkpoint,
        Some((fork, 64))
    );
    assert!(index.remove(parent));
    assert!(
        index
            .lookup(&ns, &tokens, &history(64, 4), 63)
            .checkpoint
            .is_none()
    );
    assert!(index.remove(full));
    assert_eq!(
        index
            .lookup(&ns, &tokens, &history(64, 4), 64)
            .matched_tokens,
        40
    );
    assert!(!index.remove(full));
    assert!(index.remove(fork));
    assert!(index.is_empty());
    assert_eq!(index.token_units(), 0);
}
#[test]
fn namespaces_arithmetic_and_generations_are_isolated() {
    let ns = identity();
    let mut index = PrefixIndex::new(ns.clone(), 16, 8, 1024).unwrap();
    let tokens = vec![7; 64];
    let a = index.insert(&ns, &tokens, history(64, 4)).unwrap();
    assert!(
        index
            .lookup(&ns, &tokens, &history(64, 1), 64)
            .checkpoint
            .is_none()
    );
    let b = index.insert(&ns, &tokens, history(64, 1)).unwrap();
    assert_eq!(
        index.lookup(&ns, &tokens, &history(64, 4), 64).checkpoint,
        Some((a, 64))
    );
    assert_eq!(
        index.lookup(&ns, &tokens, &history(64, 1), 64).checkpoint,
        Some((b, 64))
    );
    for change in 0..6 {
        let mut other = ns.clone();
        match change {
            0 => other.tenant_namespace = "other".into(),
            1 => other.engine.devices.reverse(),
            2 => other.engine.program_sha256 = "5".repeat(64),
            3 => other.tokenizer_template_sha256 = "6".repeat(64),
            4 => other.adapter_sha256 = Some("7".repeat(64)),
            _ => other.engine.backend_revision = "8".repeat(64),
        }
        assert_eq!(
            index
                .lookup(&other, &tokens, &history(64, 4), 64)
                .matched_tokens,
            0
        );
        assert!(index.insert(&other, &tokens, history(64, 4)).is_err());
    }
    let mut other = PrefixIndex::new(ns.clone(), 16, 8, 1024).unwrap();
    let foreign = other.insert(&ns, &tokens, history(64, 4)).unwrap();
    assert!(!index.remove(foreign));
    assert!(index.remove(a));
    let fresh = index.insert(&ns, &tokens, history(64, 4)).unwrap();
    assert_ne!(fresh, a);
    assert!(!index.remove(a));
}
#[test]
fn capacity_failures_preserve_the_existing_tree_at_64k() {
    let ns = identity();
    let mut index = PrefixIndex::new(ns.clone(), 64, 2, 65536).unwrap();
    let tokens = vec![8; 65536];
    let id = index.insert(&ns, &tokens, history(65536, 16)).unwrap();
    assert!(index.insert(&ns, &tokens[..64], history(64, 16)).is_err());
    assert_eq!(index.len(), 1);
    assert_eq!(index.token_units(), 65536);
    assert_eq!(
        index
            .lookup(&ns, &tokens, &history(65536, 16), 65536)
            .checkpoint,
        Some((id, 65536))
    );
    assert!(
        index
            .lookup(&ns, &tokens, &history(65536, 16), 65535)
            .checkpoint
            .is_none()
    );
    assert!(index.remove(id));
    assert!(index.insert(&ns, &tokens[..63], history(63, 1)).is_err());
    assert!(index.insert(&ns, &tokens[..64], history(32, 16)).is_err());
    assert!(index.is_empty());
    assert_eq!(index.token_units(), 0);
}
#[test]
fn branch_removal_matches_a_linear_exact_token_oracle() {
    let ns = identity();
    let mut index = PrefixIndex::new(ns.clone(), 4, 32, 4096).unwrap();
    let mut entries = Vec::new();
    for n in 1..=24u32 {
        let mut tokens = vec![1; (n as usize % 5 + 1) * 4];
        *tokens.last_mut().unwrap() = n + 1;
        let id = index
            .insert(&ns, &tokens, history(tokens.len() as u32, 4))
            .unwrap();
        entries.push((id, tokens));
    }
    while !entries.is_empty() {
        for (_, tokens) in &entries {
            let expected = entries
                .iter()
                .filter(|(_, p)| tokens.starts_with(p))
                .max_by_key(|(_, p)| p.len())
                .unwrap();
            assert_eq!(
                index
                    .lookup(
                        &ns,
                        tokens,
                        &history(tokens.len() as u32, 4),
                        tokens.len() as u32
                    )
                    .checkpoint,
                Some((expected.0, expected.1.len() as u32))
            );
        }
        let (id, _) = entries.remove(entries.len() / 2);
        assert!(index.remove(id));
    }
    assert_eq!(index.token_units(), 0);
}
