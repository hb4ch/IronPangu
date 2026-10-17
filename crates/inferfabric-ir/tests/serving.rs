use inferfabric_ir::serving::*;
fn manifest() -> Manifest {
    Manifest {
        version: VERSION,
        resident_sequences: 1,
        shared_pages: false,
        identity: Identity {
            program_sha256: "a".repeat(64),
            checkpoint_metadata_sha256: "b".repeat(64),
            backend_revision: "c".repeat(64),
            devices: vec![0, 1],
        },
        vocab_size: 100,
        eos: 99,
        variants: vec![Variant {
            id: "extend".into(),
            phase: Phase::Extend,
            max_sequences: 2,
            max_tokens: 8,
            max_query_tokens: 4,
            max_context: 32,
            numerical_class: "packed_v1".into(),
            graph_capable: true,
            packed: true,
        }],
        states: vec![State {
            id: "kv".into(),
            logical_node: 0,
            ranks: vec![0, 1],
            storage: Storage::Pages {
                page_tokens: 16,
                row_bytes: 64,
                encoding: "bf16".into(),
            },
            capacity_tokens: 32,
            window_tokens: None,
            checkpoint_supported: false,
        }],
        qualification: None,
        limitations: vec![],
    }
}
fn batch() -> Batch {
    Batch {
        engine_epoch: 1,
        sequence: 1,
        variant: "extend".into(),
        tokens: vec![1, 2, 3, 4, 5],
        requests: vec![
            RequestSpan {
                id: 8,
                generation: 1,
                token_offset: 0,
                query_tokens: 3,
                prior_tokens: 16,
                sample_row: None,
            },
            RequestSpan {
                id: 9,
                generation: 1,
                token_offset: 3,
                query_tokens: 2,
                prior_tokens: 0,
                sample_row: Some(1),
            },
        ],
    }
}
#[test]
fn ragged_batch_contract_rejects_aliases_gaps_and_overflow() {
    let m = manifest();
    m.validate_batch(&batch()).unwrap();
    for defect in 0..9 {
        let mut b = batch();
        match defect {
            0 => b.requests[1].id = 8,
            1 => b.requests[1].token_offset = 2,
            2 => b.requests[0].prior_tokens = u32::MAX,
            3 => b.tokens[0] = 100,
            4 => b.requests[0].query_tokens = 0,
            5 => b.requests[1].sample_row = Some(0),
            6 => b.variant = "missing".into(),
            7 => b.engine_epoch = 0,
            _ => b.requests[0].generation = 0,
        }
        assert!(m.validate_batch(&b).is_err(), "defect {defect}");
    }
}
#[test]
fn invalid_manifest_cannot_grant_capability() {
    for defect in 0..8 {
        let mut m = manifest();
        match defect {
            0 => m.version += 1,
            1 => m.identity.devices[1] = 0,
            2 => m.states[0].ranks.push(2),
            3 => m.states[0].capacity_tokens = 31,
            4 => m.variants[0].packed = false,
            5 => m.variants[0].phase = Phase::Decode,
            6 => m.identity.program_sha256 = "unknown".into(),
            _ => m.variants.push(m.variants[0].clone()),
        }
        assert!(m.validate().is_err(), "defect {defect}");
    }
}
#[test]
fn prefix_identity_includes_arithmetic_tenant_and_backend() {
    let a = PrefixIdentity {
        engine: manifest().identity,
        numerical_class: "serial".into(),
        tokenizer_template_sha256: "d".repeat(64),
        tenant_namespace: "tenant-a".into(),
        adapter_sha256: None,
    };
    assert!(a.compatible(&a));
    for defect in 0..6 {
        let mut b = a.clone();
        match defect {
            0 => b.numerical_class = "packed".into(),
            1 => b.tenant_namespace = "tenant-b".into(),
            2 => b.engine.devices.reverse(),
            3 => b.engine.backend_revision = "e".repeat(64),
            4 => b.adapter_sha256 = Some("f".repeat(64)),
            _ => b.tokenizer_template_sha256.clear(),
        }
        assert!(!a.compatible(&b));
    }
}
