use inferfabric_ir::serving::*;
use inferfabric_runtime::state_pool::StatePool;
use std::collections::BTreeMap;

fn manifest() -> Manifest {
    Manifest {
        version: VERSION,
        resident_sequences: 4,
        identity: Identity {
            program_sha256: "a".repeat(64),
            checkpoint_metadata_sha256: "b".repeat(64),
            backend_revision: "c".repeat(64),
            devices: vec![0, 1],
        },
        vocab_size: 100,
        eos: 99,
        variants: vec![Variant {
            id: "decode".into(),
            phase: Phase::Decode,
            max_sequences: 4,
            max_tokens: 4,
            max_query_tokens: 1,
            max_context: 64,
            numerical_class: "test".into(),
            graph_capable: true,
            packed: true,
        }],
        states: vec![
            State {
                id: "kv16".into(),
                logical_node: 0,
                ranks: vec![0, 1],
                storage: Storage::Pages {
                    page_tokens: 16,
                    row_bytes: 1152,
                    encoding: "bf16".into(),
                },
                capacity_tokens: 64,
                window_tokens: None,
                checkpoint_supported: false,
            },
            State {
                id: "kv32".into(),
                logical_node: 1,
                ranks: vec![0, 1],
                storage: Storage::Pages {
                    page_tokens: 32,
                    row_bytes: 656,
                    encoding: "c8".into(),
                },
                capacity_tokens: 64,
                window_tokens: None,
                checkpoint_supported: false,
            },
            State {
                id: "history".into(),
                logical_node: 2,
                ranks: vec![0, 1],
                storage: Storage::Slot {
                    bytes: 4096,
                    dtype: "bf16".into(),
                },
                capacity_tokens: 64,
                window_tokens: None,
                checkpoint_supported: false,
            },
        ],
        qualification: None,
        limitations: vec![],
    }
}
fn budgets() -> BTreeMap<(String, u32), u32> {
    manifest()
        .states
        .iter()
        .flat_map(|s| s.ranks.iter().map(|&r| ((s.id.clone(), r), 4)))
        .collect()
}
fn pool() -> StatePool {
    StatePool::new(&manifest(), 7, budgets()).unwrap()
}

#[test]
fn asymmetric_exhaustion_rolls_back_every_rank_and_component() {
    let mut capacity = budgets();
    capacity.insert(("kv32".into(), 1), 1);
    let mut pool = StatePool::new(&manifest(), 7, capacity).unwrap();
    let before = pool.usage();
    assert!(pool.reserve(7, 1, 33).is_err());
    assert_eq!(pool.usage(), before);
    let lease = pool.reserve(7, 1, 32).unwrap();
    let bindings = pool.bindings(7, lease).unwrap();
    assert_eq!(bindings[0].units.len(), 2);
    assert_eq!(bindings[2].units.len(), 1);
    assert_eq!(bindings[4].units.len(), 1);
    pool.retire(7, lease).unwrap();
    assert_eq!(pool.usage(), before);
}

#[test]
fn retirement_requires_all_events_and_borrowers_not_just_host_release() {
    let mut p = pool();
    let l = p.reserve(7, 1, 64).unwrap();
    let pin = p.borrow(7, l).unwrap();
    let t = p.submit(7, &[l], &[]).unwrap();
    p.retire(7, l).unwrap();
    assert!(p.bindings(7, l).is_err());
    assert!(p.borrow(7, l).is_err());
    assert!(p.reserve(7, 2, 16).is_err());
    p.complete(7, t, 0).unwrap();
    assert!(p.complete(7, t, 0).is_err());
    assert!(p.complete(7, t, 2).is_err());
    p.complete(7, t, 1).unwrap();
    assert!(p.reserve(7, 2, 16).is_err()); // borrower still pins the pages
    let read = p.submit(7, &[], &[pin]).unwrap();
    p.release(7, pin).unwrap();
    assert!(p.release(7, pin).is_err());
    assert!(p.reserve(7, 2, 16).is_err()); // restore still in flight
    p.complete(7, read, 1).unwrap();
    assert!(p.reserve(7, 2, 16).is_err());
    p.complete(7, read, 0).unwrap();
    assert!(p.complete(7, read, 0).is_err());
    assert!(p.usage().iter().all(|u| u.occupied_bytes == 0));
    let fresh = p.reserve(7, 1, 64).unwrap();
    assert_ne!(fresh, l);
    assert!(p.submit(7, &[l], &[]).is_err());
}

#[test]
fn fragmented_pages_do_not_alias_and_cross_pool_handles_fail() {
    let mut p = pool();
    let a = p.reserve(7, 1, 16).unwrap();
    let b = p.reserve(7, 2, 16).unwrap();
    let c = p.reserve(7, 3, 16).unwrap();
    p.retire(7, a).unwrap();
    p.retire(7, c).unwrap();
    let d = p.reserve(7, 4, 32).unwrap();
    assert_eq!(p.bindings(7, d).unwrap()[0].units, vec![0, 2]);
    let address = p.page_address(7, d, "kv16", 0, 17).unwrap();
    assert_eq!(
        (
            address.physical_page,
            address.row_in_page,
            address.byte_offset
        ),
        (2, 1, 33 * 1152)
    );
    assert!(p.page_address(7, d, "kv16", 0, 32).is_err());
    assert!(p.page_address(7, d, "history", 0, 0).is_err());
    assert!(p.page_address(7, d, "kv16", 2, 0).is_err());
    for (left, right) in p
        .bindings(7, b)
        .unwrap()
        .iter()
        .zip(p.bindings(7, d).unwrap())
    {
        assert!(left.units.iter().all(|unit| !right.units.contains(unit)));
    }
    assert!(p.bindings(8, d).is_err());
    let mut other = pool();
    let foreign = other.reserve(7, 1, 16).unwrap();
    assert!(p.bindings(7, foreign).is_err());
    let t = p.submit(7, &[d], &[]).unwrap();
    assert!(other.complete(7, t, 0).is_err());
    p.poison();
    let used = p.usage();
    assert!(p.complete(7, t, 0).is_err());
    assert!(p.retire(7, d).is_err());
    assert!(p.reserve(7, 5, 1).is_err());
    assert_eq!(p.usage(), used); // failed rank cannot free possibly-live addresses
}

#[test]
fn interleaved_lifetimes_return_to_baseline_without_duplicate_physical_ids() {
    let mut p = pool();
    let baseline = p.usage();
    for cycle in 0..100 {
        let mut active = Vec::new();
        for request in 0..4 {
            let lease = p
                .reserve(7, cycle * 4 + request, 1 + (cycle % 16) as u32)
                .unwrap();
            let ticket = p.submit(7, &[lease], &[]).unwrap();
            active.push((lease, ticket));
        }
        for i in 0..4 {
            for j in i + 1..4 {
                for (a, b) in p
                    .bindings(7, active[i].0)
                    .unwrap()
                    .iter()
                    .zip(p.bindings(7, active[j].0).unwrap())
                {
                    assert!(a.units.iter().all(|u| !b.units.contains(u)));
                }
            }
        }
        for &(lease, ticket) in active.iter().rev() {
            p.retire(7, lease).unwrap();
            p.complete(7, ticket, (cycle % 2) as u32).unwrap();
        }
        assert!(p.reserve(7, 9999, 1).is_err());
        for &(_, ticket) in &active {
            p.complete(7, ticket, (1 - cycle % 2) as u32).unwrap();
        }
        assert_eq!(p.usage(), baseline);
    }
}

#[test]
fn rounded_storage_does_not_extend_executable_context() {
    let mut m = manifest();
    m.variants[0].max_context = 33;
    let mut p = StatePool::new(&m, 7, budgets()).unwrap();
    let baseline = p.usage();
    assert!(p.reserve(7, 1, 34).is_err());
    assert!(p.reserve(7, 1, 64).is_err());
    assert_eq!(p.usage(), baseline);
    let lease = p.reserve(7, 1, 33).unwrap();
    assert!(p.page_address(7, lease, "kv32", 0, 32).is_ok());
    assert!(p.page_address(7, lease, "kv32", 0, 33).is_err());
    p.retire(7, lease).unwrap();
    assert_eq!(p.usage(), baseline);
}

#[test]
fn full_64k_ownership_with_mixed_requests_and_delayed_rank_retirement() {
    // Synthetic ownership envelope, not a claim that a native 64K variant exists.
    // Full histories deliberately avoid assuming unimplemented SWA recycling.
    let mut m = manifest();
    m.identity.devices = (0..8).collect();
    m.variants[0].max_context = 65_536;
    for s in &mut m.states {
        s.capacity_tokens = 65_536;
        s.ranks = (0..8).collect();
    }
    let mut capacities = BTreeMap::new();
    for s in &m.states {
        let units = match s.storage {
            Storage::Pages { page_tokens, .. } => 2 * 65_536 / page_tokens,
            Storage::Slot { .. } => 4,
        };
        for &rank in &s.ranks {
            capacities.insert((s.id.clone(), rank), units);
        }
    }
    let mut p = StatePool::new(&m, 7, capacities).unwrap();
    let baseline = p.usage();
    assert!(p.reserve(7, 99, 65_537).is_err());
    assert_eq!(p.usage(), baseline);
    let long = p.reserve(7, 1, 65_536).unwrap();
    let medium = p.reserve(7, 2, 32_769).unwrap();
    let short = p.reserve(7, 3, 17).unwrap();
    for rank in 0..8 {
        for (component, page_tokens, row_bytes) in [("kv16", 16, 1152), ("kv32", 32, 656)] {
            let end = p.page_address(7, long, component, rank, 65_535).unwrap();
            assert_eq!(end.physical_page, 65_536 / page_tokens - 1);
            assert_eq!(end.row_in_page, page_tokens - 1);
            assert_eq!(end.byte_offset, 65_535 * row_bytes);
            let next = p.page_address(7, medium, component, rank, 0).unwrap();
            assert_eq!(next.byte_offset, 65_536 * row_bytes);
            assert!(p.page_address(7, long, component, rank, 65_536).is_err());
        }
    }
    let occupied = p.usage();
    assert!(p.reserve(7, 4, 32_768).is_err());
    assert_eq!(p.usage(), occupied); // pressure must roll back every rank
    let flight = p.submit(7, &[long], &[]).unwrap();
    p.retire(7, long).unwrap();
    for rank in 0..7 {
        p.complete(7, flight, rank).unwrap();
    }
    assert!(p.reserve(7, 4, 65_536).is_err());
    p.complete(7, flight, 7).unwrap();
    let replacement = p.reserve(7, 4, 65_536).unwrap();
    assert_ne!(replacement, long);
    assert!(p.page_address(7, long, "kv32", 0, 0).is_err());
    for lease in [replacement, medium, short] {
        p.retire(7, lease).unwrap();
    }
    assert_eq!(p.usage(), baseline);
}
