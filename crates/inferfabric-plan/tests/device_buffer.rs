use inferfabric_plan::device_buffer::*;
use std::ffi::c_void;
fn fixture() -> Invocation {
    let tensor = |offset, access| TensorView {
        allocation: 1,
        offset,
        dtype: DType::F32,
        shape: vec![2, 2],
        strides: vec![2, 1],
        alignment: 16,
        access,
    };
    Invocation {
        format: FORMAT.into(),
        implementation: "test.elementwise.f32.v1".into(),
        binary_sha256: "a".repeat(64),
        numerical_contract: "test.add.f32.v1".into(),
        rank: 2,
        device: 5,
        stream: 0,
        allocations: vec![Allocation {
            id: 1,
            rank: 2,
            device: 5,
            bytes: 4096,
            alignment: 512,
            class: MemoryClass::Activation,
            read_only: false,
        }],
        inputs: vec![tensor(0, Access::Read)],
        outputs: vec![tensor(512, Access::Write)],
        workspace: Some(Workspace {
            allocation: 1,
            offset: 1024,
            bytes: 256,
            alignment: 512,
        }),
    }
}
#[repr(align(512))]
struct MockMemory([u8; 4096]);
#[test]
fn binds_disjoint_views_without_copying_or_owning_storage() {
    let call = fixture();
    let mut owner = Box::new(MockMemory([0; 4096]));
    let pointer = owner.0.as_mut_ptr().cast::<c_void>();
    // Host-backed mock: descriptors are inspected only, never enqueued on a device.
    let residents = [unsafe { ResidentAllocation::from_raw_parts(1, pointer, 4096, 2, 5, &owner) }];
    let bound = bind(&call, &residents).unwrap();
    assert_eq!(bound.inputs()[0].data, pointer);
    assert_eq!(bound.outputs()[0].data as usize, pointer as usize + 512);
    assert_eq!(bound.inputs()[0].allocation_bytes, 16);
    assert_eq!(bound.outputs()[0].allocation_bytes, 16);
    assert_eq!(bound.inputs()[0].dtype, 3);
    assert_eq!(bound.inputs()[0].access, 1);
    assert_eq!(bound.outputs()[0].access, 2);
    assert_eq!(bound.inputs()[0].shape[..2], [2, 2]);
    assert_eq!(
        bound.workspace(),
        ((pointer as usize + 1024) as *mut c_void, 256)
    );
    call.verify_prepared(&"a".repeat(64), "test.add.f32.v1", 128, 256)
        .unwrap();
    assert!(
        call.verify_prepared(&"a".repeat(64), "test.add.f32.v1", 257, 256)
            .is_err()
    );
    assert!(
        call.verify_prepared(&"a".repeat(64), "test.add.f32.v1", 128, 1024)
            .is_err()
    );
    assert!(
        call.verify_prepared(&"b".repeat(64), "test.add.f32.v1", 128, 256)
            .is_err()
    );
    assert!(
        call.verify_prepared(&"a".repeat(64), "other", 128, 256)
            .is_err()
    );
}
#[test]
fn rejects_malformed_and_unsafe_views() {
    for case in 0..18 {
        let mut c = fixture();
        match case {
            0 => c.outputs[0].offset = 0,
            1 => c.workspace.as_mut().unwrap().offset = 512,
            2 => c.inputs[0].strides = vec![1, 2],
            3 => c.inputs[0].shape[0] = 0,
            4 => c.inputs[0].offset = u64::MAX - 15,
            5 => c.inputs[0].allocation = 9,
            6 => c.allocations[0].read_only = true,
            7 => c.allocations[0].rank = 1,
            8 => c.allocations[0].device = 3,
            9 => c.inputs[0].access = Access::Write,
            10 => c.inputs[0].alignment = 3,
            11 => c.allocations.push(c.allocations[0].clone()),
            12 => c.workspace.as_mut().unwrap().alignment = 8192,
            13 => c.outputs[0].shape = vec![2; 9],
            14 => {
                c.inputs[0].shape = vec![i64::MAX as u64, 4];
                c.inputs[0].strides = vec![4, 1]
            }
            15 => c.format = "unknown".into(),
            16 => c.allocations[0].class = MemoryClass::Weight,
            _ => c.outputs.push(c.outputs[0].clone()),
        };
        assert!(c.verify().is_err(), "case {case}");
    }
    let mut c = fixture();
    c.inputs.push(c.inputs[0].clone());
    c.verify().unwrap(); // read/read overlap is legal
    c.workspace = None;
    c.verify_prepared(&"a".repeat(64), "test.add.f32.v1", 0, 1)
        .unwrap();
    assert!(
        c.verify_prepared(&"a".repeat(64), "test.add.f32.v1", 1, 1)
            .is_err()
    );
}
#[test]
fn rejects_bad_residency_and_disguised_cross_allocation_aliases() {
    let c = fixture();
    let owner = ();
    for case in 0..6 {
        let (id, pointer, bytes, rank, device) = match case {
            0 => (1, 0, 4096, 2, 5),
            1 => (1, 0x1001, 4096, 2, 5),
            2 => (1, 0x1000, 64, 2, 5),
            3 => (1, 0x1000, 4096, 3, 5),
            4 => (1, 0x1000, 4096, 2, 7),
            _ => (8, 0x1000, 4096, 2, 5),
        };
        let residents = [unsafe {
            ResidentAllocation::from_raw_parts(
                id,
                pointer as *mut c_void,
                bytes,
                rank,
                device,
                &owner,
            )
        }];
        assert!(bind(&c, &residents).is_err());
    }
    let mut c = c;
    c.allocations.push(Allocation {
        id: 2,
        rank: 2,
        device: 5,
        bytes: 512,
        alignment: 512,
        class: MemoryClass::Output,
        read_only: false,
    });
    c.outputs[0].allocation = 2;
    c.outputs[0].offset = 0;
    let residents = [
        unsafe { ResidentAllocation::from_raw_parts(1, 0x1000 as *mut c_void, 4096, 2, 5, &owner) },
        unsafe { ResidentAllocation::from_raw_parts(2, 0x1000 as *mut c_void, 512, 2, 5, &owner) },
    ];
    assert!(bind(&c, &residents).is_err());
    let overflow = [unsafe {
        ResidentAllocation::from_raw_parts(1, (usize::MAX - 511) as *mut c_void, 4096, 2, 5, &owner)
    }];
    assert!(bind(&fixture(), &overflow).is_err());
}
#[test]
fn serialized_contract_has_no_pointers_and_pod_layout_matches_kernel_abi() {
    let c = fixture();
    let json = serde_json::to_value(&c).unwrap();
    assert!(json.get("data").is_none());
    assert_eq!(
        serde_json::from_value::<Invocation>(json.clone()).unwrap(),
        c
    );
    let mut extra = json;
    extra["pointer"] = serde_json::json!(123);
    assert!(serde_json::from_value::<Invocation>(extra).is_err());
    if cfg!(target_pointer_width = "64") {
        assert_eq!(std::mem::size_of::<KernelTensorV1>(), 160);
        assert_eq!(std::mem::align_of::<KernelTensorV1>(), 8);
        assert_eq!(std::mem::offset_of!(KernelTensorV1, data), 16);
        assert_eq!(std::mem::offset_of!(KernelTensorV1, allocation_bytes), 24);
        assert_eq!(std::mem::offset_of!(KernelTensorV1, shape), 32);
        assert_eq!(std::mem::offset_of!(KernelTensorV1, strides), 96);
    }
}
