//! Serializable device-buffer contracts and checked binding to IFKernelTensorV1.
//! No allocation, kernel launch, host staging, device dereference or ACL dependency.
use inferfabric_model::{Result, invalid};
use serde::{Deserialize, Serialize};
use std::{
    collections::{BTreeMap, BTreeSet},
    ffi::c_void,
    marker::PhantomData,
};
pub const FORMAT: &str = "inferfabric.device-buffer.v1";

#[derive(Clone, Debug, Serialize)]
pub struct ContractSupport {
    pub format: &'static str,
    pub kernel_abi: &'static str,
    pub dtypes: Vec<DType>,
    pub layout: &'static str,
    pub alias_policy: &'static str,
    pub max_tensor_rank: u32,
    pub binding_checks: Vec<&'static str>,
}
pub fn support() -> ContractSupport {
    ContractSupport {
        format: FORMAT,
        kernel_abi: "IFKernelTensorV1",
        dtypes: vec![DType::I8, DType::Bf16, DType::F32],
        layout: "contiguous positive strides",
        alias_policy: "disjoint outputs and workspace; read/read overlap allowed",
        max_tensor_rank: 8,
        binding_checks: vec![
            "allocation ID",
            "rank/device",
            "capacity",
            "alignment",
            "byte ranges",
            "absolute address aliases",
            "binary identity",
            "prepared workspace bound",
        ],
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DType {
    I8,
    Bf16,
    F32,
}
impl DType {
    pub fn bytes(self) -> u64 {
        match self {
            Self::I8 => 1,
            Self::Bf16 => 2,
            Self::F32 => 4,
        }
    }
    pub fn abi_tag(self) -> u32 {
        match self {
            Self::I8 => 1,
            Self::Bf16 => 2,
            Self::F32 => 3,
        }
    }
}
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Access {
    Read,
    Write,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MemoryClass {
    Weight,
    State,
    Activation,
    Workspace,
    Input,
    Output,
}
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Allocation {
    pub id: u32,
    pub rank: u32,
    pub device: u32,
    pub bytes: u64,
    pub alignment: u64,
    pub class: MemoryClass,
    pub read_only: bool,
}
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TensorView {
    pub allocation: u32,
    pub offset: u64,
    pub dtype: DType,
    pub shape: Vec<u64>,
    /// Element strides. Current native ABI requires contiguous positive views.
    pub strides: Vec<u64>,
    pub alignment: u64,
    pub access: Access,
}
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Workspace {
    pub allocation: u32,
    pub offset: u64,
    pub bytes: u64,
    pub alignment: u64,
}
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Invocation {
    pub format: String,
    pub implementation: String,
    pub binary_sha256: String,
    pub numerical_contract: String,
    pub rank: u32,
    pub device: u32,
    pub stream: u32,
    pub allocations: Vec<Allocation>,
    pub inputs: Vec<TensorView>,
    pub outputs: Vec<TensorView>,
    /// None freezes zero workspace; preparation cannot silently enlarge it.
    pub workspace: Option<Workspace>,
}
fn alignment(a: u64) -> bool {
    a != 0 && a.is_power_of_two()
}
fn overlap(a: (u64, u64), b: (u64, u64)) -> bool {
    a.0 < b.1 && b.0 < a.1
}
fn end(offset: u64, bytes: u64, capacity: u64) -> Result<u64> {
    let end = offset
        .checked_add(bytes)
        .ok_or_else(|| invalid("buffer range overflow"))?;
    if bytes == 0 || end > capacity {
        return Err(invalid("buffer range outside allocation"));
    }
    Ok(end)
}
impl TensorView {
    pub fn byte_span(&self) -> Result<u64> {
        if self.shape.is_empty() || self.shape.len() > 8 || self.strides.len() != self.shape.len() {
            return Err(invalid("tensor rank/stride mismatch"));
        }
        let mut stride = 1u64;
        for (&dim, &actual) in self.shape.iter().zip(&self.strides).rev() {
            if dim == 0 || dim > i64::MAX as u64 || stride > i64::MAX as u64 || actual != stride {
                return Err(invalid("kernel ABI requires contiguous positive views"));
            }
            stride = stride
                .checked_mul(dim)
                .ok_or_else(|| invalid("tensor shape overflow"))?;
        }
        stride
            .checked_mul(self.dtype.bytes())
            .ok_or_else(|| invalid("tensor byte overflow"))
    }
}
impl Invocation {
    pub fn verify(&self) -> Result<()> {
        if self.format != FORMAT
            || self.implementation.is_empty()
            || self.numerical_contract.is_empty()
            || self.binary_sha256.len() != 64
            || !self.binary_sha256.bytes().all(|x| x.is_ascii_hexdigit())
            || self.allocations.is_empty()
            || self.allocations.len() > 4096
            || self.inputs.is_empty()
            || self.outputs.is_empty()
            || self.inputs.len() > 64
            || self.outputs.len() > 64
        {
            return Err(invalid("device invocation contract/version/port bounds"));
        }
        let mut allocations = BTreeMap::new();
        for a in &self.allocations {
            if allocations.insert(a.id, a).is_some()
                || a.bytes == 0
                || !alignment(a.alignment)
                || a.rank != self.rank
                || a.device != self.device
                || (a.class == MemoryClass::Weight && !a.read_only)
            {
                return Err(invalid("allocation identity/placement/alignment"));
            }
        }
        let mut used = BTreeSet::new();
        let mut inputs = vec![];
        let mut outputs = vec![];
        for (views, access, ranges) in [
            (&self.inputs, Access::Read, &mut inputs),
            (&self.outputs, Access::Write, &mut outputs),
        ] {
            for v in views {
                let a = allocations
                    .get(&v.allocation)
                    .ok_or_else(|| invalid("unknown tensor allocation"))?;
                if v.access != access
                    || (access == Access::Write && a.read_only)
                    || !alignment(v.alignment)
                    || v.alignment < v.dtype.bytes()
                    || a.alignment < v.alignment
                    || !v.offset.is_multiple_of(v.alignment)
                {
                    return Err(invalid("view access/alignment"));
                }
                let limit = end(v.offset, v.byte_span()?, a.bytes)?;
                ranges.push((v.allocation, (v.offset, limit)));
                used.insert(v.allocation);
            }
        }
        for (i, &(a, r)) in outputs.iter().enumerate() {
            if inputs
                .iter()
                .chain(&outputs[..i])
                .any(|&(b, s)| a == b && overlap(r, s))
            {
                return Err(invalid("kernel ABI forbids writable aliases"));
            }
        }
        if let Some(w) = &self.workspace {
            let a = allocations
                .get(&w.allocation)
                .ok_or_else(|| invalid("unknown workspace allocation"))?;
            if a.read_only
                || !alignment(w.alignment)
                || w.alignment > 4096
                || a.alignment < w.alignment
                || !w.offset.is_multiple_of(w.alignment)
            {
                return Err(invalid("workspace access/alignment"));
            }
            let limit = end(w.offset, w.bytes, a.bytes)?;
            if inputs
                .iter()
                .chain(&outputs)
                .any(|&(id, r)| id == w.allocation && overlap((w.offset, limit), r))
            {
                return Err(invalid("workspace aliases tensor"));
            }
            used.insert(w.allocation);
        }
        if used.len() != allocations.len() {
            return Err(invalid("unused invocation allocation"));
        }
        Ok(())
    }
    /// Validate actual backend preparation against frozen requirements. Calling
    /// this after prepare and before enqueue is mandatory; it never replans.
    pub fn verify_prepared(
        &self,
        binary_sha256: &str,
        numerical_contract: &str,
        workspace_bytes: u64,
        workspace_alignment: u64,
    ) -> Result<()> {
        self.verify()?;
        if binary_sha256 != self.binary_sha256
            || numerical_contract != self.numerical_contract
            || !alignment(workspace_alignment)
            || workspace_alignment > 4096
        {
            return Err(invalid(
                "prepared implementation/contract/alignment mismatch",
            ));
        }
        match &self.workspace {
            Some(w) if workspace_bytes <= w.bytes && workspace_alignment <= w.alignment => Ok(()),
            None if workspace_bytes == 0 => Ok(()),
            _ => Err(invalid(
                "prepared workspace exceeds frozen contract; replan explicitly",
            )),
        }
    }
}

/// Exact POD layout of native/include/inferfabric_kernel_v1.h. This is a bound,
/// process-local view, deliberately not serializable. It owns no device memory.
#[repr(C)]
#[derive(Debug)]
pub struct KernelTensorV1 {
    pub struct_bytes: u32,
    pub dtype: u32,
    pub rank: u32,
    pub access: u32,
    pub data: *mut c_void,
    pub allocation_bytes: u64,
    pub shape: [i64; 8],
    pub strides: [i64; 8],
}
/// Caller-owned device allocation. The unsafe constructor is the only raw
/// address trust boundary; validation cannot prove that an address belongs to ACL.
#[derive(Debug)]
pub struct ResidentAllocation<'a> {
    id: u32,
    base: *mut c_void,
    bytes: u64,
    rank: u32,
    device: u32,
    _owner: PhantomData<&'a ()>,
}
impl<'a> ResidentAllocation<'a> {
    /// # Safety
    /// `owner` must keep this real device allocation alive and unmoved for its
    /// entire borrow. Its device/rank/capacity must be truthful. All queued work
    /// must complete (also after enqueue failure) before the owner is released.
    /// This function and the binder never dereference the supplied address.
    /// Host-only contract tests may supply simulated addresses, but must never
    /// submit the resulting descriptors to a device kernel.
    pub unsafe fn from_raw_parts<T: ?Sized>(
        id: u32,
        base: *mut c_void,
        bytes: u64,
        rank: u32,
        device: u32,
        _owner: &'a T,
    ) -> Self {
        Self {
            id,
            base,
            bytes,
            rank,
            device,
            _owner: PhantomData,
        }
    }
}
pub struct BoundInvocation<'a> {
    inputs: Vec<KernelTensorV1>,
    outputs: Vec<KernelTensorV1>,
    workspace: *mut c_void,
    workspace_bytes: u64,
    _resources: PhantomData<&'a ()>,
}
impl BoundInvocation<'_> {
    pub fn inputs(&self) -> &[KernelTensorV1] {
        &self.inputs
    }
    pub fn outputs(&self) -> &[KernelTensorV1] {
        &self.outputs
    }
    pub fn workspace(&self) -> (*mut c_void, u64) {
        (self.workspace, self.workspace_bytes)
    }
}
/// Binds exact IDs and checks absolute addresses as well as serialized ranges.
/// No fallback allocation, transfer, synchronization or device access occurs.
pub fn bind<'a>(
    call: &Invocation,
    residents: &'a [ResidentAllocation<'a>],
) -> Result<BoundInvocation<'a>> {
    call.verify()?;
    if residents.len() != call.allocations.len() {
        return Err(invalid("resident allocation count"));
    }
    let mut table = BTreeMap::new();
    let mut absolute = vec![];
    for r in residents {
        let a = call
            .allocations
            .iter()
            .find(|a| a.id == r.id)
            .ok_or_else(|| invalid("unknown resident ID"))?;
        let address = r.base as usize as u64;
        let capacity_end = address
            .checked_add(r.bytes)
            .ok_or_else(|| invalid("resident pointer overflow"))?;
        if table.insert(r.id, r).is_some()
            || r.base.is_null()
            || r.bytes < a.bytes
            || r.rank != a.rank
            || r.device != a.device
            || !address.is_multiple_of(a.alignment)
            || capacity_end > usize::MAX as u64
            || absolute
                .iter()
                .any(|&range| overlap((address, capacity_end), range))
        {
            return Err(invalid(
                "resident placement/capacity/alignment/alias mismatch",
            ));
        }
        absolute.push((address, capacity_end));
    }
    let tensor = |v: &TensorView| -> Result<KernelTensorV1> {
        let r = table[&v.allocation];
        let address = (r.base as usize as u64)
            .checked_add(v.offset)
            .ok_or_else(|| invalid("view pointer overflow"))?;
        let mut shape = [0; 8];
        let mut strides = [0; 8];
        for (i, (&d, &s)) in v.shape.iter().zip(&v.strides).enumerate() {
            shape[i] = i64::try_from(d).map_err(|_| invalid("shape ABI overflow"))?;
            strides[i] = i64::try_from(s).map_err(|_| invalid("stride ABI overflow"))?;
        }
        Ok(KernelTensorV1 {
            struct_bytes: std::mem::size_of::<KernelTensorV1>() as u32,
            dtype: v.dtype.abi_tag(),
            rank: v.shape.len() as u32,
            access: if v.access == Access::Read { 1 } else { 2 },
            data: address as usize as *mut c_void,
            allocation_bytes: v.byte_span()?,
            shape,
            strides,
        })
    };
    let inputs = call
        .inputs
        .iter()
        .map(&tensor)
        .collect::<Result<Vec<_>>>()?;
    let outputs = call
        .outputs
        .iter()
        .map(&tensor)
        .collect::<Result<Vec<_>>>()?;
    let (workspace, workspace_bytes) = if let Some(w) = &call.workspace {
        let address = (table[&w.allocation].base as usize as u64)
            .checked_add(w.offset)
            .ok_or_else(|| invalid("workspace pointer overflow"))?;
        (address as usize as *mut c_void, w.bytes)
    } else {
        (std::ptr::null_mut(), 0)
    };
    Ok(BoundInvocation {
        inputs,
        outputs,
        workspace,
        workspace_bytes,
        _resources: PhantomData,
    })
}
