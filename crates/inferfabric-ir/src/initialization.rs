//! Model-independent parameter initialization IR. Tensor IDs are dense artifact-local handles.
use inferfabric_model::{Result, invalid};
use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[repr(u32)]
pub enum DType {
    I8 = 1,
    Bf16 = 2,
    F32 = 3,
}
impl DType {
    pub fn bytes(self) -> u64 {
        match self {
            Self::I8 => 1,
            Self::Bf16 => 2,
            Self::F32 => 4,
        }
    }
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Parameter {
    pub id: u32,
    pub name: String,
    pub dtype: DType,
    pub shape: Vec<u64>,
}
impl Parameter {
    pub fn bytes(&self) -> Result<u64> {
        if self.shape.is_empty() || self.shape.len() > 8 {
            return Err(invalid("initialization shape rank"));
        }
        self.shape.iter().try_fold(self.dtype.bytes(), |n, &d| {
            if d == 0 || d > i64::MAX as u64 {
                return Err(invalid("zero initialization dimension"));
            }
            n.checked_mul(d)
                .ok_or_else(|| invalid("initialization size overflow"))
        })
    }
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SourceRows {
    pub row_bytes: u64,
    pub stride_bytes: u64,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Instruction {
    /// Allocate destination and stream a contiguous file slice into it. Conversion is
    /// fused with upload so a cast never requires a second full resident allocation.
    LoadSlice {
        destination: u32,
        shard: String,
        offset: u64,
        bytes: u64,
        source_dtype: DType,
        /// Optional strided source rows packed contiguously during streaming load.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        source_rows: Option<SourceRows>,
    },
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InitializationPlan {
    pub rank: u32,
    pub device: u32,
    /// Provenance only. Payload cryptographic integrity is a separate storage policy.
    pub checkpoint_metadata_sha256: String,
    pub parameters: Vec<Parameter>,
    pub instructions: Vec<Instruction>,
}
impl InitializationPlan {
    pub fn validate(&self) -> Result<()> {
        if self.parameters.is_empty()
            || self.parameters.len() > 100_000
            || self.instructions.len() != self.parameters.len()
        {
            return Err(invalid("initialization parameter/instruction count"));
        }
        if self.checkpoint_metadata_sha256.len() != 64
            || !self
                .checkpoint_metadata_sha256
                .bytes()
                .all(|c| c.is_ascii_hexdigit())
        {
            return Err(invalid("initialization provenance"));
        }
        let mut names = BTreeSet::new();
        let mut produced = BTreeSet::new();
        let mut total = 0u64;
        for (i, p) in self.parameters.iter().enumerate() {
            if p.id as usize != i
                || p.name.is_empty()
                || p.name.len() > 4096
                || p.name.contains('\0')
                || !names.insert(&p.name)
            {
                return Err(invalid("initialization parameter ID/name"));
            }
            total = total
                .checked_add(p.bytes()?)
                .ok_or_else(|| invalid("initialization total overflow"))?;
        }
        for op in &self.instructions {
            match op {
                Instruction::LoadSlice {
                    destination,
                    shard,
                    offset,
                    bytes,
                    source_dtype,
                    source_rows,
                } => {
                    let p = self
                        .parameters
                        .get(*destination as usize)
                        .ok_or_else(|| invalid("unknown initialization destination"))?;
                    if !produced.insert(destination) {
                        return Err(invalid("duplicate initialization producer"));
                    }
                    if shard.is_empty()
                        || shard.len() > 4096
                        || shard == "."
                        || shard == ".."
                        || shard.contains(['/', '\\', '\0', ':'])
                    {
                        return Err(invalid("initialization shard path"));
                    }
                    if *source_dtype != p.dtype
                        && !(*source_dtype == DType::Bf16 && p.dtype == DType::F32)
                    {
                        return Err(invalid("unsupported initialization conversion"));
                    }
                    let elements = p.bytes()? / p.dtype.bytes();
                    let span = if let Some(rows) = source_rows {
                        if rows.row_bytes == 0
                            || *bytes == 0
                            || bytes % rows.row_bytes != 0
                            || rows.row_bytes % source_dtype.bytes() != 0
                            || rows.stride_bytes < rows.row_bytes
                            || rows.stride_bytes % source_dtype.bytes() != 0
                        {
                            return Err(invalid("initialization source rows"));
                        }
                        (bytes / rows.row_bytes - 1)
                            .checked_mul(rows.stride_bytes)
                            .and_then(|x| x.checked_add(rows.row_bytes))
                            .ok_or_else(|| invalid("initialization row span overflow"))?
                    } else {
                        *bytes
                    };
                    if elements.checked_mul(source_dtype.bytes()) != Some(*bytes)
                        || offset
                            .checked_add(span)
                            .is_none_or(|end| end > i64::MAX as u64)
                    {
                        return Err(invalid("initialization source range"));
                    }
                }
            }
        }
        Ok(())
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    fn plan() -> InitializationPlan {
        InitializationPlan {
            rank: 17,
            device: 22,
            checkpoint_metadata_sha256: "a".repeat(64),
            parameters: vec![Parameter {
                id: 0,
                name: "parameter".into(),
                dtype: DType::F32,
                shape: vec![2, 3],
            }],
            instructions: vec![Instruction::LoadSlice {
                destination: 0,
                shard: "weights.bin".into(),
                offset: 8,
                bytes: 12,
                source_dtype: DType::Bf16,
                source_rows: None,
            }],
        }
    }
    #[test]
    fn generic_rank_and_streaming_cast() {
        plan().validate().unwrap();
    }
    #[test]
    fn strided_source_bounds_and_alignment() {
        let mut p = plan();
        let Instruction::LoadSlice { source_rows, .. } = &mut p.instructions[0];
        *source_rows = Some(SourceRows {
            row_bytes: 6,
            stride_bytes: 16,
        });
        p.validate().unwrap();
        for (row_bytes, stride_bytes) in [(0, 16), (5, 16), (6, 4), (6, 15), (6, u64::MAX - 1)] {
            let mut bad = p.clone();
            let Instruction::LoadSlice { source_rows, .. } = &mut bad.instructions[0];
            *source_rows = Some(SourceRows {
                row_bytes,
                stride_bytes,
            });
            assert!(bad.validate().is_err());
        }
    }
    #[test]
    fn rejects_invalid_contracts() {
        let mut p = plan();
        p.parameters[0].id = 1;
        assert!(p.validate().is_err());
        let mut p = plan();
        p.parameters[0].shape = vec![u64::MAX, 2];
        assert!(p.validate().is_err());
        let mut p = plan();
        p.instructions.push(p.instructions[0].clone());
        assert!(p.validate().is_err());
        for shard in ["../weights", "a/b", "a\\b", "C:weights", "", "."] {
            let mut p = plan();
            let Instruction::LoadSlice { shard: s, .. } = &mut p.instructions[0];
            *s = shard.into();
            assert!(p.validate().is_err());
        }
        let mut p = plan();
        let Instruction::LoadSlice { destination, .. } = &mut p.instructions[0];
        *destination = 1;
        assert!(p.validate().is_err());
        let mut p = plan();
        let Instruction::LoadSlice { source_dtype, .. } = &mut p.instructions[0];
        *source_dtype = DType::I8;
        assert!(p.validate().is_err());
    }
}
