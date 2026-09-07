use crate::{GenerateRequest, GenerateOutputStream, Result};
use vllm_engine_core_client::protocol::dtype::ModelDtype;

pub struct BackendMetadata {
    pub max_model_len: u32,
    pub model_dtype: ModelDtype,
    pub version: String,
    pub healthy: bool,
}

/// In-process token generation; implementations own admission and cancellation.
#[async_trait::async_trait]
pub trait GenerationBackend: Send + Sync {
    fn metadata(&self) -> BackendMetadata;
    async fn generate(&self, request: GenerateRequest) -> Result<GenerateOutputStream>;
    async fn abort(&self, request_ids: &[String]) -> Result<()>;
    async fn shutdown(&self) -> Result<()>;
}
