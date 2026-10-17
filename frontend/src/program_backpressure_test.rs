//! Opt-in full-model test. Cross-compile locally; execute only in the NPU container.
use super::*;
use futures::StreamExt;
use std::time::Duration;
// Force the same worker backpressure branch before a short reference reaches
// EOS. Production output queues remain capped at 64 responses.
const TEST_QUEUE_CAPACITY: usize = 2;

fn request(id: &str, prompt: &[u32], outputs: u32) -> GenerateRequest {
    GenerateRequest {
        request_id: id.into(), prompt_token_ids: prompt.to_vec(),
        sampling_params: serde_json::from_value(serde_json::json!({"temperature":0,"max_tokens":outputs})).unwrap(),
        mm_features: None, arrival_time: None, cache_salt: None, trace_headers: None,
        priority: 0, data_parallel_rank: None, reasoning_parser_kwargs: None, lora_request: None,
    }
}
async fn collect(mut stream: GenerateOutputStream) -> anyhow::Result<Vec<u32>> {
    let mut tokens=Vec::new();let mut finished=false;
    while let Some(output)=stream.next().await {
        let output=output?;
        anyhow::ensure!(!finished,"output after terminal response");
        tokens.extend(output.token_ids);finished=output.finish_reason.is_some();
    }
    anyhow::ensure!(finished,"stream ended without terminal response");
    Ok(tokens)
}
fn slow(backend:&Backend,id:&str,prompt:&[u32],outputs:u32)->anyhow::Result<mpsc::Receiver<Result<GenerateOutput>>> {
    // Exercise the same owning worker and bounded output transport directly,
    // so socket buffering cannot hide whether the consumer is actually paused.
    let cancel=Arc::new(AtomicBool::new(false));
    let (tx,rx)=mpsc::channel(TEST_QUEUE_CAPACITY);
    backend.active.lock().unwrap().insert(id.into(),cancel.clone());
    backend.tx.try_send(Job{req:request(id,prompt,outputs),tx,cancel})?;
    Ok(rx)
}
async fn full(rx:&mpsc::Receiver<Result<GenerateOutput>>)->anyhow::Result<()> {
    tokio::time::timeout(Duration::from_secs(60),async {
        while rx.len()<TEST_QUEUE_CAPACITY {
            anyhow::ensure!(!rx.is_closed(),"slow request stopped before filling its queue");
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
        Ok::<_,anyhow::Error>(())
    }).await??;
    Ok(())
}

#[tokio::test]
#[ignore = "requires a frozen TP8/EP8 bundle, checkpoint and native library on NPU"]
async fn npu_bounded_consumer_resumption_and_cancellation() -> anyhow::Result<()> {
    let path=|name:&str|->anyhow::Result<PathBuf>{Ok(std::env::var(name)?.into())};
    let cases:serde_json::Value=serde_json::from_slice(&std::fs::read(path("INFERFABRIC_TEST_CASES")?)?)?;
    let case=&cases[0];
    anyhow::ensure!(case["qualification"]["output_is_placeholder"]!=true,"placeholder reference");
    let prompt:Vec<u32>=serde_json::from_value(case["input_ids"].clone())?;
    let mut expected:Vec<u32>=serde_json::from_value(case["output_ids"].clone())?;
    let program:serde_json::Value=serde_json::from_slice(&std::fs::read(path("INFERFABRIC_TEST_PLAN")?.join("device-program.json"))?)?;
    let eos=program["eos"].as_u64().ok_or_else(||anyhow::anyhow!("missing plan EOS"))? as u32;
    if let Some(index)=expected.iter().position(|token|*token==eos){expected.truncate(index+1);}
    anyhow::ensure!(!prompt.is_empty() && !expected.is_empty(),"empty independent fixture");
    anyhow::ensure!(expected.len()>TEST_QUEUE_CAPACITY+1,"fixture too short to test a paused continuation");
    let outputs=expected.len() as u32;
    let options=crate::program_args::ProgramArgs::parse(&["--max-num-seqs".into(),"2".into(),"--max-num-batched-tokens".into(),"4096".into()])?;
    let backend=Backend::program(path("INFERFABRIC_TEST_PLAN")?,path("INFERFABRIC_TEST_CHECKPOINT")?,path("INFERFABRIC_TEST_LIBRARY")?,options).await?;
    // Always shut down the rank group, including on assertion/timeout errors.
    let outcome=tokio::time::timeout(Duration::from_secs(180),async {
        let baseline=collect(backend.generate(request("baseline",&prompt,outputs)).await?).await?;
        anyhow::ensure!(baseline==expected,"unthrottled baseline differs from EOS-scoped reference");
        let mut paused=slow(&backend,"paused",&prompt,outputs)?;full(&paused).await?;
        let peer=collect(backend.generate(request("peer",&prompt,expected.len() as u32)).await?).await?;
        anyhow::ensure!(peer==expected,"peer differs from independent reference");
        anyhow::ensure!(paused.len()==TEST_QUEUE_CAPACITY && !paused.is_closed(),"full request advanced or terminated while paused");
        let mut actual=Vec::new();let mut finished=false;
        while let Some(output)=paused.recv().await {
            let output=output?;anyhow::ensure!(!finished,"extra paused response");
            actual.extend(output.token_ids);finished=output.finish_reason.is_some();
        }
        anyhow::ensure!(finished && actual==baseline,"paused continuation dropped or changed tokens");
        let cancelled=slow(&backend,"cancelled",&prompt,outputs)?;full(&cancelled).await?;
        backend.abort(&["cancelled".into()]).await?;
        while backend.active.lock().unwrap().contains_key("cancelled") || !cancelled.is_closed() {
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
        anyhow::ensure!(cancelled.len()==TEST_QUEUE_CAPACITY && cancelled.is_closed(),"cancellation did not retire full queue");
        let recovered=collect(backend.generate(request("recovered",&prompt,expected.len() as u32)).await?).await?;
        anyhow::ensure!(recovered==expected,"slot reuse differs from independent reference");
        anyhow::ensure!(backend.healthy.load(Ordering::Acquire),"worker unhealthy after cancellation");
        println!("bounded consumer: {} EOS-scoped tokens conserved, peer reference passed, cancellation and reuse passed",outputs);
        Ok::<_,anyhow::Error>(())
    }).await;
    let shutdown=backend.shutdown().await;
    outcome??;shutdown?;
    anyhow::ensure!(backend.active.lock().unwrap().is_empty(),"request ownership leaked");
    Ok(())
}
