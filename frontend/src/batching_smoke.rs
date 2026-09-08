//! HTTP qualification of interleaved prefill/decode, queueing and slot reuse.
use anyhow::{Result, ensure};
use futures::StreamExt;
use serde_json::{Value, json};
async fn call(client: &reqwest::Client, url: &str, body: &Value) -> Result<Value> {
    Ok(client
        .post(url)
        .json(body)
        .send()
        .await?
        .error_for_status()?
        .json()
        .await?)
}
pub async fn run(base: &str) -> Result<()> {
    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(180))
        .build()?;
    let url = format!("{base}/v1/completions");
    let requests: Vec<Value> = (0..4)
        .map(|i| {
            json!({"model":"inferfabric-qwen35",
        "prompt":vec![1u32+i;[5,37,181,257][i as usize]],"max_tokens":16,"ignore_eos":true,
        "temperature":if i==0 {0.0}else{0.8},"top_k":20,"top_p":0.9,"seed":42+i,
        "presence_penalty":0.7,"frequency_penalty":0.2,"repetition_penalty":1.1})
        })
        .collect();
    let mut expected = Vec::new();
    for request in &requests {
        expected.push(call(&client, &url, request).await?);
    }
    // More requests than device slots must queue and finish with isolated state and RNG.
    let responses =
        futures::future::join_all((0..12).map(|i| call(&client, &url, &requests[i % 4]))).await;
    for (i, response) in responses.into_iter().enumerate() {
        let response = response?;
        let baseline = &expected[i % 4];
        ensure!(
            response["choices"] == baseline["choices"] && response["usage"] == baseline["usage"],
            "interleaved request {i} differs: baseline={baseline} result={response}"
        );
    }
    // Start a decode stream, observe its first event, then admit a longer prefill.
    let decode = json!({"model":"inferfabric-qwen35","prompt":"Once upon a time","max_tokens":512,
        "ignore_eos":true,"temperature":0.8,"seed":19,"stream":true});
    let response = client
        .post(&url)
        .json(&decode)
        .send()
        .await?
        .error_for_status()?;
    let mut stream = response.bytes_stream();
    ensure!(
        stream.next().await.transpose()?.is_some(),
        "decode stream empty"
    );
    let mixed = call(&client, &url, &requests[3]).await?;
    ensure!(
        mixed["choices"] == expected[3]["choices"],
        "mixed prefill differs"
    );
    drop(stream); // Cancel ongoing decode; its slot must be reset before reuse.
    let repeated = call(&client, &url, &requests[0]).await?;
    ensure!(
        repeated["choices"] == expected[0]["choices"],
        "state leaked after cancellation"
    );
    // Cancel during long prefill before the first generated token.
    let pending = json!({"model":"inferfabric-qwen35","prompt":vec![2u32;1500],"max_tokens":16,"stream":true});
    drop(
        client
            .post(&url)
            .json(&pending)
            .send()
            .await?
            .error_for_status()?,
    );
    let recovered = call(&client, &url, &requests[1]).await?;
    ensure!(
        recovered["choices"] == expected[1]["choices"],
        "prefill cancellation damaged another slot"
    );
    ensure!(
        client
            .get(format!("{base}/health"))
            .send()
            .await?
            .status()
            .is_success(),
        "health failed"
    );
    println!(
        "{}",
        json!({"parallel_requests":12,"isolated_baseline_agreement":"passed",
        "seed_and_penalty_isolation":"passed","mixed_prefill_decode":"passed","prefill_and_decode_cancellation":"passed","slot_reuse":"passed"})
    );
    Ok(())
}
