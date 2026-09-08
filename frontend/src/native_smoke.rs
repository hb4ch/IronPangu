//! Actual checkpoint HTTP acceptance; execute only inside the NPU container.
use anyhow::{Result, ensure};
use serde_json::{Value, json};
use std::time::Duration;
pub async fn run(base: &str) -> Result<()> {
    let client = reqwest::Client::builder()
        .timeout(Duration::from_secs(120))
        .build()?;
    ensure!(
        client
            .get(format!("{base}/health"))
            .send()
            .await?
            .status()
            .is_success(),
        "not ready"
    );
    let url = format!("{base}/v1/chat/completions");
    let body = json!({"model":"inferfabric-qwen35","messages":[{"role":"user","content":"What is 2 + 2? Answer briefly."}],"temperature":0,"max_tokens":16,"chat_template_kwargs":{"enable_thinking":false}});
    let first: Value = client
        .post(&url)
        .json(&body)
        .send()
        .await?
        .error_for_status()?
        .json()
        .await?;
    ensure!(
        first["choices"][0]["message"]["content"] == "4",
        "unexpected arithmetic answer: {first}"
    );
    ensure!(
        first["choices"][0]["finish_reason"] == "stop",
        "EOS did not terminate"
    );
    let mut stream = body.clone();
    stream["stream"] = json!(true);
    let sse = client
        .post(&url)
        .json(&stream)
        .send()
        .await?
        .error_for_status()?
        .text()
        .await?;
    let mut text = String::new();
    let mut done = false;
    let mut terminal = false;
    for line in sse.lines().filter_map(|l| l.strip_prefix("data: ")) {
        if line == "[DONE]" {
            done = true;
            continue;
        }
        let event: Value = serde_json::from_str(line)?;
        ensure!(event.get("error").is_none(), "stream error: {event}");
        if let Some(t) = event["choices"][0]["delta"]["content"].as_str() {
            text.push_str(t);
        }
        terminal |= event["choices"][0]["finish_reason"] == "stop";
    }
    ensure!(done && terminal && text == "4", "SSE mismatch: {sse}");
    let completion_url = format!("{base}/v1/completions");
    let different = json!({"model":"inferfabric-qwen35","prompt":"The capital of France is","temperature":0,"max_tokens":8});
    let completion: Value = client
        .post(&completion_url)
        .json(&different)
        .send()
        .await?
        .error_for_status()?
        .json()
        .await?;
    ensure!(
        completion["choices"][0]["text"]
            .as_str()
            .unwrap_or_default()
            .starts_with(" Paris"),
        "completion mismatch: {completion}"
    );
    let again: Value = client
        .post(&url)
        .json(&body)
        .send()
        .await?
        .error_for_status()?
        .json()
        .await?;
    ensure!(
        again["choices"] == first["choices"] && again["usage"] == first["usage"],
        "request state leaked"
    );
    for (field, value) in [("logprobs", json!(1))] {
        let mut invalid = body.clone();
        invalid[field] = value;
        ensure!(
            client
                .post(&url)
                .json(&invalid)
                .send()
                .await?
                .status()
                .as_u16()
                == 400,
            "unsupported {field} was accepted"
        );
    }
    let models: Value = client
        .get(format!("{base}/v1/models"))
        .send()
        .await?
        .error_for_status()?
        .json()
        .await?;
    let context = models["data"][0]["max_model_len"]
        .as_u64()
        .ok_or_else(|| anyhow::anyhow!("missing context metadata"))? as usize;
    let oversized = json!({"model":"inferfabric-qwen35","prompt":vec![1u32;context + 1],"temperature":0,"max_tokens":1});
    ensure!(
        client
            .post(&completion_url)
            .json(&oversized)
            .send()
            .await?
            .status()
            .as_u16()
            == 400,
        "oversized prompt was accepted"
    );
    if context > 512 {
        let request = json!({"model":"inferfabric-qwen35","prompt":vec![1u32;512],"temperature":0,"max_tokens":2,"ignore_eos":true});
        let response: Value = client
            .post(&completion_url)
            .json(&request)
            .send()
            .await?
            .error_for_status()?
            .json()
            .await?;
        ensure!(
            response["usage"]["prompt_tokens"] == 512,
            "long prompt not consumed: {response}"
        );
        ensure!(
            response["usage"]["completion_tokens"] == 2,
            "long prompt completion failed: {response}"
        );
    }
    // Hold a streaming request while submitting another. The native admission policy is explicit.
    let long = json!({"model":"inferfabric-qwen35","prompt":vec![1u32;80],"temperature":0,"max_tokens":40,"ignore_eos":true,"stream":true});
    let pending = client
        .post(&completion_url)
        .json(&long)
        .send()
        .await?
        .error_for_status()?;
    let overlap = client.post(&url).json(&body).send().await?;
    let overlap_body: Value = overlap.error_for_status()?.json().await?;
    ensure!(
        overlap_body["choices"][0]["message"]["content"] == "4",
        "overlap failed: {overlap_body}"
    );
    drop(pending); // Closing SSE must cancel and release admission.
    let mut recovered = None;
    for _ in 0..50 {
        let response = client.post(&url).json(&body).send().await?;
        if response.status().is_success() {
            recovered = Some(response.json::<Value>().await?);
            break;
        }
        tokio::time::sleep(Duration::from_millis(20)).await;
    }
    ensure!(
        recovered
            .as_ref()
            .is_some_and(|r| r["choices"] == first["choices"]),
        "cancellation did not reclaim request state"
    );
    println!(
        "{}",
        json!({"backend":"native experimental","frontend":"vLLM Rust 0.25.1","health":"passed","chat":first,"completion":completion,"sse_agreement":"passed","request_reset":"passed","unsupported_sampling":"passed","context_limit":"passed","overlap_admission":"passed","cancel_recovery":"passed"})
    );
    Ok(())
}
pub async fn sampling(base: &str) -> Result<()> {
    let client = reqwest::Client::builder()
        .timeout(Duration::from_secs(120))
        .build()?;
    let url = format!("{base}/v1/completions");
    async fn call(c: &reqwest::Client, url: &str, body: &Value) -> Result<Value> {
        let response = c.post(url).json(body).send().await?;
        let status = response.status();
        let text = response.text().await?;
        ensure!(status.is_success(), "HTTP {status}: {text}");
        Ok(serde_json::from_str(&text)?)
    }
    let body = json!({"model":"inferfabric-qwen35","prompt":"Once upon a time,","max_tokens":12,"seed":42,"temperature":1.1,"top_p":0.92,"top_k":20,"min_p":0.03,"presence_penalty":1.2,"frequency_penalty":0.2,"repetition_penalty":1.1});
    let first = call(&client, &url, &body).await?;
    let second = call(&client, &url, &body).await?;
    ensure!(
        first["choices"] == second["choices"],
        "seeded requests differ"
    );
    let mut alternatives = std::collections::BTreeSet::new();
    for seed in 1..5 {
        let mut changed = body.clone();
        changed["seed"] = json!(seed);
        let r = call(&client, &url, &changed).await?;
        alternatives.insert(r["choices"][0]["text"].to_string());
    }
    ensure!(alternatives.len() > 1, "different seeds did not vary");
    let mut streamed = body.clone();
    streamed["stream"] = json!(true);
    let sse = client
        .post(&url)
        .json(&streamed)
        .send()
        .await?
        .error_for_status()?
        .text()
        .await?;
    let mut collected = String::new();
    let mut done = false;
    for line in sse.lines().filter_map(|l| l.strip_prefix("data: ")) {
        if line == "[DONE]" {
            done = true;
            continue;
        }
        let event: Value = serde_json::from_str(line)?;
        ensure!(event.get("error").is_none(), "stream error {event}");
        collected.push_str(event["choices"][0]["text"].as_str().unwrap_or_default());
    }
    ensure!(
        done && collected == first["choices"][0]["text"].as_str().unwrap_or_default(),
        "seeded SSE mismatch"
    );
    let defaults = json!({"model":"inferfabric-qwen35","prompt":"Once upon a time,","max_tokens":12,"seed":314});
    let inherited = call(&client, &url, &defaults).await?;
    let mut explicit = defaults.clone();
    for (k, v) in [
        ("temperature", json!(1.)),
        ("top_k", json!(20)),
        ("top_p", json!(1.)),
        ("presence_penalty", json!(2.)),
        ("frequency_penalty", json!(0.)),
        ("repetition_penalty", json!(1.)),
        ("min_p", json!(0.)),
    ] {
        explicit[k] = v;
    }
    let explicit_result = call(&client, &url, &explicit).await?;
    ensure!(
        inherited["choices"] == explicit_result["choices"],
        "model defaults mismatch"
    );
    let mut greedy = body.clone();
    greedy["temperature"] = json!(0);
    let g1 = call(&client, &url, &greedy).await?;
    greedy["seed"] = json!(999);
    let g2 = call(&client, &url, &greedy).await?;
    ensure!(g1["choices"] == g2["choices"], "greedy depends on seed");
    let stop = json!({"model":"inferfabric-qwen35","prompt":[1,2,3],"temperature":0,"presence_penalty":0,"max_tokens":4,"stop_token_ids":[16],"ignore_eos":true});
    let stopped = call(&client, &url, &stop).await?;
    ensure!(
        stopped["usage"]["completion_tokens"] == 1,
        "stop token mismatch {stopped}"
    );
    let mut minimum = stop.clone();
    minimum["min_tokens"] = json!(2);
    let minimum_result = call(&client, &url, &minimum).await?;
    ensure!(
        minimum_result["usage"]["completion_tokens"]
            .as_u64()
            .unwrap_or(0)
            >= 2,
        "min_tokens ignored"
    );
    println!(
        "{}",
        json!({"sampling":"NPU","seed_replay":"passed","distinct_seed_outputs":alternatives.len(),"sse_seed_agreement":"passed","model_card_defaults":"passed","greedy_seed_independence":"passed","minimum_tokens":"passed","sample":first,"default_sample":inherited})
    );
    Ok(())
}
