//! Weight-independent HTTP acceptance client; run inside the development container.
use anyhow::{Result, ensure};
use serde_json::{Value, json};
use std::time::Duration;

pub async fn run(base: &str) -> Result<()> {
    let client = reqwest::Client::builder()
        .timeout(Duration::from_secs(20))
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
    let body = json!({"model":"ironpangu-mock","messages":[{"role":"user","content":"hello world"}],"temperature":0,"max_tokens":4});
    let url = format!("{base}/v1/chat/completions");
    let response: Value = client
        .post(&url)
        .json(&body)
        .send()
        .await?
        .error_for_status()?
        .json()
        .await?;
    ensure!(
        response["usage"]["completion_tokens"] == 4,
        "wrong token usage: {response}"
    );
    let expected = response["choices"][0]["message"]["content"]
        .as_str()
        .unwrap_or_default();
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
    let mut text = String::new();
    let mut terminal = false;
    let mut done = false;
    let mut chunks = 0;
    for line in sse.lines().filter_map(|l| l.strip_prefix("data: ")) {
        if line == "[DONE]" {
            done = true;
            continue;
        }
        let event: Value = serde_json::from_str(line)?;
        ensure!(event.get("error").is_none(), "stream error: {event}");
        if let Some(content) = event["choices"][0]["delta"]["content"].as_str() {
            text.push_str(content);
            chunks += 1;
        }
        terminal |= event["choices"][0]["finish_reason"] == "length";
    }
    ensure!(
        done && terminal && chunks > 1 && text == expected,
        "stream/collected mismatch"
    );
    let calls = (0..4).map(|_| async {
        let result: Value = client
            .post(&url)
            .json(&body)
            .send()
            .await?
            .error_for_status()?
            .json()
            .await?;
        ensure!(
            result["choices"][0]["message"]["content"] == expected,
            "concurrent output mismatch"
        );
        Ok::<_, anyhow::Error>(())
    });
    futures::future::try_join_all(calls).await?;
    let mut unsupported = body.clone();
    unsupported["temperature"] = json!(1);
    ensure!(
        client
            .post(&url)
            .json(&unsupported)
            .send()
            .await?
            .status()
            .as_u16()
            == 400,
        "unsupported sampling must return 400"
    );
    println!(
        "{}",
        json!({"health":"passed","chat":"passed","stream_agreement":"passed","parallel_requests":4,"unsupported_sampling":"passed","backend":"synthetic mock"})
    );
    Ok(())
}
