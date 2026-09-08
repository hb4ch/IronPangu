//! Continuous admission over fixed ACL graph lanes. One token per lane per replay;
//! a scheduler iteration may include several prefill rounds within its token budget.
use super::{Active, Job, error};
use std::sync::{
    Arc,
    atomic::{AtomicBool, Ordering},
};
use tokio::sync::mpsc;
use vllm_llm::{FinishReason, GenerateOutput, GeneratePromptInfo};
struct Running {
    job: Job,
    prompt_cursor: usize,
    generated: u32,
    next_input: Option<u32>,
    decoded_this_tick: bool,
}
impl Running {
    fn cancelled(&self) -> bool {
        self.job.cancel.load(Ordering::Acquire) || self.job.tx.is_closed()
    }
}
fn release(slot: &mut Option<Running>, active: &Active) {
    if let Some(running) = slot.take() {
        active.lock().unwrap().remove(&running.job.req.request_id);
    }
}
/// A round contains at most one token per lane; decode gets one token per tick.
fn select_round(eligible: &[bool], cursor: &mut usize, remaining: usize) -> Vec<usize> {
    let mut selected = Vec::new();
    for offset in 0..eligible.len() {
        let lane = (*cursor + offset) % eligible.len();
        if eligible[lane] && selected.len() < remaining {
            selected.push(lane);
        }
    }
    if let Some(&last) = selected.last() {
        *cursor = (last + 1) % eligible.len();
    }
    selected
}
pub(super) fn run(
    engine: &mut inferfabric_native::EngineSet<'_, '_>,
    rx: &mut mpsc::Receiver<Job>,
    active: &Active,
    stop: &Arc<AtomicBool>,
    token_budget: usize,
) -> inferfabric_model::Result<()> {
    let mut slots: Vec<Option<Running>> = (0..engine.batch_capacity()).map(|_| None).collect();
    let mut cursor = 0;
    let mut ticks = 0u64;
    let mut mixed_rounds = 0u64;
    let mut max_active = 0usize;
    let result = (|| {
        while !stop.load(Ordering::Acquire) {
            ticks += 1;
            for slot in slots.iter_mut().flatten() {
                slot.decoded_this_tick = false;
            }
            let mut remaining = token_budget;
            let mut tick_tokens = 0;
            while remaining > 0 && !stop.load(Ordering::Acquire) {
                for slot in &mut slots {
                    if slot.as_ref().is_some_and(Running::cancelled) {
                        release(slot, active);
                    }
                }
                // Slot addresses never change. Reset only the released lane before admission.
                for (lane, slot) in slots.iter_mut().enumerate() {
                    if slot.is_none() {
                        let job = match rx.try_recv() {
                            Ok(job) => job,
                            Err(_) => break,
                        };
                        if job.cancel.load(Ordering::Acquire) || job.tx.is_closed() {
                            active.lock().unwrap().remove(&job.req.request_id);
                            continue;
                        }
                        // Install before fallible initialization so errors reach this request too.
                        *slot = Some(Running {
                            job,
                            prompt_cursor: 0,
                            generated: 0,
                            next_input: None,
                            decoded_this_tick: false,
                        });
                        let running = slot.as_ref().unwrap();
                        engine.reset_slot(lane)?;
                        let p = &running.job.req.sampling_params;
                        let mut stops = p.stop_token_ids.clone();
                        stops.extend(p.eos_token_id);
                        engine.configure_slot(
                            lane,
                            crate::native_sampling::settings(p),
                            p.seed.map(|x| x as u64).unwrap_or_else(rand::random),
                            &running.job.req.prompt_token_ids,
                            &stops,
                        )?;
                    }
                }
                let active_now = slots.iter().flatten().count();
                if active_now > max_active {
                    max_active = active_now;
                    eprintln!(
                        "batch max_active={max_active} capacity={} token_budget={token_budget}",
                        engine.batch_capacity()
                    );
                }
                let eligible: Vec<bool> = slots
                    .iter()
                    .map(|slot| {
                        slot.as_ref().is_some_and(|r| {
                            r.prompt_cursor < r.job.req.prompt_token_ids.len()
                                || !r.decoded_this_tick
                        })
                    })
                    .collect();
                let selected = select_round(&eligible, &mut cursor, remaining);
                if selected.is_empty() {
                    break;
                }
                let mut prefill = 0;
                let mut decode = 0;
                let inputs: Vec<_> = selected
                    .iter()
                    .map(|&lane| {
                        let r = slots[lane].as_ref().unwrap();
                        let token = if r.prompt_cursor < r.job.req.prompt_token_ids.len() {
                            prefill += 1;
                            r.job.req.prompt_token_ids[r.prompt_cursor]
                        } else {
                            decode += 1;
                            r.next_input.expect("decode input")
                        };
                        (lane, token)
                    })
                    .collect();
                engine.consume_batch(&inputs)?;
                remaining -= inputs.len();
                tick_tokens += inputs.len();
                if prefill > 0 && decode > 0 {
                    mixed_rounds += 1;
                    if mixed_rounds == 1 || mixed_rounds.is_multiple_of(256) {
                        eprintln!(
                            "batch mixed_round={mixed_rounds} prefill={prefill} decode={decode} active={} tick_tokens={tick_tokens} budget={token_budget}",
                            inputs.len()
                        );
                    }
                }
                for lane in selected {
                    let r = slots[lane].as_mut().unwrap();
                    if r.cancelled() {
                        release(&mut slots[lane], active);
                        continue;
                    }
                    if r.prompt_cursor < r.job.req.prompt_token_ids.len() {
                        r.prompt_cursor += 1;
                        if r.prompt_cursor < r.job.req.prompt_token_ids.len() {
                            continue;
                        }
                    }
                    r.decoded_this_tick = true;
                    let next = engine.sample_slot(lane)?;
                    let p = &r.job.req.sampling_params;
                    r.generated += 1;
                    let finish = if p.eos_token_id == Some(next) {
                        Some(FinishReason::stop_eos())
                    } else if p.stop_token_ids.contains(&next) {
                        Some(FinishReason::Stop(Some(
                            vllm_engine_core_client::protocol::output::StopReason::TokenId(next),
                        )))
                    } else if r.generated >= p.max_tokens {
                        Some(FinishReason::Length)
                    } else {
                        None
                    };
                    let terminal = finish.is_some();
                    let output = GenerateOutput {
                        request_id: r.job.req.request_id.clone(),
                        prompt_info: (r.generated == 1).then(|| GeneratePromptInfo {
                            prompt_token_ids: r.job.req.prompt_token_ids.clone().into(),
                            prompt_logprobs: None,
                        }),
                        token_ids: vec![next],
                        logprobs: None,
                        finish_reason: finish,
                        cached_token_count: 0,
                        kv_transfer_params: None,
                    };
                    r.next_input = Some(next);
                    if r.job.tx.try_send(Ok(output)).is_err() || terminal {
                        release(&mut slots[lane], active);
                    }
                }
            }
            if tick_tokens == 0 {
                std::thread::sleep(std::time::Duration::from_millis(2));
            }
        }
        Ok(())
    })();
    if let Err(ref e) = result {
        for r in slots.iter().flatten() {
            let _ = r.job.tx.try_send(Err(error(format!("{e}"))));
        }
        while let Ok(job) = rx.try_recv() {
            let _ = job.tx.try_send(Err(error(format!("{e}"))));
        }
    }
    for slot in &mut slots {
        release(slot, active);
    }
    active.lock().unwrap().clear();
    eprintln!(
        "batch scheduler stopped ticks={ticks} mixed_rounds={mixed_rounds} max_active={max_active} token_budget={token_budget}"
    );
    result
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn round_respects_budget_and_rotates_without_starving_lanes() {
        let mut cursor = 0;
        let mut counts = [0; 4];
        for _ in 0..12 {
            let selected = select_round(&[true; 4], &mut cursor, 1);
            assert_eq!(selected.len(), 1);
            counts[selected[0]] += 1;
        }
        assert_eq!(counts, [3; 4]);
        assert_eq!(
            select_round(&[false, true, false, true], &mut cursor, 4).len(),
            2
        );
        assert!(select_round(&[false; 4], &mut cursor, 4).is_empty());
    }
}
