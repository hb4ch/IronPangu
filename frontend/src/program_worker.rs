//! Continuous admission with compiled packed prefill and single-token decode.
use super::{Active, Job, error};
use std::sync::{
    Arc,
    atomic::{AtomicBool, Ordering},
};
use tokio::sync::mpsc;
use vllm_llm::{Error, FinishReason, GenerateOutput, GeneratePromptInfo, GenerateRequest};

pub(super) fn validate(req: &GenerateRequest) -> vllm_llm::Result<()> {
    let p = &req.sampling_params;
    if p.temperature != 0.0
        || p.min_tokens != 0
        || p.presence_penalty != 0.0
        || p.frequency_penalty != 0.0
        || p.repetition_penalty != 1.0
    {
        return Err(Error::Unsupported { message: "program ABI v1 supports greedy sampling without penalties or min_tokens; non-greedy qualification is pending".into() });
    }
    Ok(())
}
pub(super) trait ResidentEngine {
    fn batch_capacity(&self)->inferfabric_model::Result<u32> {Ok(0)}
    fn batch(&mut self,_inputs:&[inferfabric_native::program::BatchInput])->inferfabric_model::Result<Vec<Option<u32>>> {
        Err(inferfabric_model::invalid("native sequence batching unavailable"))
    }
    fn uncaptured_prefill(&self)->bool {false}
    fn restore_prefix(&mut self,_slot:u32,_request:&GenerateRequest)->inferfabric_model::Result<u32> {Ok(0)}
    fn prefill_limit(&self,position:u32,prompt_tokens:u32)->u32 {prompt_tokens-position}
    fn checkpoint(&mut self,_slot:u32,_request:&GenerateRequest,_position:u32)->inferfabric_model::Result<()> {Ok(())}
    fn clear_prefix_cache(&mut self)->inferfabric_model::Result<()> {Ok(())}
    fn prefill_capacity(&self) -> u32 {
        1
    }
    fn prefill_chunk(&self, limit: u32) -> u32 {
        let n=self.prefill_capacity();
        if n<=limit {n} else {1}
    }
    fn prefill(
        &mut self,
        _tokens: &[u32],
        _position: u32,
        _sample: bool,
    ) -> inferfabric_model::Result<Option<u32>> {
        Err(inferfabric_model::invalid("packed prefill is unavailable"))
    }
    fn reset(&mut self, slot: u32) -> inferfabric_model::Result<()>;
    fn admit(&mut self, slot: u32, _tokens: u32) -> inferfabric_model::Result<()> {
        self.reset(slot)
    }
    fn release(&mut self, _slot: u32) -> inferfabric_model::Result<()> {
        Ok(())
    }
    fn select(&mut self, slot: u32) -> inferfabric_model::Result<()>;
    fn token(
        &mut self,
        token: u32,
        position: u32,
        sample: bool,
    ) -> inferfabric_model::Result<Option<u32>>;
}
impl ResidentEngine for inferfabric_native::program::Engine {
    fn batch_capacity(&self)->inferfabric_model::Result<u32> {self.batch_capacity()}
    fn batch(&mut self,inputs:&[inferfabric_native::program::BatchInput])->inferfabric_model::Result<Vec<Option<u32>>> {self.batch(inputs)}
    fn uncaptured_prefill(&self)->bool { self.uncaptured_prefill() }
    fn prefill_chunk(&self, limit: u32) -> u32 { self.prefill_chunk(limit) }
    fn admit(&mut self, slot: u32, tokens: u32) -> inferfabric_model::Result<()> {
        self.prepare_resident(slot, tokens)
    }
    fn release(&mut self, slot: u32) -> inferfabric_model::Result<()> {
        self.release_resident(slot)
    }
    fn prefill_capacity(&self) -> u32 {
        self.prefill_capacity()
    }
    fn prefill(
        &mut self,
        tokens: &[u32],
        position: u32,
        sample: bool,
    ) -> inferfabric_model::Result<Option<u32>> {
        self.prefill(tokens, position, sample)
    }
    fn reset(&mut self, slot: u32) -> inferfabric_model::Result<()> {
        self.reset_resident(slot)
    }
    fn select(&mut self, slot: u32) -> inferfabric_model::Result<()> {
        self.select_resident(slot)
    }
    fn token(
        &mut self,
        token: u32,
        position: u32,
        sample: bool,
    ) -> inferfabric_model::Result<Option<u32>> {
        self.token(token, position, sample)
    }
}
struct Running {
    job: Job,
    position: u32,
    generated: u32,
    next: Option<u32>,
    cached_tokens: u32,
}
fn cancelled(job: &Job) -> bool {
    job.cancel.load(Ordering::Acquire) || job.tx.is_closed()
}
fn release(slot: &mut Option<Running>, active: &Active) {
    if let Some(r) = slot.take() {
        active.lock().unwrap().remove(&r.job.req.request_id);
    }
}
fn advance(
    engine: &mut impl ResidentEngine,
    r: &mut Running,
    budget: usize,
) -> inferfabric_model::Result<(bool, usize)> {
    let prompt = &r.job.req.prompt_token_ids;
    let prefill = (r.position as usize) < prompt.len();
    let token = if prefill {
        prompt[r.position as usize]
    } else {
        r.next
            .ok_or_else(|| inferfabric_model::invalid("missing decode input"))?
    };
    let count = if prefill {
        let limit=engine.prefill_limit(r.position,prompt.len() as u32);
        if limit==0 || limit as usize>prompt.len()-r.position as usize {
            return Err(inferfabric_model::invalid("invalid prefix checkpoint boundary"));
        }
        engine.prefill_chunk(budget.min(limit as usize) as u32) as usize
    } else {1};
    let sample = r.position as usize + count >= prompt.len();
    let next = if count > 1 {
        engine.prefill(
            &prompt[r.position as usize..r.position as usize + count],
            r.position,
            sample,
        )?
    } else {
        engine.token(token, r.position, sample)?
    };
    finish_advance(r,count,next)
}
fn finish_advance(r:&mut Running,count:usize,next:Option<u32>)->inferfabric_model::Result<(bool,usize)> {
    let prompt=&r.job.req.prompt_token_ids;
    let sample=r.position as usize+count>=prompt.len();
    r.position += count as u32;
    if cancelled(&r.job) {
        return Ok((true, count));
    }
    if !sample {
        return Ok((false, count));
    }
    let token = next.ok_or_else(|| inferfabric_model::invalid("missing program output token"))?;
    r.next = Some(token);
    r.generated += 1;
    let p = &r.job.req.sampling_params;
    let finish = if p.eos_token_id == Some(token) {
        Some(FinishReason::stop_eos())
    } else if p.stop_token_ids.contains(&token) {
        Some(FinishReason::Stop(Some(
            vllm_engine_core_client::protocol::output::StopReason::TokenId(token),
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
            prompt_token_ids: prompt.clone().into(),
            prompt_logprobs: None,
        }),
        token_ids: vec![token],
        logprobs: None,
        finish_reason: finish,
        cached_token_count: r.cached_tokens as usize,
        kv_transfer_params: None,
    };
    Ok((r.job.tx.try_send(Ok(output)).is_err() || terminal, count))
}
pub(super) fn run(
    engine: &mut impl ResidentEngine,
    rx: &mut mpsc::Receiver<Job>,
    active: &Active,
    stop: &Arc<AtomicBool>,
    residents: usize,
    token_budget: usize,
) -> inferfabric_model::Result<()> {
    if residents == 0 || residents > 64 || token_budget == 0 {
        return Err(inferfabric_model::invalid(
            "invalid program scheduler capacity",
        ));
    }
    let mut slots: Vec<Option<Running>> = (0..residents).map(|_| None).collect();
    let chunk = engine.prefill_capacity();
    let batch_capacity=engine.batch_capacity()? as usize;
    let (mut cursor, mut max_active, mut mixed_ticks, mut consumed) = (0, 0, 0, 0u64);
    let mut result: inferfabric_model::Result<()> = (|| {
        while !stop.load(Ordering::Acquire) {
            for (index, slot) in slots.iter_mut().enumerate() {
                if slot.as_ref().is_some_and(|r| cancelled(&r.job)) {
                    engine.release(index as u32)?;
                    release(slot, active);
                }
                while slot.is_none() {
                    let job = match rx.try_recv() {
                        Ok(job) => job,
                        Err(_) => break,
                    };
                    if cancelled(&job) {
                        active.lock().unwrap().remove(&job.req.request_id);
                        continue;
                    }
                    // Install before fallible reset so failures reach this request.
                    // Every prior native invocation has completed on its own stream.
                    *slot = Some(Running {
                        job,
                        position: 0,
                        generated: 0,
                        next: None,
                        cached_tokens: 0,
                    });
                    let request = &slot.as_ref().unwrap().job.req;
                    let tokens = u32::try_from(request.prompt_token_ids.len())
                        .ok()
                        .and_then(|n| n.checked_add(request.sampling_params.max_tokens))
                        .ok_or_else(|| {
                            inferfabric_model::invalid("request reservation overflow")
                        })?;
                    engine.admit(index as u32, tokens)?;
                    let cached=engine.restore_prefix(index as u32,request)?;
                    if cached as usize>=request.prompt_token_ids.len() {
                        return Err(inferfabric_model::invalid("prefix restore must leave a token for logits"));
                    }
                    let running=slot.as_mut().unwrap();
                    running.position=cached;running.cached_tokens=cached;
                }
            }
            let count = slots.iter().flatten().count();
            if count > max_active {
                max_active = count;
                eprintln!(
                    "program scheduler active={count} capacity={residents} token_budget={token_budget} prefill_tokens={chunk} decode_tokens=1"
                );
            }
            if count == 0 {
                if rx.is_closed() {
                    break;
                }
                std::thread::sleep(std::time::Duration::from_millis(2));
                continue;
            }
            let start = cursor;
            let (mut tokens, mut prefill, mut decode) = (0, 0, 0);
            if batch_capacity>0 {
                let budget=token_budget.min(batch_capacity);
                let mut inputs=Vec::new();
                for offset in 0..residents {
                    let index=(start+offset)%residents;
                    let Some(r)=slots[index].as_ref() else{continue};
                    if cancelled(&r.job) || r.job.tx.capacity()==0 {continue;}
                    let prompt=&r.job.req.prompt_token_ids;
                    let extending=(r.position as usize)<prompt.len();
                    let count=if extending {
                        let limit=engine.prefill_limit(r.position,prompt.len() as u32) as usize;
                        if limit==0 || limit>prompt.len()-r.position as usize {return Err(inferfabric_model::invalid("invalid batch prefix boundary"));}
                        engine.prefill_chunk(limit.min(budget) as u32) as usize
                    } else{1};
                    if count==0 || count>budget {return Err(inferfabric_model::invalid("invalid batch quantum"));}
                    if count>budget-tokens {cursor=index;break;}
                    let ids=if extending {prompt[r.position as usize..r.position as usize+count].to_vec()}
                        else{vec![r.next.ok_or_else(||inferfabric_model::invalid("missing batch decode input"))?]};
                    inputs.push(inferfabric_native::program::BatchInput{resident:index as u32,position:r.position,tokens:ids,
                        sample:r.position as usize+count>=prompt.len()});
                    tokens+=count;if extending{prefill+=1}else{decode+=1};cursor=(index+1)%residents;
                }
                if !inputs.is_empty() {
                    let outputs=engine.batch(&inputs)?;
                    if outputs.len()!=inputs.len() || inputs.iter().zip(&outputs).any(|(i,o)|i.sample!=o.is_some()) {
                        return Err(inferfabric_model::invalid("batch output/member mismatch"));
                    }
                    for (input,output) in inputs.iter().zip(outputs) {
                        let index=input.resident as usize;let r=slots[index].as_mut().unwrap();
                        let (finished,_)=finish_advance(r,input.tokens.len(),output)?;
                        if !finished && !cancelled(&r.job) && (r.position as usize)<r.job.req.prompt_token_ids.len() {
                            engine.checkpoint(index as u32,&r.job.req,r.position)?;
                        }
                        if finished {engine.release(index as u32)?;release(&mut slots[index],active);}
                    }
                    consumed+=tokens as u64;
                } else {std::thread::sleep(std::time::Duration::from_millis(2));}
                if prefill>0 && decode>0 {mixed_ticks+=1;}
                continue;
            }
            for offset in 0..residents {
                if tokens == token_budget || stop.load(Ordering::Acquire) {
                    break;
                }
                let index = (start + offset) % residents;
                let Some(r) = slots[index].as_mut() else {
                    continue;
                };
                if cancelled(&r.job) {
                    engine.release(index as u32)?;
                    release(&mut slots[index], active);
                    continue;
                }
                // Only this worker produces outputs. A free slot therefore
                // cannot disappear before advance, except on receiver close.
                // Preserve state instead of computing and dropping a token.
                if r.job.tx.capacity()==0 {
                    continue;
                }
                if (r.position as usize) < r.job.req.prompt_token_ids.len() {
                    if engine.uncaptured_prefill() {
                        let remaining=r.job.req.prompt_token_ids.len()-r.position as usize;
                        let quantum=engine.prefill_chunk(remaining.min(token_budget) as u32) as usize;
                        if quantum>token_budget-tokens {
                            // Give this resident the next full quantum instead
                            // of fragmenting its forward after peer decode work.
                            cursor=index;
                            break;
                        }
                    }
                    prefill += 1;
                } else {
                    decode += 1;
                }
                engine.select(index as u32)?;
                let (finished, used) = advance(engine, r, token_budget - tokens)?;
                tokens += used;
                consumed += used as u64;
                if !finished && !cancelled(&r.job) && (r.position as usize)<r.job.req.prompt_token_ids.len() {
                    engine.checkpoint(index as u32,&r.job.req,r.position)?;
                }
                cursor = (index + 1) % residents;
                if finished {
                    engine.release(index as u32)?;
                    release(&mut slots[index], active);
                }
            }
            if prefill > 0 && decode > 0 {
                mixed_ticks += 1;
            }
            if tokens==0 {
                // All remaining clients may be backpressured. Avoid spinning;
                // retain bounded cancellation/shutdown polling latency.
                std::thread::sleep(std::time::Duration::from_millis(2));
            }
        }
        Ok(())
    })();
    rx.close();
    let message = result
        .as_ref()
        .err()
        .map(ToString::to_string)
        .unwrap_or_else(|| "program worker stopped".into());
    for (index,slot) in slots.iter_mut().enumerate() {
        if slot.is_some() && result.is_ok() {result=engine.release(index as u32);}
        if let Some(r) = slot.as_ref() {
            let _ = r.job.tx.try_send(Err(error(message.clone())));
        }
        release(slot, active);
    }
    while let Ok(job) = rx.try_recv() {
        let _ = job.tx.try_send(Err(error(message.clone())));
    }
    active.lock().unwrap().clear();
    if result.is_ok() {result=engine.clear_prefix_cache();}
    eprintln!(
        "program scheduler stopped max_active={max_active} mixed_ticks={mixed_ticks} consumed_tokens={consumed} token_budget={token_budget} prefill_tokens={chunk} decode_tokens=1"
    );
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex;
    fn request(id: &str, prompt: Vec<u32>, outputs: u32) -> GenerateRequest {
        GenerateRequest {
            request_id: id.into(),
            prompt_token_ids: prompt,
            sampling_params: serde_json::from_value(
                serde_json::json!({"temperature":0,"max_tokens":outputs}),
            )
            .unwrap(),
            mm_features: None,
            arrival_time: None,
            cache_salt: None,
            trace_headers: None,
            priority: 0,
            data_parallel_rank: None,
            reasoning_parser_kwargs: None,
            lora_request: None,
        }
    }
    #[derive(Default)]
    struct Fake {
        uncaptured: bool,
        selected: usize,
        position: [u32; 2],
        sum: [u32; 2],
        trace: Vec<(usize, u32)>,
        resets: Vec<u32>,
        fail_at: Option<usize>,
        cancel_at: Option<(usize, Arc<AtomicBool>)>,
        chunk: u32,
        family: Vec<u32>,
        chunks: Vec<(usize, u32, usize, bool)>,
        restored:u32,
        checkpoint_at:Option<u32>,
        checkpoints:Vec<u32>,
        cache_cleared:bool,
        drain_at:Option<(usize,mpsc::Receiver<vllm_llm::Result<GenerateOutput>>)>,
        drained:Vec<u32>,
        batched:bool,
        batches:Vec<Vec<inferfabric_native::program::BatchInput>>,
    }
    impl ResidentEngine for Fake {
        fn batch_capacity(&self)->inferfabric_model::Result<u32> {Ok(if self.batched{self.chunk}else{0})}
        fn batch(&mut self,inputs:&[inferfabric_native::program::BatchInput])->inferfabric_model::Result<Vec<Option<u32>>> {
            assert!(self.batched);assert!(inputs.iter().map(|i|i.tokens.len()).sum::<usize>()<=self.chunk as usize);
            self.batches.push(inputs.to_vec());
            inputs.iter().map(|i|{self.selected=i.resident as usize;self.prefill(&i.tokens,i.position,i.sample)}).collect()
        }
        fn uncaptured_prefill(&self)->bool {self.uncaptured}
        fn restore_prefix(&mut self,slot:u32,request:&GenerateRequest)->inferfabric_model::Result<u32> {
            self.position[slot as usize]=self.restored;
            self.sum[slot as usize]=request.prompt_token_ids[..self.restored as usize].iter().sum();
            Ok(self.restored)
        }
        fn prefill_limit(&self,position:u32,prompt:u32)->u32 {
            self.checkpoint_at.filter(|&boundary|position<boundary && boundary<prompt).unwrap_or(prompt)-position
        }
        fn checkpoint(&mut self,_slot:u32,_request:&GenerateRequest,position:u32)->inferfabric_model::Result<()> {
            if self.checkpoint_at==Some(position) {self.checkpoints.push(position);}
            Ok(())
        }
        fn clear_prefix_cache(&mut self)->inferfabric_model::Result<()> {self.cache_cleared=true;Ok(())}
        fn prefill_capacity(&self) -> u32 {
            self.family.last().copied().unwrap_or(self.chunk.max(1))
        }
        fn prefill_chunk(&self, limit: u32) -> u32 {
            if self.uncaptured {return limit.min(self.chunk);}
            if self.family.is_empty() { if self.chunk>1 && self.chunk<=limit {self.chunk} else {1} }
            else {self.family.iter().rev().copied().find(|&n|n<=limit).unwrap_or(1)}
        }
        fn prefill(
            &mut self,
            tokens: &[u32],
            position: u32,
            sample: bool,
        ) -> inferfabric_model::Result<Option<u32>> {
            assert!(if self.uncaptured {tokens.len()<=self.chunk as usize} else if self.family.is_empty() {tokens.len()==self.chunk as usize} else {self.family.contains(&(tokens.len() as u32))});
            self.chunks
                .push((self.selected, position, tokens.len(), sample));
            let mut output = None;
            for (i, &token) in tokens.iter().enumerate() {
                output = self.token(token, position + i as u32, sample && i + 1 == tokens.len())?;
            }
            Ok(output)
        }
        fn reset(&mut self, slot: u32) -> inferfabric_model::Result<()> {
            self.resets.push(slot);
            self.position[slot as usize] = 0;
            self.sum[slot as usize] = 0;
            Ok(())
        }
        fn select(&mut self, slot: u32) -> inferfabric_model::Result<()> {
            self.selected = slot as usize;
            Ok(())
        }
        fn token(
            &mut self,
            token: u32,
            position: u32,
            sample: bool,
        ) -> inferfabric_model::Result<Option<u32>> {
            if self.fail_at == Some(self.trace.len()) {
                return Err(inferfabric_model::invalid("injected rank failure"));
            }
            if let Some((at, flag)) = &self.cancel_at
                && *at == self.trace.len()
            {
                flag.store(true, Ordering::Release);
            }
            if let Some((at,receiver))=&mut self.drain_at
                && *at==self.trace.len()
            {
                self.drained.extend(receiver.try_recv().unwrap().unwrap().token_ids);
            }
            assert_eq!(position, self.position[self.selected]);
            self.position[self.selected] += 1;
            self.sum[self.selected] += token;
            self.trace.push((self.selected, token));
            Ok(sample.then_some(self.sum[self.selected]))
        }
    }
    fn job(
        req: GenerateRequest,
        active: &Active,
    ) -> (Job, mpsc::Receiver<vllm_llm::Result<GenerateOutput>>) {
        let (tx, rx) = mpsc::channel(32);
        let cancel = Arc::new(AtomicBool::new(false));
        active
            .lock()
            .unwrap()
            .insert(req.request_id.clone(), cancel.clone());
        (Job { req, tx, cancel }, rx)
    }
    #[test]
    fn backpressured_request_resumes_without_dropping_tokens_or_blocking_peer() {
        let active=Arc::new(Mutex::new(Default::default()));
        let (tx,mut rx)=mpsc::channel(2);
        let (mut slow,_)=job(request("slow",vec![1],2),&active);
        let (bounded,output)=mpsc::channel(1);slow.tx=bounded;
        let (peer,mut peer_output)=job(request("peer",vec![10],3),&active);
        tx.try_send(slow).unwrap();tx.try_send(peer).unwrap();drop(tx);
        // The peer's second step drains the slow client's first token. This
        // deterministic consumer event requires no timing assumptions.
        let mut engine=Fake{drain_at:Some((2,output)),..Default::default()};
        run(&mut engine,&mut rx,&active,&Arc::new(AtomicBool::new(false)),2,2).unwrap();
        assert_eq!(engine.trace.iter().map(|(slot,_)|*slot).collect::<Vec<_>>(),vec![0,1,1,0,1]);
        assert_eq!(engine.drained,vec![1]);
        let final_output=engine.drain_at.as_mut().unwrap().1.try_recv().unwrap().unwrap();
        assert_eq!(final_output.token_ids,vec![2]);assert!(final_output.finish_reason.is_some());
        assert_eq!(peer_output.try_recv().unwrap().unwrap().token_ids,vec![10]);
        assert_eq!(peer_output.try_recv().unwrap().unwrap().token_ids,vec![20]);
        assert_eq!(peer_output.try_recv().unwrap().unwrap().token_ids,vec![40]);
        assert!(active.lock().unwrap().is_empty());assert!(engine.cache_cleared);
    }
    #[test]
    fn full_output_queue_does_not_prevent_request_cancellation() {
        let active=Arc::new(Mutex::new(Default::default()));
        let (tx,mut rx)=mpsc::channel(2);
        let (mut slow,_)=job(request("slow",vec![1],100),&active);
        let (bounded,mut output)=mpsc::channel(1);slow.tx=bounded;
        let cancelled=slow.cancel.clone();
        let (peer,_peer_output)=job(request("peer",vec![10],3),&active);
        tx.try_send(slow).unwrap();tx.try_send(peer).unwrap();drop(tx);
        let mut engine=Fake{cancel_at:Some((3,cancelled)),..Default::default()};
        run(&mut engine,&mut rx,&active,&Arc::new(AtomicBool::new(false)),2,2).unwrap();
        assert_eq!(engine.trace.iter().map(|(slot,_)|*slot).collect::<Vec<_>>(),vec![0,1,1,1]);
        assert_eq!(output.try_recv().unwrap().unwrap().token_ids,vec![1]);
        assert!(matches!(output.try_recv(),Err(mpsc::error::TryRecvError::Disconnected)));
        assert!(active.lock().unwrap().is_empty());assert!(engine.cache_cleared);
    }
    #[test]
    fn packed_scheduler_submits_mixed_members_and_preserves_outputs() {
        let active=Arc::new(Mutex::new(Default::default()));let(tx,mut rx)=mpsc::channel(2);
        let (short,mut a)=job(request("short",vec![1],3),&active);
        let (long,mut b)=job(request("long",vec![1;9],2),&active);
        tx.try_send(short).unwrap();tx.try_send(long).unwrap();drop(tx);
        let mut engine=Fake{batched:true,uncaptured:true,chunk:8,..Default::default()};
        run(&mut engine,&mut rx,&active,&Arc::new(AtomicBool::new(false)),2,8).unwrap();
        let collect=|r:&mut mpsc::Receiver<vllm_llm::Result<GenerateOutput>>|{
            let mut tokens=Vec::new();while let Ok(value)=r.try_recv(){tokens.extend(value.unwrap().token_ids);}tokens};
        assert_eq!(collect(&mut a),vec![1,2,4]);assert_eq!(collect(&mut b),vec![9,18]);
        assert!(engine.batches.iter().any(|batch|batch.len()==2 && batch.iter().any(|i|i.resident==0 && i.position>=1)
            && batch.iter().any(|i|i.resident==1 && i.position<9)));
        assert!(engine.batches.iter().all(|b|b.iter().map(|i|i.tokens.len()).sum::<usize>()<=8));
        assert!(active.lock().unwrap().is_empty());
    }
    #[test]
    fn packed_scheduler_keeps_full_4k_prefill_quantum_beside_decode() {
        let active=Arc::new(Mutex::new(Default::default()));let(tx,mut rx)=mpsc::channel(2);
        let (short,_a)=job(request("short",vec![1],3),&active);
        let (long,mut b)=job(request("long",vec![1;8192],1),&active);
        tx.try_send(short).unwrap();tx.try_send(long).unwrap();drop(tx);
        let mut engine=Fake{batched:true,uncaptured:true,chunk:4096,..Default::default()};
        run(&mut engine,&mut rx,&active,&Arc::new(AtomicBool::new(false)),2,4096).unwrap();
        let long=engine.batches.iter().flatten().filter(|i|i.resident==1).map(|i|i.tokens.len()).collect::<Vec<_>>();
        assert_eq!(long,vec![4096,4096]);assert_eq!(b.try_recv().unwrap().unwrap().token_ids,vec![8192]);
    }
    #[test]
    fn packed_cancellation_drops_only_cancelled_output() {
        let active=Arc::new(Mutex::new(Default::default()));let(tx,mut rx)=mpsc::channel(2);
        let (a,mut ar)=job(request("cancel",vec![1;2],1),&active);let flag=a.cancel.clone();
        let (b,mut br)=job(request("peer",vec![3;2],1),&active);
        tx.try_send(a).unwrap();tx.try_send(b).unwrap();drop(tx);
        let mut engine=Fake{batched:true,uncaptured:true,chunk:8,cancel_at:Some((0,flag)),..Default::default()};
        run(&mut engine,&mut rx,&active,&Arc::new(AtomicBool::new(false)),2,8).unwrap();
        assert!(ar.try_recv().is_err());assert_eq!(br.try_recv().unwrap().unwrap().token_ids,vec![6]);
        assert_eq!(engine.batches.len(),1);assert_eq!(engine.batches[0].len(),2);
        assert!(active.lock().unwrap().is_empty());
    }
    #[test]
    fn packed_failure_terminates_members_and_queued_requests() {
        let active=Arc::new(Mutex::new(Default::default()));let(tx,mut rx)=mpsc::channel(3);
        let mut outputs=Vec::new();for id in ["a","b","queued"] {
            let (job,out)=job(request(id,vec![1;2],1),&active);tx.try_send(job).unwrap();outputs.push(out);
        }drop(tx);
        let mut engine=Fake{batched:true,uncaptured:true,chunk:8,fail_at:Some(1),..Default::default()};
        assert!(run(&mut engine,&mut rx,&active,&Arc::new(AtomicBool::new(false)),2,8).is_err());
        for out in &mut outputs {assert!(out.try_recv().unwrap().is_err());}
        assert!(active.lock().unwrap().is_empty());
    }
    #[test]
    fn program_admission_forwards_cache_salt_without_enabling_other_features() {
        let mut req=request("salted",vec![1;9],1);
        req.cache_salt=Some("isolated".into());
        assert!(crate::native_sampling::validate(&req).is_err());
        crate::native_sampling::validate_with_cache_salt(&req,true).unwrap();
        req.priority=1;
        assert!(crate::native_sampling::validate_with_cache_salt(&req,true).is_err());
    }
    #[test]
    fn restored_prefix_preserves_prompt_usage_and_checkpoint_boundary() {
        let active=Arc::new(Mutex::new(Default::default()));
        let (tx,mut rx)=mpsc::channel(2);
        let (job,mut output)=job(request("cached",vec![1;9],1),&active);
        tx.try_send(job).unwrap();drop(tx);
        let mut engine=Fake{restored:4,checkpoint_at:Some(8),family:vec![4,8],..Default::default()};
        run(&mut engine,&mut rx,&active,&Arc::new(AtomicBool::new(false)),1,8).unwrap();
        let result=output.try_recv().unwrap().unwrap();
        assert_eq!(result.token_ids,vec![9]);
        assert_eq!(result.cached_token_count,4);
        assert_eq!(result.prompt_info.unwrap().prompt_token_ids.len(),9);
        assert_eq!(engine.trace.len(),5);
        assert_eq!(engine.chunks,vec![(0,4,4,false)]);
        assert_eq!(engine.checkpoints,vec![8]);
        assert!(engine.cache_cleared);
        assert!(active.lock().unwrap().is_empty());
    }
    #[test]
    fn uncaptured_4k_forwards_follow_uncached_suffix_and_actual_tail() {
        for (prompt,restored,expected) in [
            (65536,0,vec![4096;16]),
            (65536,61440,vec![4096]),
            (65520,0,{let mut widths=vec![4096;15];widths.push(4080);widths}),
            (65536,60001,vec![4096,1439]),
        ] {
            let active=Arc::new(Mutex::new(Default::default()));
            let (tx,mut rx)=mpsc::channel(1);
            let (job,mut output)=job(request("long",vec![1;prompt],1),&active);
            tx.try_send(job).unwrap();drop(tx);
            let mut engine=Fake{uncaptured:true,chunk:4096,restored,..Default::default()};
            run(&mut engine,&mut rx,&active,&Arc::new(AtomicBool::new(false)),1,4096).unwrap();
            assert_eq!(engine.chunks.iter().map(|c|c.2).collect::<Vec<_>>(),expected);
            assert_eq!(engine.trace.len(),prompt-restored as usize);
            let result=output.try_recv().unwrap().unwrap();
            assert_eq!(result.token_ids,vec![prompt as u32]);
            assert_eq!(result.cached_token_count,restored as usize);
        }
    }
    #[test]
    fn uncaptured_prefill_defers_for_full_quantum_after_peer_decode() {
        let active=Arc::new(Mutex::new(Default::default()));
        let (tx,mut rx)=mpsc::channel(2);
        let (short,mut short_output)=job(request("short",vec![10],3),&active);
        let (long,mut long_output)=job(request("long",vec![1;8192],1),&active);
        tx.try_send(short).unwrap();tx.try_send(long).unwrap();drop(tx);
        let mut engine=Fake{uncaptured:true,chunk:4096,..Default::default()};
        run(&mut engine,&mut rx,&active,&Arc::new(AtomicBool::new(false)),2,4096).unwrap();
        assert_eq!(engine.chunks,vec![(1,0,4096,false),(1,4096,4096,true)]);
        assert_eq!(engine.trace[0],(0,10));
        assert!(engine.trace[1..engine.trace.len()-1].iter().any(|&(slot,_)|slot==0));
        for token in [10,20,40] {assert_eq!(short_output.try_recv().unwrap().unwrap().token_ids,vec![token]);}
        assert_eq!(long_output.try_recv().unwrap().unwrap().token_ids,vec![8192]);
    }
    #[test]
    fn interleaves_long_prefill_with_decode_and_reuses_only_finished_slot() {
        let active = Arc::new(Mutex::new(Default::default()));
        let (tx, mut rx) = mpsc::channel(4);
        let (a, mut ar) = job(request("a", vec![1; 6], 1), &active);
        let (b, mut br) = job(request("b", vec![10], 2), &active);
        let (c, mut cr) = job(request("c", vec![20], 1), &active);
        for j in [a, b, c] {
            tx.try_send(j).unwrap();
        }
        drop(tx);
        let mut engine = Fake::default();
        run(
            &mut engine,
            &mut rx,
            &active,
            &Arc::new(AtomicBool::new(false)),
            2,
            1,
        )
        .unwrap();
        assert_eq!(
            engine.trace,
            vec![
                (0, 1),
                (1, 10),
                (0, 1),
                (1, 10),
                (0, 1),
                (1, 20),
                (0, 1),
                (0, 1),
                (0, 1)
            ]
        );
        assert_eq!(engine.resets, vec![0, 1, 1]);
        assert_eq!(ar.try_recv().unwrap().unwrap().token_ids, vec![6]);
        assert_eq!(br.try_recv().unwrap().unwrap().token_ids, vec![10]);
        assert_eq!(br.try_recv().unwrap().unwrap().token_ids, vec![20]);
        assert_eq!(cr.try_recv().unwrap().unwrap().token_ids, vec![20]);
        assert!(active.lock().unwrap().is_empty());
    }
    #[test]
    fn packed_prefill_obeys_budget_tails_and_decode_fairness() {
        for budget in [1, 3, 4, 5, 8] {
            let active = Arc::new(Mutex::new(Default::default()));
            let (tx, mut rx) = mpsc::channel(4);
            let (a, mut ar) = job(request("long", vec![1; 9], 2), &active);
            let (b, mut br) = job(request("short", vec![10], 2), &active);
            tx.try_send(a).unwrap();
            tx.try_send(b).unwrap();
            drop(tx);
            let mut engine = Fake {
                chunk: 4,
                ..Default::default()
            };
            run(
                &mut engine,
                &mut rx,
                &active,
                &Arc::new(AtomicBool::new(false)),
                2,
                budget,
            )
            .unwrap();
            assert_eq!(ar.try_recv().unwrap().unwrap().token_ids, vec![9]);
            assert_eq!(ar.try_recv().unwrap().unwrap().token_ids, vec![18]);
            assert_eq!(br.try_recv().unwrap().unwrap().token_ids, vec![10]);
            assert_eq!(br.try_recv().unwrap().unwrap().token_ids, vec![20]);
            assert!(
                engine
                    .trace
                    .iter()
                    .position(|&(slot, token)| slot == 1 && token == 10)
                    .unwrap()
                    <= 4
            );
            if budget < 4 {
                assert!(engine.chunks.is_empty());
            } else {
                assert!(!engine.chunks.is_empty());
            }
            assert!(
                engine
                    .chunks
                    .iter()
                    .all(|&(_, pos, count, sample)| count <= budget
                        && pos + count as u32 <= 9
                        && !sample)
            );
            assert!(active.lock().unwrap().is_empty());
        }
    }
    #[test]
    fn family_uses_only_declared_shapes_within_budget_and_prompt() {
        for budget in [3, 8, 16] {
            let active=Arc::new(Mutex::new(Default::default()));
            let (tx,mut rx)=mpsc::channel(4);
            let (a,mut ar)=job(request("family",vec![1;15],1),&active);
            tx.try_send(a).unwrap();drop(tx);
            let mut engine=Fake {family:vec![2,4,8,16],..Default::default()};
            run(&mut engine,&mut rx,&active,&Arc::new(AtomicBool::new(false)),1,budget).unwrap();
            assert_eq!(ar.try_recv().unwrap().unwrap().token_ids,vec![15]);
            assert!(engine.chunks.iter().all(|&(_,pos,count,_)|count<=budget && pos+count as u32<=15));
            if budget>=8 {assert_eq!(engine.chunks.iter().map(|x|x.2).collect::<Vec<_>>(),vec![8,4,2]);}
            else {assert!(engine.chunks.iter().all(|x|x.2==2));}
            assert!(active.lock().unwrap().is_empty());
        }
    }
    #[test]
    fn final_chunk_samples_once_and_cancelled_chunk_emits_nothing() {
        for cancel in [false, true] {
            let active = Arc::new(Mutex::new(Default::default()));
            let (tx, mut rx) = mpsc::channel(4);
            let (a, mut ar) = job(request("chunk", vec![2; 8], 1), &active);
            let flag = a.cancel.clone();
            tx.try_send(a).unwrap();
            drop(tx);
            let mut engine = Fake {
                chunk: 4,
                cancel_at: cancel.then_some((1, flag)),
                ..Default::default()
            };
            run(
                &mut engine,
                &mut rx,
                &active,
                &Arc::new(AtomicBool::new(false)),
                1,
                4,
            )
            .unwrap();
            if cancel {
                assert!(ar.try_recv().is_err());
                assert_eq!(engine.chunks.len(), 1);
            } else {
                assert_eq!(ar.try_recv().unwrap().unwrap().token_ids, vec![16]);
                assert_eq!(engine.chunks, vec![(0, 0, 4, false), (0, 4, 4, true)]);
            }
            assert!(active.lock().unwrap().is_empty());
        }
    }
    #[test]
    fn rank_failure_reaches_all_residents_and_queued_requests() {
        let active = Arc::new(Mutex::new(Default::default()));
        let (tx, mut rx) = mpsc::channel(4);
        let mut receivers = Vec::new();
        for id in ["a", "b", "c"] {
            let (j, r) = job(request(id, vec![1; 3], 2), &active);
            tx.try_send(j).unwrap();
            receivers.push(r);
        }
        let mut engine = Fake {
            fail_at: Some(1),
            ..Default::default()
        };
        assert!(
            run(
                &mut engine,
                &mut rx,
                &active,
                &Arc::new(AtomicBool::new(false)),
                2,
                2
            )
            .is_err()
        );
        for mut r in receivers {
            assert!(r.try_recv().unwrap().is_err());
        }
        assert!(active.lock().unwrap().is_empty());
        assert!(tx.is_closed());
    }
    #[test]
    fn queued_cancellation_and_disconnect_do_not_consume_a_slot() {
        let active = Arc::new(Mutex::new(Default::default()));
        let (tx, mut rx) = mpsc::channel(4);
        let (a, _ar) = job(request("cancelled", vec![1], 1), &active);
        a.cancel.store(true, Ordering::Release);
        let (b, br) = job(request("disconnected", vec![2], 1), &active);
        drop(br);
        let (c, mut cr) = job(request("live", vec![3], 1), &active);
        for j in [a, b, c] {
            tx.try_send(j).unwrap();
        }
        drop(tx);
        let mut engine = Fake::default();
        run(
            &mut engine,
            &mut rx,
            &active,
            &Arc::new(AtomicBool::new(false)),
            2,
            2,
        )
        .unwrap();
        assert_eq!(engine.trace, vec![(0, 3)]);
        assert_eq!(engine.resets, vec![0]);
        assert!(cr.try_recv().unwrap().unwrap().finish_reason.is_some());
        assert!(active.lock().unwrap().is_empty());
    }
    #[test]
    fn active_cancellation_during_native_step_does_not_emit_or_reset_peer() {
        let active = Arc::new(Mutex::new(Default::default()));
        let (tx, mut rx) = mpsc::channel(4);
        let (a, mut ar) = job(request("a", vec![1; 4], 1), &active);
        let flag = a.cancel.clone();
        let (b, mut br) = job(request("b", vec![2; 4], 1), &active);
        let (c, mut cr) = job(request("c", vec![3], 1), &active);
        for j in [a, b, c] {
            tx.try_send(j).unwrap();
        }
        drop(tx);
        let mut engine = Fake {
            cancel_at: Some((0, flag)),
            ..Default::default()
        };
        run(
            &mut engine,
            &mut rx,
            &active,
            &Arc::new(AtomicBool::new(false)),
            2,
            2,
        )
        .unwrap();
        assert_eq!(engine.resets, vec![0, 1, 0]);
        assert!(ar.try_recv().is_err());
        assert_eq!(br.try_recv().unwrap().unwrap().token_ids, vec![8]);
        assert_eq!(cr.try_recv().unwrap().unwrap().token_ids, vec![3]);
        assert!(active.lock().unwrap().is_empty());
    }
    #[test]
    fn shutdown_during_native_step_closes_active_and_queued_streams() {
        let active = Arc::new(Mutex::new(Default::default()));
        let stop = Arc::new(AtomicBool::new(false));
        let (tx, mut rx) = mpsc::channel(4);
        let mut receivers = Vec::new();
        for id in ["a", "b", "queued"] {
            let (j, r) = job(request(id, vec![1; 3], 2), &active);
            tx.try_send(j).unwrap();
            receivers.push(r);
        }
        let mut engine = Fake {
            cancel_at: Some((0, stop.clone())),
            ..Default::default()
        };
        run(&mut engine, &mut rx, &active, &stop, 2, 2).unwrap();
        assert_eq!(engine.trace.len(), 1);
        for mut r in receivers {
            assert!(r.try_recv().unwrap().is_err());
        }
        assert!(active.lock().unwrap().is_empty());
        assert!(tx.is_closed());
    }
    #[test]
    fn initial_program_capabilities_reject_unimplemented_sampling() {
        let mut request = GenerateRequest {
            request_id: "test".into(),
            prompt_token_ids: vec![1],
            sampling_params: serde_json::from_value(
                serde_json::json!({"temperature":0,"max_tokens":2}),
            )
            .unwrap(),
            mm_features: None,
            arrival_time: None,
            cache_salt: None,
            trace_headers: None,
            priority: 0,
            data_parallel_rank: None,
            reasoning_parser_kwargs: None,
            lora_request: None,
        };
        assert!(validate(&request).is_ok());
        request.sampling_params.temperature = 0.7;
        assert!(validate(&request).is_err());
        request.sampling_params.temperature = 0.;
        request.sampling_params.min_tokens = 1;
        assert!(validate(&request).is_err());
        request.sampling_params.min_tokens = 0;
        request.sampling_params.repetition_penalty = 1.1;
        assert!(validate(&request).is_err());
    }
}
