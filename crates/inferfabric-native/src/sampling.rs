use crate::{
    ffi::{Session, SessionPtr},
    write,
};
use inferfabric_model::{Result, invalid};
use std::collections::{BTreeMap, BTreeSet};
pub const VOCAB: usize = 248320;
#[derive(Debug, Clone)]
pub struct Sampling {
    pub temperature: f32,
    pub top_k: u32,
    pub top_p: f32,
    pub min_p: f32,
    pub repetition_penalty: f32,
    pub frequency_penalty: f32,
    pub presence_penalty: f32,
    pub min_tokens: u32,
}
impl Default for Sampling {
    fn default() -> Self {
        Self {
            temperature: 0.,
            top_k: 0,
            top_p: 1.,
            min_p: 0.,
            repetition_penalty: 1.,
            frequency_penalty: 0.,
            presence_penalty: 0.,
            min_tokens: 0,
        }
    }
}
impl Sampling {
    pub fn validate(&self) -> Result<()> {
        if !self.temperature.is_finite()
            || !(0. ..=2.).contains(&self.temperature)
            || self.top_k as usize > VOCAB
            || !self.top_p.is_finite()
            || self.top_p <= 0.
            || self.top_p > 1.
            || !self.min_p.is_finite()
            || !(0. ..=1.).contains(&self.min_p)
            || !self.repetition_penalty.is_finite()
            || self.repetition_penalty <= 0.
            || !(1. / self.repetition_penalty).is_finite()
            || (self.temperature > 0. && !(1. / self.temperature).is_finite())
            || !self.frequency_penalty.is_finite()
            || !(-2. ..=2.).contains(&self.frequency_penalty)
            || !self.presence_penalty.is_finite()
            || !(-2. ..=2.).contains(&self.presence_penalty)
        {
            return Err(invalid("invalid native sampling parameters"));
        }
        Ok(())
    }
}
type Prepare = unsafe extern "C" fn(
    SessionPtr,
    u64,
    u64,
    u64,
    u64,
    u64,
    u64,
    u64,
    u64,
    u64,
    i64,
    i64,
    i32,
    *mut u64,
) -> i32;
type RandomFill = unsafe extern "C" fn(SessionPtr, u64, i64, u64) -> i32;
pub(crate) struct Sampler {
    params: u64,
    counts: u64,
    seen: u64,
    mask: u64,
    controls: u64,
    random: u64,
    out: u64,
    graphs: [u64; 2],
    random_fill: RandomFill,
    capacity: usize,
    settings: Sampling,
    generated: usize,
    frequencies: BTreeMap<u32, u32>,
    stop: Vec<u32>,
}
fn floats(s: &Session<'_>, id: u64, values: &[f32]) -> Result<()> {
    write(
        s,
        id,
        &values
            .iter()
            .flat_map(|x| x.to_le_bytes())
            .collect::<Vec<_>>(),
    )
}
fn element(s: &Session<'_>, id: u64, index: u32, value: f32) -> Result<()> {
    s.api.check(unsafe {
        (s.api.write)(
            s.ptr,
            id,
            index as u64 * 4,
            value.to_le_bytes().as_ptr().cast(),
            4,
        )
    })
}
impl Sampler {
    pub(crate) fn prepare(s: &Session<'_>, logits: u64, capacity: usize) -> Result<Self> {
        let prepare: Prepare = unsafe {
            *s.api
                ._library
                .get(b"inferfabric_acl_sampler_prepare\0")
                .map_err(|e| invalid(e.to_string()))?
        };
        let random_fill: RandomFill = unsafe {
            *s.api
                ._library
                .get(b"inferfabric_acl_random_fill\0")
                .map_err(|e| invalid(e.to_string()))?
        };
        let mut sampler = Self {
            params: s.allocate(32)?,
            counts: s.allocate(VOCAB * 4)?,
            seen: s.allocate(VOCAB * 4)?,
            mask: s.allocate(VOCAB * 4)?,
            controls: s.allocate(24)?,
            random: s.allocate(capacity * 4)?,
            out: s.allocate(8)?,
            graphs: [0; 2],
            random_fill,
            capacity,
            settings: Sampling::default(),
            generated: 0,
            frequencies: BTreeMap::new(),
            stop: vec![],
        };
        sampler.configure(s, Sampling::default(), 0, &[], &[])?;
        let probabilities = s.allocate(VOCAB * 4)?;
        for greedy in 0..2 {
            let mut op = 0;
            s.api.check(unsafe {
                prepare(
                    s.ptr,
                    logits,
                    sampler.params,
                    sampler.counts,
                    sampler.seen,
                    sampler.mask,
                    sampler.controls,
                    sampler.random,
                    sampler.out,
                    probabilities,
                    VOCAB as i64,
                    capacity as i64,
                    greedy,
                    &mut op,
                )
            })?;
            s.api.check(unsafe { (s.api.execute)(s.ptr, op) })?;
            let expected = sampler.read_token(s)?;
            s.api.check(unsafe {
                (s.api.capture)(s.ptr, op, &mut sampler.graphs[greedy as usize])
            })?;
            s.api
                .check(unsafe { (s.api.replay)(s.ptr, sampler.graphs[greedy as usize]) })?;
            if expected != sampler.read_token(s)? {
                return Err(invalid("sampler startup eager/graph mismatch"));
            }
        }
        Ok(sampler)
    }
    pub(crate) fn configure(
        &mut self,
        s: &Session<'_>,
        settings: Sampling,
        seed: u64,
        prompt: &[u32],
        stop: &[u32],
    ) -> Result<()> {
        settings.validate()?;
        if settings.min_tokens as usize > self.capacity
            || prompt.iter().chain(stop).any(|&i| i as usize >= VOCAB)
        {
            return Err(invalid("sampler token or minimum length invalid"));
        }
        self.settings = settings;
        self.generated = 0;
        self.frequencies.clear();
        self.stop = stop.to_vec();
        let p = &self.settings;
        let inv_temp = if p.temperature == 0. {
            1.
        } else {
            1. / p.temperature
        };
        if !inv_temp.is_finite() {
            return Err(invalid("temperature reciprocal overflow"));
        }
        floats(
            s,
            self.params,
            &[
                inv_temp,
                0.,
                p.top_p,
                if p.min_p == 0. {
                    f32::NEG_INFINITY
                } else {
                    p.min_p.ln()
                },
                p.repetition_penalty,
                1. / p.repetition_penalty,
                p.frequency_penalty,
                p.presence_penalty,
            ],
        )?;
        let mut values = vec![0.; VOCAB];
        floats(s, self.counts, &values)?;
        floats(s, self.mask, &values)?;
        for &id in &prompt.iter().copied().collect::<BTreeSet<_>>() {
            values[id as usize] = 1.;
        }
        floats(s, self.seen, &values)?;
        if p.min_tokens > 0 {
            for &id in stop {
                element(s, self.mask, id, f32::NEG_INFINITY)?;
            }
        }
        write(
            s,
            self.controls,
            &[0i64, 0, self.k_index()]
                .into_iter()
                .flat_map(i64::to_le_bytes)
                .collect::<Vec<_>>(),
        )?;
        // CANN produces the complete request's random sequence on device before decoding.
        s.api
            .check(unsafe { (self.random_fill)(s.ptr, self.random, self.capacity as i64, seed) })
    }
    fn k_index(&self) -> i64 {
        if self.settings.top_k == 0 {
            VOCAB as i64 - 1
        } else {
            self.settings.top_k as i64 - 1
        }
    }
    fn read_token(&self, s: &Session<'_>) -> Result<u32> {
        let mut bytes = [0u8; 8];
        s.api
            .check(unsafe { (s.api.read)(s.ptr, self.out, 0, bytes.as_mut_ptr().cast(), 8) })?;
        let id = i64::from_le_bytes(bytes);
        if !(0..VOCAB as i64).contains(&id) {
            return Err(invalid("invalid NPU sampled token"));
        }
        Ok(id as u32)
    }
    pub(crate) fn draw(&mut self, s: &Session<'_>) -> Result<u32> {
        if self.generated >= self.capacity {
            return Err(invalid("sample capacity exceeded"));
        }
        if self.generated == self.settings.min_tokens as usize {
            for &id in &self.stop {
                element(s, self.mask, id, 0.)?;
            }
        }
        write(
            s,
            self.controls,
            &[0i64, self.generated as i64, self.k_index()]
                .into_iter()
                .flat_map(i64::to_le_bytes)
                .collect::<Vec<_>>(),
        )?;
        s.api.check(unsafe {
            (s.api.replay)(
                s.ptr,
                self.graphs[usize::from(self.settings.temperature == 0.)],
            )
        })?;
        let id = self.read_token(s)?;
        let count = self.frequencies.entry(id).or_default();
        *count += 1;
        element(s, self.counts, id, *count as f32)?;
        element(s, self.seen, id, 1.)?;
        self.generated += 1;
        Ok(id)
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn validates_sampling_ranges() {
        let mut p = Sampling {
            temperature: 1.,
            top_k: 20,
            presence_penalty: 2.,
            ..Sampling::default()
        };
        assert!(p.validate().is_ok());
        for bad in [f32::NAN, -1., f32::INFINITY] {
            p.temperature = bad;
            assert!(p.validate().is_err());
        }
        p.temperature = 1.;
        p.top_p = 0.;
        assert!(p.validate().is_err());
    }
}
