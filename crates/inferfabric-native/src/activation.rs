//! Conservative live intervals for values crossing lowered layer boundaries.
use crate::ffi::{Session, SessionPtr};
use inferfabric_model::{Result, invalid};

const ALIGNMENT: usize = 512;
#[derive(Debug, Clone, Copy)]
struct Interval {
    bytes: usize,
    first: usize,
    last: usize,
}
#[derive(Debug)]
struct Plan {
    bytes: usize,
    offsets: Vec<usize>,
}
fn align(bytes: usize) -> Result<usize> {
    bytes
        .checked_add(ALIGNMENT - 1)
        .map(|n| n / ALIGNMENT * ALIGNMENT)
        .ok_or_else(|| invalid("activation alignment overflow"))
}
fn plan(intervals: &[Interval]) -> Result<Plan> {
    let mut order: Vec<usize> = (0..intervals.len()).collect();
    order.sort_by_key(|&i| (intervals[i].first, i));
    let mut offsets = vec![0; intervals.len()];
    let mut placed: Vec<usize> = vec![];
    let mut bytes = 0;
    for i in order {
        let value = intervals[i];
        if value.bytes == 0 || value.first > value.last {
            return Err(invalid("invalid activation live interval"));
        }
        let size = align(value.bytes)?;
        let mut conflicts: Vec<_> = placed
            .iter()
            .copied()
            .filter(|&j| intervals[j].last >= value.first && intervals[j].first <= value.last)
            .collect();
        conflicts.sort_by_key(|&j| offsets[j]);
        let mut offset: usize = 0;
        for j in conflicts {
            let end = offset
                .checked_add(size)
                .ok_or_else(|| invalid("activation arena overflow"))?;
            if end <= offsets[j] {
                break;
            }
            offset = offset.max(
                offsets[j]
                    .checked_add(align(intervals[j].bytes)?)
                    .ok_or_else(|| invalid("activation arena overflow"))?,
            );
        }
        bytes = bytes.max(
            offset
                .checked_add(size)
                .ok_or_else(|| invalid("activation arena overflow"))?,
        );
        offsets[i] = offset;
        placed.push(i);
    }
    Ok(Plan { bytes, offsets })
}
pub(crate) fn bind_hidden(s: &Session<'_>, layers: usize, width_bytes: usize) -> Result<Vec<u64>> {
    let intervals: Vec<_> = (0..layers + 2)
        .map(|i| Interval {
            bytes: width_bytes,
            first: i,
            last: i + 1,
        })
        .collect();
    if std::env::var("INFERFABRIC_DEDICATED_ACTIVATIONS").as_deref() == Ok("1") {
        return intervals.iter().map(|v| s.allocate(v.bytes)).collect();
    }
    let plan = plan(&intervals)?;
    let parent = s.allocate(plan.bytes)?;
    type View = unsafe extern "C" fn(SessionPtr, u64, u64, u64, *mut u64) -> i32;
    let view: View = unsafe {
        *s.api
            ._library
            .get(b"inferfabric_acl_buffer_view\0")
            .map_err(|e| invalid(e.to_string()))?
    };
    plan.offsets
        .iter()
        .map(|&offset| {
            let mut handle = 0;
            s.api.check(unsafe {
                view(
                    s.ptr,
                    parent,
                    offset as u64,
                    width_bytes as u64,
                    &mut handle,
                )
            })?;
            Ok(handle)
        })
        .collect()
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn adjacent_layers_and_residuals_do_not_alias() {
        let intervals: Vec<_> = (0..26)
            .map(|i| Interval {
                bytes: 16384,
                first: i,
                last: i + 1,
            })
            .collect();
        let p = plan(&intervals).unwrap();
        assert_eq!(p.bytes, 32768);
        for pair in p.offsets.windows(2) {
            assert_ne!(pair[0], pair[1]);
        }
    }
    #[test]
    fn varied_intervals_never_overlap_live_storage() {
        let values: Vec<_> = (0..80)
            .map(|i| Interval {
                bytes: 1 + (i * 997) % 10000,
                first: (i * 7) % 29,
                last: (i * 7) % 29 + (i * 11) % 9,
            })
            .collect();
        let p = plan(&values).unwrap();
        for (i, a) in values.iter().enumerate() {
            assert_eq!(p.offsets[i] % ALIGNMENT, 0);
            assert!(p.offsets[i] + a.bytes <= p.bytes);
            for (j, b) in values.iter().enumerate().take(i) {
                if a.first <= b.last && b.first <= a.last {
                    assert!(
                        p.offsets[i] + a.bytes <= p.offsets[j]
                            || p.offsets[j] + b.bytes <= p.offsets[i]
                    );
                }
            }
        }
    }
    #[test]
    fn long_lived_snapshot_and_alignment() {
        let values = [
            Interval {
                bytes: 513,
                first: 0,
                last: 9,
            },
            Interval {
                bytes: 1,
                first: 1,
                last: 2,
            },
            Interval {
                bytes: 512,
                first: 3,
                last: 8,
            },
        ];
        let p = plan(&values).unwrap();
        assert_eq!(p.offsets, [0, 1024, 1024]);
        assert_eq!(p.bytes, 1536);
        assert!(
            plan(&[Interval {
                bytes: usize::MAX,
                first: 0,
                last: 0
            }])
            .is_err()
        );
    }
}
