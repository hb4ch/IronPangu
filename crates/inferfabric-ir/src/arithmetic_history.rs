//! Exact, completed invocation history for continuation compatibility.
//! This is host metadata, not a certificate that device snapshots exist.
use crate::serving::Variant;
use inferfabric_model::{Result, invalid};
use serde::Serialize;

#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct Span {
    start: u32,
    end: u32,
    query_tokens: u32,
    numerical_class: String,
}

/// Constructed only by checked append operations. Adjacent equal invocation
/// shapes/classes are coalesced; a split must fall on an invocation boundary.
#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct ArithmeticHistory {
    version: u32,
    spans: Vec<Span>,
}
impl Default for ArithmeticHistory {
    fn default() -> Self {
        Self {
            version: 1,
            spans: Vec::new(),
        }
    }
}
impl ArithmeticHistory {
    pub fn tokens(&self) -> u32 {
        self.spans.last().map_or(0, |s| s.end)
    }
    pub fn spans(&self) -> &[Span] {
        &self.spans
    }
    pub fn check_append(&self, variant: &Variant, position: u32, count: u32) -> Result<()> {
        if position != self.tokens()
            || count == 0
            || variant.numerical_class.is_empty()
            || variant.max_sequences != 1
            || count != variant.max_tokens
            || count != variant.max_query_tokens
            || position
                .checked_add(count)
                .is_none_or(|end| end > variant.max_context)
        {
            return Err(invalid(
                "noncontiguous or unsupported arithmetic history invocation",
            ));
        }
        Ok(())
    }
    /// Call only after the owning streams have completed successfully on all ranks.
    pub fn append_completed(&mut self, variant: &Variant, position: u32, count: u32) -> Result<()> {
        self.check_append(variant, position, count)?;
        let end = position + count;
        if let Some(last) = self.spans.last_mut()
            && last.query_tokens == count
            && last.numerical_class == variant.numerical_class
        {
            last.end = end;
        } else {
            self.spans.push(Span {
                start: position,
                end,
                query_tokens: count,
                numerical_class: variant.numerical_class.clone(),
            });
        }
        Ok(())
    }
    /// No snapshot or arithmetic boundary is invented inside a packed invocation.
    pub fn prefix(&self, boundary: u32) -> Result<Self> {
        if boundary > self.tokens() {
            return Err(invalid("arithmetic prefix exceeds completed history"));
        }
        let mut result = Self::default();
        for span in &self.spans {
            if span.start >= boundary {
                break;
            }
            let mut part = span.clone();
            part.end = part.end.min(boundary);
            if (part.end - part.start) % part.query_tokens != 0 {
                return Err(invalid("arithmetic prefix splits a packed invocation"));
            }
            result.spans.push(part);
        }
        Ok(result)
    }
    /// Namespace, token equality, page alignment and complete device-state
    /// ownership must also pass; this only compares the arithmetic intervals.
    pub fn same_prefix(&self, other: &Self, boundary: u32) -> bool {
        match (self.prefix(boundary), other.prefix(boundary)) {
            (Ok(a), Ok(b)) => a == b,
            _ => false,
        }
    }
}
