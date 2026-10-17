//! Conservative legality analysis for compact request-output rows.
//! This analysis does not rewrite or enable execution by itself. Unknown or
//! stateful operations remain in the full-token prefix, including on zero-row calls.
use crate::token_program::Instruction;
use inferfabric_model::{Result, invalid};
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct OutputRows {
    /// Full-token value selected before entering the compact suffix.
    pub source: usize,
    pub source_width: usize,
    /// Inclusive instruction range. All earlier instructions still execute.
    pub first: usize,
    pub last: usize,
    pub row_domain: String,
    pub selection_kernel: String,
    pub empty_rows: String,
}

// A closed allowlist: these backend operations preserve independent token rows.
// In particular allreduce, attention, routing, recurrent/state operations and
// unknown kernels cannot move across this boundary.
fn independent(node: &Instruction) -> bool {
    node.kind == 0
        && node.inputs.len() == 1
        && matches!(
            node.op.as_str(),
            "norm" | "linear" | "gather" | "cast" | "copy"
        )
}

fn suffix_step(nodes: &[Instruction], uses: &[usize], end: usize) -> Option<usize> {
    let n = &nodes[end];
    if independent(n) && nodes[n.inputs[0]].kind == 0 {
        return Some(end);
    }
    // Rank-major storage is not itself token-row storage. Only move this
    // producer/layout-conversion pair together; never select rows between them.
    if end > 0 && n.op == "head_to_token" && n.inputs == [end - 1] && n.kind == 0 {
        let gather = &nodes[end - 1];
        if gather.op == "gather_heads"
            && gather.kind == 0
            && gather.inputs.len() == 1
            && uses[end - 1] == 1
            && gather.width == n.width
            && gather.attrs.get("heads").is_some()
            && gather.attrs.get("heads") == n.attrs.get("heads")
        {
            return Some(end - 1);
        }
    }
    None
}

pub fn plan(nodes: &[Instruction], output: usize) -> Result<Option<OutputRows>> {
    if output >= nodes.len()
        || nodes
            .iter()
            .enumerate()
            .any(|(id, n)| n.inputs.iter().any(|&i| i >= id))
    {
        return Err(invalid(
            "output row analysis requires a topological program",
        ));
    }
    // Never skip an effect after the output. General DAG partitioning is a
    // separate extension; this implementation proves a contiguous unary suffix.
    if output + 1 != nodes.len() {
        return Ok(None);
    }
    let mut uses = vec![0usize; nodes.len()];
    for node in nodes {
        for &input in &node.inputs {
            uses[input] += 1;
        }
    }
    let Some(mut first) = suffix_step(nodes, &uses, output) else {
        return Ok(None);
    };
    while first > 0 {
        let input = nodes[first].inputs[0];
        if input + 1 != first || uses[input] != 1 {
            break;
        }
        let Some(start) = suffix_step(nodes, &uses, input) else {
            break;
        };
        first = start;
    }
    let source = nodes[first].inputs[0];
    if nodes[source].kind != 0 || nodes[source].width == 0 {
        return Ok(None);
    }
    Ok(Some(OutputRows {
        source,
        source_width: nodes[source].width,
        first,
        last: output,
        row_domain: "request_sample_rows".into(),
        selection_kernel: "ordered_device_row_copy".into(),
        empty_rows: "execute_full_token_prefix_skip_output_suffix".into(),
    }))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeMap;
    fn node(op: &str, inputs: Vec<usize>) -> Instruction {
        Instruction {
            logical_node: 0,
            name: op.into(),
            op: op.into(),
            inputs,
            attrs: BTreeMap::new(),
            kind: 0,
            width: 16,
            dtype: "bf16".into(),
            kernel: op.into(),
            reason: String::new(),
        }
    }
    #[test]
    fn only_proven_output_suffix_changes_row_domain() {
        let nodes = vec![
            node("cache_append", vec![]),
            node("norm", vec![0]),
            node("linear", vec![1]),
            node("gather", vec![2]),
        ];
        let selected = plan(&nodes, 3).unwrap().unwrap();
        assert_eq!((selected.source, selected.first, selected.last), (0, 1, 3));
        assert_eq!(
            selected.empty_rows,
            "execute_full_token_prefix_skip_output_suffix"
        );
        let mut unknown = nodes.clone();
        unknown[2].op = "custom_unknown".into();
        assert_eq!(plan(&unknown, 3).unwrap().unwrap().first, 3);
        unknown[3].op = "state_write".into();
        assert!(plan(&unknown, 3).unwrap().is_none());
    }
    #[test]
    fn effects_fanout_and_invalid_dependencies_are_not_removed() {
        let nodes = vec![
            node("embedding", vec![]),
            node("norm", vec![0]),
            node("state_write", vec![1]),
            node("linear", vec![1]),
            node("gather", vec![3]),
        ];
        let selected = plan(&nodes, 4).unwrap().unwrap();
        assert_eq!((selected.source, selected.first), (1, 3));
        assert!(plan(&nodes, 1).unwrap().is_none());
        let mut broken = nodes;
        broken[3].inputs = vec![4];
        assert!(plan(&broken, 4).is_err());
    }
    #[test]
    fn rank_major_layout_is_never_selected_as_token_rows() {
        let mut nodes = vec![
            node("embedding", vec![]),
            node("gather_heads", vec![0]),
            node("head_to_token", vec![1]),
        ];
        assert!(plan(&nodes, 2).unwrap().is_none());
        nodes[1].attrs.insert("heads".into(), "8".into());
        nodes[2].attrs.insert("heads".into(), "8".into());
        assert_eq!(plan(&nodes, 2).unwrap().unwrap().first, 1);
        nodes[2].attrs.insert("heads".into(), "4".into());
        assert!(plan(&nodes, 2).unwrap().is_none());
    }
}
