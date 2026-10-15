//! OpenPangu adapter over the shared authoring frontend (no private parser).
use inferfabric_model::{OpenPanguParallel, Result, invalid};
pub fn parse(source: &str) -> Result<OpenPanguParallel> {
    match crate::definition(source)? {
        crate::authoring::ModelDefinition::OpenPangu { parallel }
        | crate::authoring::ModelDefinition::Execution { parallel, .. } => Ok(parallel),
        _ => Err(invalid("expected OpenPangu model definition")),
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    const SOURCE: &str = include_str!("../../../examples/openpangu/92b-tp8-ep8.inferfabric");
    #[test]
    fn topology_and_budget_contract() {
        assert_eq!(parse(SOURCE).unwrap().devices.len(), 8);
        for s in [
            SOURCE.replace("ep = 8", "ep = 4"),
            SOURCE.replace("6, 7", "6, 6"),
            SOURCE.replace("seqs = 1", "seqs = 0"),
            SOURCE.replace("tp = 8;", "tp = 8; tp = 8;"),
            SOURCE.replace("tp = 8;", "tp = 8; pp = 2;"),
        ] {
            assert!(parse(&s).is_err());
        }
        assert!(parse("contract = openpangu_v2_tp_ep_v1\ntp = 8").is_err());
    }
}
