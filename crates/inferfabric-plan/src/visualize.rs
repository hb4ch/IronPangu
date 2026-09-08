use crate::{ir::PhysicalPlan, json, verify};
use inferfabric_model::{Result, invalid};
pub fn html(plan: &PhysicalPlan) -> Result<String> {
    verify(plan)?;
    // JSON is embedded as inert script text; escape HTML parser terminators.
    let encoded = String::from_utf8(json(plan)?)
        .map_err(|e| invalid(e.to_string()))?
        .replace('&', "\\u0026")
        .replace('<', "\\u003c")
        .replace('>', "\\u003e")
        .replace('\u{2028}', "\\u2028")
        .replace('\u{2029}', "\\u2029");
    Ok(include_str!("visualize.html").replace("__PLAN__", &encoded))
}
