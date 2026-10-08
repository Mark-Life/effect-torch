//! JSON snapshot of a lazy semantic graph for inspection tools.
//!
//! The walk never executes or validates the graph, so partial graphs with
//! unbound `Input` placeholders are accepted.

use crate::operation_name;
use effect_torch_graph::{
    node_children, AttentionRounding, AttentionWindow, CrossEntropyReduction, KvAttentionMode,
    Node, NodeKind, PositionOffset, RotaryLayout,
};
use serde_json::{json, Map, Value};
use std::collections::HashSet;
use std::sync::Arc;

/// Serializes every node reachable from `roots` as JSON:
/// `{ "nodes": [...], "roots": [...] }`. Nodes are deduplicated and listed in
/// postorder, so every input precedes its consumers. Roots keep caller order
/// and duplicates.
pub fn graph_snapshot_json(roots: &[Arc<Node>]) -> Result<String, String> {
    if roots.is_empty() {
        return Err("graphSnapshot requires at least one root".to_string());
    }
    let nodes: Vec<Value> = postorder(roots).iter().map(|n| node_json(n)).collect();
    let root_ids: Vec<u64> = roots.iter().map(|r| r.id).collect();
    serde_json::to_string(&json!({ "nodes": nodes, "roots": root_ids }))
        .map_err(|e| format!("graphSnapshot: {e}"))
}

// Iterative so deep chains cannot overflow the call stack.
fn postorder(roots: &[Arc<Node>]) -> Vec<Arc<Node>> {
    let mut emitted: HashSet<u64> = HashSet::new();
    let mut order = Vec::new();
    let mut stack: Vec<(Arc<Node>, bool)> =
        roots.iter().rev().map(|r| (r.clone(), false)).collect();
    while let Some((node, expanded)) = stack.pop() {
        if emitted.contains(&node.id) {
            continue;
        }
        if expanded {
            emitted.insert(node.id);
            order.push(node);
            continue;
        }
        let children = node_children(&node.kind);
        stack.push((node, true));
        stack.extend(
            children
                .into_iter()
                .rev()
                .filter(|c| !emitted.contains(&c.id))
                .map(|c| (c, false)),
        );
    }
    order
}

fn node_json(node: &Node) -> Value {
    let inputs: Vec<u64> = node_children(&node.kind).iter().map(|c| c.id).collect();
    let mut out = Map::new();
    out.insert("id".into(), json!(node.id));
    out.insert("op".into(), json!(operation_name(&node.kind)));
    out.insert("shape".into(), json!(node.shape));
    out.insert("dtype".into(), json!(node.dtype.name()));
    out.insert("inputs".into(), json!(inputs));
    out.insert(
        "attributes".into(),
        Value::Object(node_attributes(&node.kind)),
    );
    if let NodeKind::Expose { name, .. } = &node.kind {
        out.insert("exposeName".into(), json!(name));
    }
    Value::Object(out)
}

fn num(v: f64) -> Value {
    if v.is_nan() {
        json!("nan")
    } else if v.is_infinite() {
        json!(if v > 0.0 { "inf" } else { "-inf" })
    } else {
        json!(v)
    }
}

fn attrs<const N: usize>(entries: [(&str, Value); N]) -> Map<String, Value> {
    entries
        .into_iter()
        .map(|(k, v)| (k.to_string(), v))
        .collect()
}

fn window(w: AttentionWindow) -> Value {
    match w {
        AttentionWindow::Inherit => json!("inherit"),
        AttentionWindow::Full => json!("full"),
        AttentionWindow::Local(n) => json!(n),
    }
}

fn rounding(r: AttentionRounding) -> Value {
    json!(match r {
        AttentionRounding::Fused => "fused",
        AttentionRounding::Stepwise => "stepwise",
    })
}

fn reduction(r: CrossEntropyReduction) -> Value {
    json!(match r {
        CrossEntropyReduction::Mean => "mean",
        CrossEntropyReduction::Sum => "sum",
    })
}

fn layout(l: RotaryLayout) -> Value {
    json!(match l {
        RotaryLayout::HalfSplit => "halfSplit",
        RotaryLayout::InterleavedPairs => "interleavedPairs",
    })
}

fn offset(o: PositionOffset) -> Value {
    json!(match o {
        PositionOffset::Absolute => "absolute",
        PositionOffset::Cursor => "cursor",
    })
}

fn kv_mode(m: KvAttentionMode) -> Value {
    json!(match m {
        KvAttentionMode::Causal => "causal",
        KvAttentionMode::BidirectionalBlock => "bidirectionalBlock",
    })
}

// Exhaustive on purpose: a new NodeKind variant must decide its attributes.
fn node_attributes(kind: &NodeKind) -> Map<String, Value> {
    use NodeKind as K;
    match kind {
        K::Input { slot, .. } | K::ScalarInput { slot, .. } => attrs([("slot", json!(slot))]),
        K::Full { value, .. } => attrs([("value", num(*value))]),
        K::Arange {
            start, end, step, ..
        } => attrs([
            ("start", num(*start)),
            ("end", num(*end)),
            ("step", num(*step)),
        ]),
        K::Uniform { lo, hi, .. } => attrs([("lo", num(*lo)), ("hi", num(*hi))]),
        K::Eye { n, .. } => attrs([("n", json!(n))]),
        K::FromBytes { data, .. } => attrs([("bytes", json!(data.len()))]),
        K::Cast { dtype, .. } => attrs([("dtype", json!(dtype.name()))]),
        K::Pow { exp, .. } => attrs([("exp", num(*exp))]),
        K::Gelu { approximate, .. } => attrs([("approximate", json!(approximate))]),
        K::Sum { dims, keepdims, .. }
        | K::Mean { dims, keepdims, .. }
        | K::Max { dims, keepdims, .. }
        | K::Min { dims, keepdims, .. }
        | K::Prod { dims, keepdims, .. } => {
            attrs([("dims", json!(dims)), ("keepdims", json!(keepdims))])
        }
        K::Argmax { dim, .. }
        | K::Argmin { dim, .. }
        | K::Cumsum { dim, .. }
        | K::IndexSelect { dim, .. }
        | K::ScatterAdd { dim, .. }
        | K::Gather { dim, .. }
        | K::Concat { dim, .. } => attrs([("dim", json!(dim))]),
        K::TopKIndices { k, .. } => attrs([("k", json!(k))]),
        K::Reshape { shape, .. } | K::BroadcastTo { shape, .. } => attrs([("shape", json!(shape))]),
        K::Permute { dims, .. } => attrs([("dims", json!(dims))]),
        K::Slice { ranges, .. } => {
            let ranges: Vec<[usize; 3]> = ranges.iter().map(|&(s, e, st)| [s, e, st]).collect();
            attrs([("ranges", json!(ranges))])
        }
        K::CrossEntropy {
            ignore_index,
            reduction: r,
            ..
        }
        | K::CrossEntropyBackward {
            ignore_index,
            reduction: r,
            ..
        } => attrs([
            ("ignoreIndex", json!(ignore_index)),
            ("reduction", reduction(*r)),
        ]),
        K::Sdpa {
            scale,
            causal,
            window: w,
            ..
        }
        | K::SdpaBackward {
            scale,
            causal,
            window: w,
            ..
        } => attrs([
            ("scale", num(*scale)),
            ("causal", json!(causal)),
            ("window", window(*w)),
        ]),
        K::SdpaConfigured {
            scale,
            causal,
            window: w,
            rounding: r,
            layer_id,
            retention,
            ..
        } => attrs([
            ("scale", num(*scale)),
            ("causal", json!(causal)),
            ("window", window(*w)),
            ("rounding", rounding(*r)),
            ("layerId", json!(layer_id)),
            ("retention", window(*retention)),
        ]),
        K::KvAttention {
            scale,
            layer,
            window: w,
            mode,
            rounding: r,
            ..
        } => attrs([
            ("scale", num(*scale)),
            ("layer", json!(layer)),
            ("window", json!(w)),
            ("mode", kv_mode(*mode)),
            ("rounding", rounding(*r)),
        ]),
        K::SdpaBackwardOut { index, .. }
        | K::KdaBackwardOut { index, .. }
        | K::ChunkedHeadCeBackwardOut { index, .. }
        | K::LayerNormBackwardOut { index, .. }
        | K::AdamWOut { index, .. }
        | K::SgdOut { index, .. } => attrs([("index", json!(index))]),
        K::PositionEmbedding { seq_len, .. } => attrs([("seqLen", json!(seq_len))]),
        K::KdaChunk { scale, .. } | K::KdaBackward { scale, .. } => attrs([("scale", num(*scale))]),
        K::KdaRecurrence { scale, layer, .. } => {
            attrs([("scale", num(*scale)), ("layer", json!(layer))])
        }
        K::ConvState { layer, .. } => attrs([("layer", json!(layer))]),
        K::ChunkedHeadCe { ignore_index, .. } | K::ChunkedHeadCeBackward { ignore_index, .. } => {
            attrs([("ignoreIndex", json!(ignore_index))])
        }
        K::RotaryEmbedding {
            seq_len,
            theta,
            offset: o,
            layout: l,
            ..
        } => attrs([
            ("seqLen", json!(seq_len)),
            ("theta", num(*theta)),
            ("offset", offset(*o)),
            ("layout", layout(*l)),
        ]),
        K::RotaryEmbeddingExplicit { layout: l, .. } => attrs([("layout", layout(*l))]),
        K::RotaryEmbeddingBackward {
            shape,
            seq_len,
            theta,
            layout: l,
            ..
        } => attrs([
            ("shape", json!(shape)),
            ("seqLen", json!(seq_len)),
            ("theta", num(*theta)),
            ("layout", layout(*l)),
        ]),
        K::LayerNorm { eps, .. } | K::RmsNorm { eps, .. } | K::LayerNormBackward { eps, .. } => {
            attrs([("eps", num(*eps))])
        }
        K::QuantizedEmbedding { padding_index, .. } => {
            attrs([("paddingIndex", json!(padding_index))])
        }
        K::Conv1d {
            stride,
            padding,
            dilation,
            groups,
            ..
        }
        | K::Conv2d {
            stride,
            padding,
            dilation,
            groups,
            ..
        } => attrs([
            ("stride", json!(stride)),
            ("padding", json!(padding)),
            ("dilation", json!(dilation)),
            ("groups", json!(groups)),
        ]),
        K::ConvTranspose1d {
            stride,
            padding,
            output_padding,
            dilation,
            groups,
            ..
        }
        | K::ConvTranspose2d {
            stride,
            padding,
            output_padding,
            dilation,
            groups,
            ..
        } => attrs([
            ("stride", json!(stride)),
            ("padding", json!(padding)),
            ("outputPadding", json!(output_padding)),
            ("dilation", json!(dilation)),
            ("groups", json!(groups)),
        ]),
        K::Conv1dBackwardW {
            kernel,
            out_channels,
            stride,
            padding,
            dilation,
            groups,
            ..
        } => attrs([
            ("kernel", json!(kernel)),
            ("outChannels", json!(out_channels)),
            ("stride", json!(stride)),
            ("padding", json!(padding)),
            ("dilation", json!(dilation)),
            ("groups", json!(groups)),
        ]),
        K::Conv2dBackwardW {
            kernel,
            out_channels,
            stride,
            padding,
            dilation,
            groups,
            ..
        } => attrs([
            ("kernel", json!(kernel)),
            ("outChannels", json!(out_channels)),
            ("stride", json!(stride)),
            ("padding", json!(padding)),
            ("dilation", json!(dilation)),
            ("groups", json!(groups)),
        ]),
        K::AdamWStep {
            beta1,
            beta2,
            eps,
            weight_decay,
            ..
        } => attrs([
            ("beta1", num(*beta1)),
            ("beta2", num(*beta2)),
            ("eps", num(*eps)),
            ("weightDecay", num(*weight_decay)),
        ]),
        K::SgdStep {
            momentum,
            dampening,
            nesterov,
            weight_decay,
            ..
        } => attrs([
            ("momentum", num(*momentum)),
            ("dampening", num(*dampening)),
            ("nesterov", json!(nesterov)),
            ("weightDecay", num(*weight_decay)),
        ]),
        K::Expose { name, .. } => attrs([("name", json!(name))]),
        K::Leaf(_)
        | K::Zeros { .. }
        | K::Ones { .. }
        | K::Randn { .. }
        | K::Add { .. }
        | K::Sub { .. }
        | K::Mul { .. }
        | K::Div { .. }
        | K::Eq { .. }
        | K::Gt { .. }
        | K::Lt { .. }
        | K::Ge { .. }
        | K::Le { .. }
        | K::Maximum { .. }
        | K::Minimum { .. }
        | K::Neg { .. }
        | K::Abs { .. }
        | K::Sqrt { .. }
        | K::Exp { .. }
        | K::Log { .. }
        | K::Sin { .. }
        | K::Cos { .. }
        | K::Tanh { .. }
        | K::Relu { .. }
        | K::Erf { .. }
        | K::Floor { .. }
        | K::Ceil { .. }
        | K::Round { .. }
        | K::Sign { .. }
        | K::Where { .. }
        | K::ShortConv1d { .. }
        | K::LastTokenRow { .. }
        | K::ShortConv1dBackwardX { .. }
        | K::ShortConv1dBackwardW { .. }
        | K::Linear { .. }
        | K::ExpertLinearRows { .. }
        | K::GroupedExpertLinearRows { .. }
        | K::QuantizedLinear { .. }
        | K::Matmul { .. }
        | K::Inverse { .. }
        | K::Det { .. }
        | K::Solve { .. }
        | K::StopGradient { .. }
        | K::Checkpoint { .. } => Map::new(),
    }
}

#[cfg(test)]
#[path = "snapshot_tests.rs"]
mod tests;
