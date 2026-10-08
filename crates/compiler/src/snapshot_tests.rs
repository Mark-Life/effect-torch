use super::*;
use effect_torch_graph::Device;
use effect_torch_runtime::DType;

fn full(shape: &[usize], value: f64) -> Arc<Node> {
    Node::new(NodeKind::Full {
        shape: shape.to_vec(),
        value,
        dtype: DType::F32,
        device: Device::Cpu(0),
    })
    .unwrap()
}

fn snapshot(roots: &[Arc<Node>]) -> Value {
    serde_json::from_str(&graph_snapshot_json(roots).unwrap()).unwrap()
}

fn by_id(snap: &Value, id: u64) -> Value {
    snap["nodes"]
        .as_array()
        .unwrap()
        .iter()
        .find(|n| n["id"] == json!(id))
        .cloned()
        .unwrap()
}

#[test]
fn postorder_and_dedup() {
    let a = full(&[2, 3], 1.0);
    let b = Node::new(NodeKind::Add {
        a: a.clone(),
        b: a.clone(),
    })
    .unwrap();
    let m = Node::new(NodeKind::Mul {
        a: b.clone(),
        b: a.clone(),
    })
    .unwrap();
    let c = Node::new(NodeKind::Expose {
        a: m.clone(),
        name: "h".into(),
    })
    .unwrap();
    let snap = snapshot(&[c.clone(), c.clone()]);
    let nodes = snap["nodes"].as_array().unwrap();
    assert_eq!(nodes.len(), 4);
    let ids: Vec<u64> = nodes.iter().map(|n| n["id"].as_u64().unwrap()).collect();
    let unique: HashSet<u64> = ids.iter().copied().collect();
    assert_eq!(unique.len(), ids.len());
    for (i, n) in nodes.iter().enumerate() {
        for input in n["inputs"].as_array().unwrap() {
            let pos = ids.iter().position(|id| json!(id) == *input).unwrap();
            assert!(pos < i);
        }
    }
    assert_eq!(snap["roots"], json!([c.id, c.id]));
    assert_eq!(by_id(&snap, b.id)["inputs"], json!([a.id, a.id]));
    assert_eq!(by_id(&snap, c.id)["exposeName"], json!("h"));
    assert_eq!(by_id(&snap, c.id)["attributes"]["name"], json!("h"));
    let ops: Vec<&str> = nodes.iter().map(|n| n["op"].as_str().unwrap()).collect();
    assert_eq!(ops, ["full", "add", "mul", "expose"]);
    assert_eq!(by_id(&snap, a.id)["shape"], json!([2, 3]));
    assert_eq!(by_id(&snap, a.id)["dtype"], json!("f32"));
    assert!(by_id(&snap, a.id).get("exposeName").is_none());
}

#[test]
fn attributes_are_json_safe() {
    let neg_inf = full(&[4], f64::NEG_INFINITY);
    let reshaped = Node::new(NodeKind::Reshape {
        a: neg_inf.clone(),
        shape: vec![2, 2],
    })
    .unwrap();
    let summed = Node::new(NodeKind::Sum {
        a: reshaped.clone(),
        dims: vec![1],
        keepdims: true,
    })
    .unwrap();
    let snap = snapshot(std::slice::from_ref(&summed));
    assert_eq!(
        by_id(&snap, neg_inf.id)["attributes"]["value"],
        json!("-inf")
    );
    assert_eq!(
        by_id(&snap, reshaped.id)["attributes"]["shape"],
        json!([2, 2])
    );
    let sum = by_id(&snap, summed.id);
    assert_eq!(sum["attributes"]["dims"], json!([1]));
    assert_eq!(sum["attributes"]["keepdims"], json!(true));
    assert_eq!(num(f64::NAN), json!("nan"));
    assert_eq!(num(f64::INFINITY), json!("inf"));
}

fn full_dt(shape: &[usize], dtype: DType) -> Arc<Node> {
    Node::new(NodeKind::Full {
        shape: shape.to_vec(),
        value: 0.0,
        dtype,
        device: Device::Cpu(0),
    })
    .unwrap()
}

fn attributes_of(node: &Arc<Node>) -> Value {
    by_id(&snapshot(std::slice::from_ref(node)), node.id)["attributes"].clone()
}

#[test]
fn optimizer_attributes() {
    let p = full(&[2], 1.0);
    let scalar = full(&[], 1.0);
    let adam = Node::new(NodeKind::AdamWStep {
        param: p.clone(),
        grad: p.clone(),
        m: p.clone(),
        v: p.clone(),
        lr: scalar.clone(),
        c1: scalar.clone(),
        c2: scalar.clone(),
        beta1: 0.9,
        beta2: 0.98,
        eps: 1e-8,
        weight_decay: 1.0,
    })
    .unwrap();
    let m_out = Node::new(NodeKind::AdamWOut {
        step: adam.clone(),
        index: 1,
    })
    .unwrap();
    assert_eq!(attributes_of(&m_out), json!({ "index": 1 }));
    assert_eq!(attributes_of(&adam)["weightDecay"], json!(1.0));
    let sgd = Node::new(NodeKind::SgdStep {
        param: p.clone(),
        grad: p.clone(),
        velocity: p.clone(),
        first: scalar.clone(),
        lr: scalar,
        momentum: 0.5,
        dampening: 0.25,
        nesterov: true,
        weight_decay: 0.0,
    })
    .unwrap();
    assert_eq!(
        attributes_of(&sgd),
        json!({ "momentum": 0.5, "dampening": 0.25, "nesterov": true, "weightDecay": 0.0 })
    );
    let velocity = Node::new(NodeKind::SgdOut {
        step: sgd,
        index: 1,
    })
    .unwrap();
    assert_eq!(attributes_of(&velocity), json!({ "index": 1 }));
}

#[test]
fn indexing_and_loss_attributes() {
    let x = full(&[2, 4], 0.5);
    let top = Node::new(NodeKind::TopKIndices { a: x.clone(), k: 3 }).unwrap();
    assert_eq!(attributes_of(&top), json!({ "k": 3 }));
    let idx = full_dt(&[2, 4], DType::I64);
    let scatter = Node::new(NodeKind::ScatterAdd {
        a: x.clone(),
        dim: 1,
        indexes: idx,
        src: x.clone(),
    })
    .unwrap();
    assert_eq!(attributes_of(&scatter), json!({ "dim": 1 }));
    let target = full_dt(&[2], DType::I64);
    let ce = Node::new(NodeKind::CrossEntropyBackward {
        logits: x,
        target,
        ignore_index: -100,
        reduction: CrossEntropyReduction::Sum,
    })
    .unwrap();
    assert_eq!(
        attributes_of(&ce),
        json!({ "ignoreIndex": -100, "reduction": "sum" })
    );
}

#[test]
fn attention_attributes() {
    let q = full(&[1, 2, 4], 0.5);
    let sdpa = Node::new(NodeKind::Sdpa {
        q: q.clone(),
        k: q.clone(),
        v: q,
        scale: 0.5,
        causal: true,
        window: AttentionWindow::Local(8),
    })
    .unwrap();
    assert_eq!(
        attributes_of(&sdpa),
        json!({ "scale": 0.5, "causal": true, "window": 8 })
    );
    assert_eq!(window(AttentionWindow::Inherit), json!("inherit"));
    assert_eq!(window(AttentionWindow::Full), json!("full"));
}

#[test]
fn empty_roots_fail() {
    assert!(graph_snapshot_json(&[]).is_err());
}

fn deep_chain(length: usize) -> usize {
    let mut node = full(&[1], 0.0);
    for _ in 0..length {
        node = Node::new(NodeKind::Neg { a: node }).unwrap();
    }
    let snap = snapshot(&[node]);
    snap["nodes"].as_array().unwrap().len()
}

#[test]
fn deep_chain_no_stack_overflow() {
    assert_eq!(deep_chain(200_000), 200_001);
    let handle = std::thread::Builder::new()
        .stack_size(256 * 1024)
        .spawn(|| deep_chain(200_000))
        .unwrap();
    assert_eq!(handle.join().unwrap(), 200_001);
}
