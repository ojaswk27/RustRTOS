"""
Exports trained PPO actor weights to JSON (for inspection) and generates
src/policy.rs (for embedding in Rust) in one step.

Network architecture: 24 -> 32 (ReLU) -> 32 (ReLU) -> 7.
Q10 format: weights multiplied by 1024 and rounded to i32.

Usage:  uv run python export_weights.py
Output: policy_weights.json  (human-readable debug dump)
        src/policy.rs        (generated Rust source — do not edit by hand)
"""

import json
import numpy as np
from stable_baselines3 import PPO

MODEL_PATH = "ppo_rtos_model/ppo_rtos.zip"
OUTPUT_JSON = "policy_weights.json"
OUTPUT_RUST = "src/policy.rs"
Q10_SCALE = 1024


def extract_actor_weights(model):
    """Pull weight matrices and bias vectors from the PPO actor network."""
    policy = model.policy
    layers = []
    for module in policy.mlp_extractor.policy_net:
        if hasattr(module, "weight"):
            w = module.weight.detach().cpu().numpy()
            b = module.bias.detach().cpu().numpy()
            layers.append((w, b))
    w = policy.action_net.weight.detach().cpu().numpy()
    b = policy.action_net.bias.detach().cpu().numpy()
    layers.append((w, b))
    return layers


def to_q10(arr: np.ndarray) -> np.ndarray:
    return (arr * Q10_SCALE).round().astype(int)


def fmt_2d(arr: np.ndarray) -> str:
    rows = ",\n    ".join(
        "[" + ", ".join(str(x) for x in row) + "]"
        for row in arr
    )
    return "[\n    " + rows + "\n]"


def fmt_1d(arr: np.ndarray) -> str:
    return "[" + ", ".join(str(x) for x in arr) + "]"


RUST_TEMPLATE = """\
// GENERATED — do not edit by hand. Run: uv run python export_weights.py

const SCALE: i32 = 1024;
const IN: usize = {in_size};
const H: usize = {h_size};
const OUT: usize = {out_size};

static W1: [[i32; {in_size}]; {h_size}] = {w1};
static B1: [i32; {h_size}] = {b1};

static W2: [[i32; {h_size}]; {h_size}] = {w2};
static B2: [i32; {h_size}] = {b2};

static W3: [[i32; {h_size}]; {out_size}] = {w3};
static B3: [i32; {out_size}] = {b3};

#[inline]
fn relu(x: i32) -> i32 {{
    if x > 0 {{ x }} else {{ 0 }}
}}

/// Run the policy network on a Q10-encoded state vector.
/// Returns action index 0-{n_tasks_max}: 0-{n_tasks} = run task, {out_size_minus1} = idle.
pub fn infer(state: &[i32; {in_size}]) -> usize {{
    let mut h1 = [0i32; {h_size}];
    for j in 0..{h_size} {{
        let mut acc: i32 = 0;
        for i in 0..{in_size} {{
            acc = acc.saturating_add(W1[j][i].saturating_mul(state[i]));
        }}
        h1[j] = relu(acc / SCALE + B1[j]);
    }}

    let mut h2 = [0i32; {h_size}];
    for j in 0..{h_size} {{
        let mut acc: i32 = 0;
        for i in 0..{h_size} {{
            acc = acc.saturating_add(W2[j][i].saturating_mul(h1[i]));
        }}
        h2[j] = relu(acc / SCALE + B2[j]);
    }}

    let mut best_idx: usize = 0;
    let mut best_val: i32 = i32::MIN;
    for j in 0..{out_size} {{
        let mut acc: i32 = 0;
        for i in 0..{h_size} {{
            acc = acc.saturating_add(W3[j][i].saturating_mul(h2[i]));
        }}
        let val = acc / SCALE + B3[j];
        if val > best_val {{
            best_val = val;
            best_idx = j;
        }}
    }}

    best_idx
}}
"""


def generate_policy_rs(layers: list, path: str) -> None:
    assert len(layers) == 3, f"Expected 3 layers (two hidden + output), got {len(layers)}"
    (w1, b1), (w2, b2), (w3, b3) = layers

    in_size = w1.shape[1]   # 24
    h_size = w1.shape[0]    # 32
    out_size = w3.shape[0]  # 7

    assert w2.shape == (h_size, h_size), f"Layer 2 shape mismatch: {w2.shape}"
    assert w3.shape[1] == h_size, f"Layer 3 input size mismatch: {w3.shape}"

    content = RUST_TEMPLATE.format(
        in_size=in_size,
        h_size=h_size,
        out_size=out_size,
        out_size_minus1=out_size - 1,
        n_tasks_max=out_size - 2,
        n_tasks=out_size - 2,
        w1=fmt_2d(to_q10(w1)),
        b1=fmt_1d(to_q10(b1)),
        w2=fmt_2d(to_q10(w2)),
        b2=fmt_1d(to_q10(b2)),
        w3=fmt_2d(to_q10(w3)),
        b3=fmt_1d(to_q10(b3)),
    )

    with open(path, "w") as f:
        f.write(content)
    print(f"Written: {path}  ({in_size}→{h_size}→{h_size}→{out_size})")


def main():
    print(f"Loading model from {MODEL_PATH}...")
    model = PPO.load(MODEL_PATH)
    layers = extract_actor_weights(model)

    print("Network architecture:")
    for i, (w, b) in enumerate(layers):
        print(f"  Layer {i}: weight {w.shape}, bias {b.shape}")

    # JSON dump (for debugging / inspection)
    export = {"q10_scale": Q10_SCALE, "layers": []}
    for w, b in layers:
        export["layers"].append({
            "weight_shape": list(w.shape),
            "weights": w.tolist(),
            "biases": b.tolist(),
            "weights_q10": to_q10(w).tolist(),
            "biases_q10": to_q10(b).tolist(),
        })
    with open(OUTPUT_JSON, "w") as f:
        json.dump(export, f, indent=2)
    print(f"Written: {OUTPUT_JSON}")

    # Generate src/policy.rs
    generate_policy_rs(layers, OUTPUT_RUST)
    print("\nNext step: cargo build --release")


if __name__ == "__main__":
    main()
