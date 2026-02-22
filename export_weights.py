"""
Exports trained PPO actor network weights to JSON for embedding in Rust.

The actor MLP has architecture 24 -> 32 (ReLU) -> 32 (ReLU) -> 7.
Weights are stored as nested lists of floats, and also as Q10 fixed-point
integers (scaled by 1024) ready for direct paste into policy.rs.

Usage: python export_weights.py
Output: policy_weights.json
"""

import json
import numpy as np
from stable_baselines3 import PPO

MODEL_PATH = "ppo_rtos_model/ppo_rtos.zip"
OUTPUT = "policy_weights.json"
Q10_SCALE = 1024


def extract_actor_weights(model):
    """Pull weight matrices and bias vectors from the PPO actor network."""
    policy = model.policy
    # SB3 MlpPolicy stores the policy net in policy.mlp_extractor.policy_net
    # and the final action layer in policy.action_net
    layers = []

    # Hidden layers from the MLP extractor
    for module in policy.mlp_extractor.policy_net:
        if hasattr(module, "weight"):
            w = module.weight.detach().cpu().numpy()  # shape (out, in)
            b = module.bias.detach().cpu().numpy()  # shape (out,)
            layers.append((w, b))

    # Final action head
    w = policy.action_net.weight.detach().cpu().numpy()
    b = policy.action_net.bias.detach().cpu().numpy()
    layers.append((w, b))

    return layers


def main():
    print(f"Loading model from {MODEL_PATH}...")
    model = PPO.load(MODEL_PATH)
    layers = extract_actor_weights(model)

    print("Network architecture:")
    for i, (w, b) in enumerate(layers):
        print(f"  Layer {i}: weight {w.shape}, bias {b.shape}")

    # Build JSON with both float and Q10 integer representations
    export = {"layers": [], "q10_scale": Q10_SCALE}
    for w, b in layers:
        export["layers"].append(
            {
                "weight_shape": list(w.shape),
                "weights": w.tolist(),
                "biases": b.tolist(),
                "weights_q10": (w * Q10_SCALE).round().astype(int).tolist(),
                "biases_q10": (b * Q10_SCALE).round().astype(int).tolist(),
            }
        )

    with open(OUTPUT, "w") as f:
        json.dump(export, f, indent=2)

    print(f"\nExported to {OUTPUT}")
    print(f"To use in Rust, copy the weights_q10 and biases_q10 arrays into policy.rs")


if __name__ == "__main__":
    main()
