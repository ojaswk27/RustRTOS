/// Fixed-point neural network inference for the scheduling policy.
///
/// Uses Q10 format: multiply float by 1024 and round to i32.
/// This avoids floating-point entirely, which is important on embedded
/// targets where FPU use can introduce non-deterministic timing and
/// where we want the scheduler decision to be fast and predictable.
///
/// Network architecture: 24 inputs -> 32 (ReLU) -> 32 (ReLU) -> 7 outputs.
/// The argmax of the output selects which task to run (0-5) or idle (6).

const SCALE: i32 = 1024;
const IN: usize = 24;
const H: usize = 32;
const OUT: usize = 7;

// ── Placeholder weights ──────────────────────────────────────────────
// Replace these with values from policy_weights.json (weights_q10 fields)
// after training. Each weight matrix is stored as [output_neurons][input_neurons]
// for cache-friendly row iteration during inference.

static W1: [[i32; IN]; H] = [[0; IN]; H];
static B1: [i32; H] = [0; H];

static W2: [[i32; H]; H] = [[0; H]; H];
static B2: [i32; H] = [0; H];

static W3: [[i32; H]; OUT] = [[0; H]; OUT];
static B3: [i32; OUT] = [0; OUT];

#[inline]
fn relu(x: i32) -> i32 {
    if x > 0 {
        x
    } else {
        0
    }
}

/// Run the policy network on a Q10-encoded state vector.
/// Returns the action index (0..6).
///
/// Computation per layer: out[j] = ReLU( sum_i(W[j][i] * input[i]) / SCALE + B[j] )
/// The division by SCALE after multiply-accumulate keeps values in Q10 range.
/// Final layer has no ReLU — we just take the argmax.
pub fn infer(state: &[i32; IN]) -> usize {
    // Layer 1: IN -> H with ReLU
    let mut h1 = [0i32; H];
    for j in 0..H {
        let mut acc: i32 = 0;
        for i in 0..IN {
            acc = acc.saturating_add(W1[j][i].saturating_mul(state[i]));
        }
        h1[j] = relu(acc / SCALE + B1[j]);
    }

    // Layer 2: H -> H with ReLU
    let mut h2 = [0i32; H];
    for j in 0..H {
        let mut acc: i32 = 0;
        for i in 0..H {
            acc = acc.saturating_add(W2[j][i].saturating_mul(h1[i]));
        }
        h2[j] = relu(acc / SCALE + B2[j]);
    }

    // Output layer: H -> OUT (no activation, just argmax)
    let mut best_idx: usize = 0;
    let mut best_val: i32 = i32::MIN;
    for j in 0..OUT {
        let mut acc: i32 = 0;
        for i in 0..H {
            acc = acc.saturating_add(W3[j][i].saturating_mul(h2[i]));
        }
        let val = acc / SCALE + B3[j];
        if val > best_val {
            best_val = val;
            best_idx = j;
        }
    }

    best_idx
}
