// TF-Mamba block tests against the PyTorch fixtures (need REUSE_GGUF; skip otherwise).
// REUSE_DEVICE selects the backend device (default CPU).
#include "tap_util.h"

// Relative max error tolerances; see tests/TOLERANCES.md.
static const double TOL_BLOCK1 = 1e-5;
static const double TOL_BLOCK30 = 2e-5;

static void block1_matches_fixture() { check_tap("block1_out", "block1_out", TOL_BLOCK1); }

static void blocks_1_to_30_match_fixture() { check_tap("block30_out", "block30_out", TOL_BLOCK30); }

// The full model graph (encoder, all TF-Mamba blocks, both decoders) must stay well inside GraphRun's capacity
// (graph_max_nodes): a ggml context/graph overflow aborts the process. Graph leafs are bounded
// by the context's own tensors plus all model weights.
static void graph_fits_node_budget() {
    if (!std::getenv("REUSE_GGUF")) { std::printf("  SKIP: REUSE_GGUF not set\n"); return; }
    reuse_ctx * ctx = model();
    if (!ctx) { CHECK(false); return; }
    const Weights & w = ctx->weights;
    CHECK(w.hp.num_tfmamba == 30);
    GraphRun g(graph_max_nodes(w.hp.num_tfmamba));
    ggml_tensor * in = ggml_new_tensor_3d(g.ctx, GGML_TYPE_F32, 481, 201, 2);
    const DecoderOut d = forward(g.ctx, w, in);
    ggml_build_forward_expand(g.gf, d.mag);
    ggml_build_forward_expand(g.gf, d.pha_r);
    ggml_build_forward_expand(g.gf, d.pha_i);
    size_t n_tensors = 0;
    for (ggml_tensor * t = ggml_get_first_tensor(g.ctx); t; t = ggml_get_next_tensor(g.ctx, t)) ++n_tensors;
    const size_t n_nodes = (size_t) ggml_graph_n_nodes(g.gf);
    const size_t leaf_bound = n_tensors + w.size();
    std::printf("  capacity %zu: ctx tensors %zu, graph nodes %zu, leafs <= %zu\n",
                g.max_nodes, n_tensors, n_nodes, leaf_bound);
    CHECK(3 * n_tensors <= g.max_nodes);
    CHECK(3 * n_nodes <= g.max_nodes);
    CHECK(2 * leaf_bound <= g.max_nodes);
}

// Every matrix product in the forward graph must ask for an F32 src1 and F32 accumulation
// ([TAG_GGML_PREC]). Without it ggml Metal stages F32 x F32 mul_mat operands in half: ~1e-4
// relative error per block and inf/NaN once the TF-Mamba residual stream exceeds 65504 (44.1 kHz
// input, block 7). Checked structurally so it also guards CPU-only builds.
static void mul_mats_request_f32() {
    if (!std::getenv("REUSE_GGUF")) { std::printf("  SKIP: REUSE_GGUF not set\n"); return; }
    reuse_ctx * ctx = model();
    if (!ctx) { CHECK(false); return; }
    const Weights & w = ctx->weights;
    GraphRun g(graph_max_nodes(w.hp.num_tfmamba));
    ggml_tensor * in = ggml_new_tensor_3d(g.ctx, GGML_TYPE_F32, 883, 401, 2);  // 2 s at 44.1 kHz
    const DecoderOut d = forward(g.ctx, w, in);
    ggml_build_forward_expand(g.gf, d.mag);
    ggml_build_forward_expand(g.gf, d.pha_r);
    ggml_build_forward_expand(g.gf, d.pha_i);
    int n_mm = 0, n_bad = 0;
    for (int i = 0; i < ggml_graph_n_nodes(g.gf); ++i) {
        const ggml_tensor * n = ggml_graph_node(g.gf, i);
        if (n->op != GGML_OP_MUL_MAT) continue;
        ++n_mm;
        // op_params: [0] accumulator precision, [3] src1 precision (ggml_prec_set_acc / _src)
        if (n->op_params[0] != GGML_PREC_F32 || n->op_params[3] != GGML_PREC_F32) ++n_bad;
    }
    std::printf("  %d mul_mat nodes, %d without F32 precision\n", n_mm, n_bad);
    CHECK(n_mm == 30 * 2 * (2 * 4 + 1));  // 30 blocks x (time, freq) x (2 Mambas x 4 + output_proj)
    CHECK(n_bad == 0);
}

int main() {
    const TapTest tests[] = {
        {"block1_matches_fixture", block1_matches_fixture},
        {"blocks_1_to_30_match_fixture", blocks_1_to_30_match_fixture},
        {"graph_fits_node_budget", graph_fits_node_budget},
        {"mul_mats_request_f32", mul_mats_request_f32},
    };
    return run_tests(tests);
}
