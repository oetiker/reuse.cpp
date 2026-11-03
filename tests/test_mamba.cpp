// TF-Mamba block tests against the PyTorch fixtures (need REUSE_GGUF; skip otherwise).
// REUSE_DEVICE selects the backend device (default CPU).
#include "tap_util.h"

// Relative max error tolerances; see tests/TOLERANCES.md.
static const double TOL_BLOCK1 = 1e-5;
static const double TOL_BLOCK30 = 2e-5;

static void block1_matches_fixture() { check_tap("block1_out", "block1_out", TOL_BLOCK1); }

static void blocks_1_to_30_match_fixture() { check_tap("block30_out", "block30_out", TOL_BLOCK30); }

// The full encoder + all TF-Mamba blocks graph must stay well inside GraphRun's capacity
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
    ggml_tensor * x = encoder(g.ctx, w, in);
    for (int i = 0; i < w.hp.num_tfmamba; ++i) x = tf_mamba_block(g.ctx, w, i, x);
    ggml_build_forward_expand(g.gf, x);
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

int main() {
    const TapTest tests[] = {
        {"block1_matches_fixture", block1_matches_fixture},
        {"blocks_1_to_30_match_fixture", blocks_1_to_30_match_fixture},
        {"graph_fits_node_budget", graph_fits_node_budget},
    };
    return run_tests(tests);
}
