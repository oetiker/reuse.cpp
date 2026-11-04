// Graph tap tests against the PyTorch fixtures (need REUSE_GGUF; skip otherwise).
// REUSE_DEVICE selects the backend device (default CPU).
#include "tap_util.h"

// Relative max error tolerance for enc_out; see tests/TOLERANCES.md.
static const double TOL_ENC = 1e-5;

static void encoder_matches_fixture() { check_tap("enc_out", "enc_out", TOL_ENC); }

static void run_tap_rejects_unknown_tap() {
    if (!std::getenv("REUSE_GGUF")) { std::printf("  SKIP: REUSE_GGUF not set\n"); return; }
    reuse_ctx * ctx = model();
    if (!ctx) { CHECK(false); return; }
    std::vector<float> z(16 * 8, 0.f);
    bool threw = false;
    try { run_tap(ctx, "no_such_tap", z, z, 16, 8); }
    catch (const std::exception & e) { threw = std::string(e.what()).find("no_such_tap") != std::string::npos; }
    CHECK(threw);
}

// Every CONV_2D in the full model graph must request F32 accumulation ([TAG_GGML_PREC],
// op_params[6] in the fork). Without it ggml Vulkan on coopmat devices (e.g. RADV with
// KHR_cooperative_matrix) runs conv2d through the cm1 shader, which stages both operands and
// accumulates in float16: ~5e-3 relative error at enc_out. Checked structurally so it also guards
// builds without such a device.
static void conv2ds_request_f32() {
    if (!std::getenv("REUSE_GGUF")) { std::printf("  SKIP: REUSE_GGUF not set\n"); return; }
    reuse_ctx * ctx = model();
    if (!ctx) { CHECK(false); return; }
    const Weights & w = ctx->weights;
    GraphRun g(graph_max_nodes(w.hp.num_tfmamba));
    ggml_tensor * in = ggml_new_tensor_3d(g.ctx, GGML_TYPE_F32, 481, 201, 2);
    const DecoderOut d = forward(g.ctx, w, in);
    ggml_build_forward_expand(g.gf, d.mag);
    ggml_build_forward_expand(g.gf, d.pha_r);
    ggml_build_forward_expand(g.gf, d.pha_i);
    int n_conv = 0, n_bad = 0;
    for (int i = 0; i < ggml_graph_n_nodes(g.gf); ++i) {
        const ggml_tensor * n = ggml_graph_node(g.gf, i);
        if (n->op != GGML_OP_CONV_2D) continue;
        ++n_conv;
        if (n->op_params[6] != GGML_PREC_F32) ++n_bad;
    }
    std::printf("  %d conv_2d nodes, %d without F32 precision\n", n_conv, n_bad);
    // encoder 6; per decoder dense_block 4 + 2 up_convs; final_conv 1, phase_conv_r/i 2
    CHECK(n_conv == 6 + 2 * 6 + 1 + 2);
    CHECK(n_bad == 0);
}

int main() {
    const TapTest tests[] = {
        {"encoder_matches_fixture", encoder_matches_fixture},
        {"run_tap_rejects_unknown_tap", run_tap_rejects_unknown_tap},
        {"conv2ds_request_f32", conv2ds_request_f32},
    };
    return run_tests(tests);
}
