// Graph execution on the context's backend and the run_tap test hook (see graph.h).
#include "graph.h"
#include "model.h"

#include <stdexcept>

GraphRun::GraphRun(size_t max_nodes) {
    ggml_init_params p = {
        /*.mem_size   =*/ ggml_tensor_overhead() * max_nodes + ggml_graph_overhead_custom(max_nodes, false),
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };
    ctx = ggml_init(p);
    if (!ctx) throw std::runtime_error("graph: ggml_init failed");
    gf = ggml_new_graph_custom(ctx, max_nodes, false);
}

GraphRun::~GraphRun() {
    if (galloc_) ggml_gallocr_free(galloc_);
    if (ctx) ggml_free(ctx);
}

void GraphRun::alloc(ggml_backend_t be, ggml_tensor * out) {
    ggml_set_output(out);
    ggml_build_forward_expand(gf, out);
    for (int i = 0; i < ggml_graph_n_nodes(gf); ++i) {
        ggml_tensor * n = ggml_graph_node(gf, i);
        if (!ggml_backend_supports_op(be, n))
            throw std::runtime_error(std::string("graph: backend ") + ggml_backend_name(be) +
                                     " does not support op " + ggml_op_desc(n) + " (node " + n->name + ")");
    }
    galloc_ = ggml_gallocr_new(ggml_backend_get_default_buffer_type(be));
    if (!galloc_ || !ggml_gallocr_alloc_graph(galloc_, gf))
        throw std::runtime_error(std::string("graph: cannot allocate compute buffer on ") + ggml_backend_name(be));
}

void GraphRun::compute(ggml_backend_t be) {
    if (ggml_backend_graph_compute(be, gf) != GGML_STATUS_SUCCESS)
        throw std::runtime_error(std::string("graph: compute failed on ") + ggml_backend_name(be));
}

// Builds the forward pass from the [F, T, 2] input up to `tap`; nullptr if the tap is unknown.
static ggml_tensor * build_tap(ggml_context * ctx, const Weights & w, const std::string & tap, ggml_tensor * mag_pha) {
    ggml_tensor * x = encoder(ctx, w, mag_pha);
    if (tap == "enc_out") return x;
    return nullptr;
}

std::vector<float> run_tap(reuse_ctx * rctx, const std::string & tap,
                           const std::vector<float> & mag, const std::vector<float> & pha,
                           int n_freq, int n_frames, std::vector<int64_t> * ne_out) {
    if (!rctx) throw std::runtime_error("run_tap: NULL context");
    const size_t n = (size_t) n_freq * (size_t) n_frames;
    if (n_freq < 1 || n_frames < 1 || mag.size() != n || pha.size() != n)
        throw std::runtime_error("run_tap: mag/pha size does not match n_freq * n_frames");
    ggml_backend_t be = rctx->backend.be;

    GraphRun g;
    ggml_tensor * in = ggml_new_tensor_3d(g.ctx, GGML_TYPE_F32, n_freq, n_frames, 2);
    ggml_set_name(in, "mag_pha");
    ggml_set_input(in);
    ggml_tensor * out = build_tap(g.ctx, rctx->weights, tap, in);
    if (!out) throw std::runtime_error("run_tap: unknown tap '" + tap + "'");
    g.alloc(be, out);
    ggml_backend_tensor_set(in, mag.data(), 0, n * sizeof(float));
    ggml_backend_tensor_set(in, pha.data(), n * sizeof(float), n * sizeof(float));
    g.compute(be);

    if (out->type != GGML_TYPE_F32 || !ggml_is_contiguous(out))
        throw std::runtime_error("run_tap: tap '" + tap + "' is not a contiguous F32 tensor");
    std::vector<float> res((size_t) ggml_nelements(out));
    ggml_backend_tensor_get(out, res.data(), 0, ggml_nbytes(out));
    if (ne_out) *ne_out = {out->ne[0], out->ne[1], out->ne[2], out->ne[3]};
    return res;
}
