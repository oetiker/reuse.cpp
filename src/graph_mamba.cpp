// TF-Mamba blocks (torch TSMamba.<i>): bidirectional Mamba-1 along time, then along frequency.
//
// Sequence layout inside a MambaBlock: u [D, L, n_seq] (channels innermost, one sequence per
// dim-2 slice), the ggml form of torch's [B*n_seq, L, D].
#include "graph.h"

#include <cinttypes>
#include <cstdio>
#include <string>

// ---- graph constants -------------------------------------------------------------------------

static ggml_tensor * graph_const(ggml_context * ctx, ggml_type type, const char * kind,
                                 int64_t ne0, int64_t ne1, int64_t ne2, int64_t ne3) {
    char name[GGML_MAX_NAME];
    std::snprintf(name, sizeof name, "const:%s:%" PRId64 "x%" PRId64 "x%" PRId64 "x%" PRId64,
                  kind, ne0, ne1, ne2, ne3);
    if (ggml_tensor * t = ggml_get_tensor(ctx, name)) return t;
    ggml_tensor * t = ggml_new_tensor_4d(ctx, type, ne0, ne1, ne2, ne3);
    ggml_set_name(t, name);
    ggml_set_input(t);
    return t;
}

ggml_tensor * const_zeros_f32(ggml_context * ctx, int64_t ne0, int64_t ne1, int64_t ne2, int64_t ne3) {
    return graph_const(ctx, GGML_TYPE_F32, "zero", ne0, ne1, ne2, ne3);
}

ggml_tensor * const_zeros_i32(ggml_context * ctx, int64_t n) {
    return graph_const(ctx, GGML_TYPE_I32, "zero", n, 1, 1, 1);
}

ggml_tensor * const_rev_i32(ggml_context * ctx, int64_t L, int64_t n) {
    ggml_tensor * t = graph_const(ctx, GGML_TYPE_I32, "rev", L, n, 1, 1);
    return ggml_reshape_2d(ctx, t, L, n);
}

// ---- Mamba -------------------------------------------------------------------------------

// Reverses dimension 1 (the token axis) of u [D, L, n].
static ggml_tensor * flip_tokens(ggml_context * ctx, ggml_tensor * u) {
    return ggml_get_rows(ctx, u, const_rev_i32(ctx, u->ne[1], u->ne[2]));
}

// mamba_ssm.Mamba.forward (v1, see tools/mamba_ssm) on u [d_model, L, n_seq]; `p` is the module
// prefix. Returns [d_model, L, n_seq].
static ggml_tensor * mamba(ggml_context * ctx, const Weights & w, const std::string & p, ggml_tensor * u) {
    const int64_t L = u->ne[1], n_seq = u->ne[2];
    const ggml_tensor * A = w.get(p + ".A");          // {d_state, d_inner}
    const int64_t d_state = A->ne[0], d_inner = A->ne[1];
    ggml_tensor * conv_w = wt(w, p + ".conv1d.weight");  // {d_conv, 1, d_inner}
    const int64_t d_conv = conv_w->ne[0];
    const int64_t dt_rank = w.get(p + ".dt_proj.weight")->ne[0];

    // in_proj (no bias) -> x, z = chunk(2) along channels
    ggml_tensor * xz = ggml_mul_mat(ctx, wt(w, p + ".in_proj.weight"), u);  // [2*d_inner, L, n_seq]
    ggml_tensor * x = ggml_view_3d(ctx, xz, d_inner, L, n_seq, xz->nb[1], xz->nb[2], 0);
    ggml_tensor * z = ggml_view_3d(ctx, xz, d_inner, L, n_seq, xz->nb[1], xz->nb[2],
                                   d_inner * ggml_element_size(xz));

    // causal depthwise conv1d: tokens innermost, d_conv-1 zeros on the left, + bias, silu
    ggml_tensor * sx = ggml_cont(ctx, ggml_transpose(ctx, x));                   // [L, d_inner, n_seq]
    sx = ggml_pad_ext(ctx, sx, (int) (d_conv - 1), 0, 0, 0, 0, 0, 0, 0);          // [d_conv-1+L, ...]
    x = ggml_ssm_conv(ctx, sx, ggml_reshape_2d(ctx, conv_w, d_conv, d_inner));  // [d_inner, L, n_seq]
    x = ggml_silu(ctx, ggml_add(ctx, x, wt(w, p + ".conv1d.bias")));

    // x_proj -> dt (dt_rank), B (d_state), C (d_state)
    ggml_tensor * xdb = ggml_mul_mat(ctx, wt(w, p + ".x_proj.weight"), x);  // [dt_rank+2*d_state, L, n_seq]
    const size_t es = ggml_element_size(xdb);
    ggml_tensor * dt = ggml_view_3d(ctx, xdb, dt_rank, L, n_seq, xdb->nb[1], xdb->nb[2], 0);
    ggml_tensor * B = ggml_view_4d(ctx, xdb, d_state, 1, L, n_seq, d_state * es, xdb->nb[1], xdb->nb[2],
                                   dt_rank * es);
    ggml_tensor * C = ggml_view_4d(ctx, xdb, d_state, 1, L, n_seq, d_state * es, xdb->nb[1], xdb->nb[2],
                                   (dt_rank + d_state) * es);

    // dt_proj weight, then + dt_proj.bias; softplus is applied inside ggml_ssm_scan (as in
    // selective_scan_ref: softplus(dt + delta_bias)), so it must not be applied here.
    dt = ggml_mul_mat(ctx, wt(w, p + ".dt_proj.weight"), dt);  // [d_inner, L, n_seq]
    dt = ggml_add(ctx, dt, wt(w, p + ".dt_proj.bias"));

    // Mamba-1 scan: head_dim 1, n_head d_inner, n_group 1, zero initial state for every sequence.
    ggml_tensor * s0 = const_zeros_f32(ctx, d_state, 1, d_inner, 1);
    ggml_tensor * ids = const_zeros_i32(ctx, n_seq);
    ggml_tensor * ys = ggml_ssm_scan(ctx, s0, ggml_reshape_4d(ctx, x, 1, d_inner, L, n_seq), dt,
                                     const_cast<ggml_tensor *>(A), B, C, ids, 1);
    // result = y {d_inner, L, n_seq} followed by the final states; keep y
    ggml_tensor * y = ggml_view_3d(ctx, ys, d_inner, L, n_seq, d_inner * ggml_element_size(ys),
                                   d_inner * L * ggml_element_size(ys), 0);

    // + u * D, gate with silu(z), out_proj (no bias)
    y = ggml_add(ctx, y, ggml_mul(ctx, x, wt(w, p + ".D")));
    y = ggml_mul(ctx, y, ggml_silu(ctx, ggml_cont(ctx, z)));
    return ggml_mul_mat(ctx, wt(w, p + ".out_proj.weight"), y);  // [d_model, L, n_seq]
}

// MambaBlock (mamba_block2_SEMamba.py) on u [D, L, n_seq]: forward Mamba + residual, backward
// Mamba on the token-reversed sequence + residual (reversed back), concat (fw first),
// output_proj, LayerNorm.
static ggml_tensor * mamba_block(ggml_context * ctx, const Weights & w, const std::string & p, ggml_tensor * u) {
    ggml_tensor * fw = ggml_add(ctx, mamba(ctx, w, p + ".forward_blocks", u), u);
    ggml_tensor * ur = flip_tokens(ctx, u);
    ggml_tensor * bw = ggml_add(ctx, mamba(ctx, w, p + ".backward_blocks", ur), ur);
    bw = flip_tokens(ctx, bw);
    ggml_tensor * y = ggml_concat(ctx, fw, bw, 0);  // [2D, L, n_seq]
    y = ggml_mul_mat(ctx, wt(w, p + ".output_proj.weight"), y);
    y = ggml_add(ctx, y, wt(w, p + ".output_proj.bias"));
    y = ggml_norm(ctx, y, 1e-5f);
    y = ggml_mul(ctx, y, wt(w, p + ".norm.weight"));
    return ggml_add(ctx, y, wt(w, p + ".norm.bias"));
}

ggml_tensor * tf_mamba_block(ggml_context * ctx, const Weights & w, int i, ggml_tensor * x) {
    const std::string p = "TSMamba." + std::to_string(i);
    // [F, T, C] -> [C, T, F]: sequences along time, one per frequency bin
    ggml_tensor * u = ggml_cont(ctx, ggml_permute(ctx, x, 2, 1, 0, 3));
    u = ggml_add(ctx, mamba_block(ctx, w, p + ".time_mamba", u), u);
    // [C, T, F] -> [C, F, T]: sequences along frequency, one per frame
    u = ggml_cont(ctx, ggml_permute(ctx, u, 0, 2, 1, 3));
    u = ggml_add(ctx, mamba_block(ctx, w, p + ".freq_mamba", u), u);
    // [C, F, T] -> [F, T, C]
    return ggml_cont(ctx, ggml_permute(ctx, u, 2, 0, 1, 3));
}

size_t graph_max_nodes(int n_blocks) {
    // Measured: encoder + 30 blocks = 4066 context tensors (~135 per TF-Mamba block: 2 passes x
    // 2 directions x ~27 per Mamba, plus flips, projections, norms, permutes). 512 per block and
    // 2048 base give ~4x headroom (checked by test_mamba's graph_fits_node_budget).
    return 2048 + (size_t) (n_blocks > 0 ? n_blocks : 0) * 512;
}
