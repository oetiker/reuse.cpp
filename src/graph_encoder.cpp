// Dense encoder graph and the shared conv/norm/PReLU/DenseBlock helpers (see graph.h).
#include "graph.h"

#include <string>

static ggml_tensor * per_channel(ggml_context * ctx, const ggml_tensor * v) {
    // [C] -> [1, 1, C, 1] for broadcasting over [F, T, C, N].
    return ggml_reshape_4d(ctx, const_cast<ggml_tensor *>(v), 1, 1, v->ne[0], 1);
}

ggml_tensor * conv_2d_f32(ggml_context * ctx, ggml_tensor * kernel, ggml_tensor * x,
                          int s0, int s1, int p0, int p1, int d0, int d1) {
    ggml_tensor * y = ggml_conv_2d_direct(ctx, kernel, x, s0, s1, p0, p1, d0, d1);
    ggml_prec_set_acc(y, GGML_PREC_F32);
    return y;
}

ggml_tensor * conv2d(ggml_context * ctx, const Weights & w, const std::string & prefix, ggml_tensor * x,
                     int stride_t, int stride_f, int pad_t, int pad_f, int dil_t, int dil_f) {
    // ggml dim 0 = frequency (torch kw), dim 1 = time (torch kh).
    ggml_tensor * y = conv_2d_f32(ctx, wt(w, prefix + ".weight"), x,
                                  stride_f, stride_t, pad_f, pad_t, dil_f, dil_t);
    return ggml_add(ctx, y, per_channel(ctx, w.get(prefix + ".bias")));
}

ggml_tensor * instance_norm_affine(ggml_context * ctx, ggml_tensor * x, const ggml_tensor * w, const ggml_tensor * b) {
    if (!ggml_is_contiguous(x)) x = ggml_cont(ctx, x);
    const int64_t F = x->ne[0], T = x->ne[1], C = x->ne[2], N = x->ne[3];
    ggml_tensor * y = ggml_reshape_3d(ctx, x, F * T, C, N);
    y = ggml_norm(ctx, y, 1e-5f);
    y = ggml_mul(ctx, y, ggml_reshape_2d(ctx, const_cast<ggml_tensor *>(w), 1, C));
    y = ggml_add(ctx, y, ggml_reshape_2d(ctx, const_cast<ggml_tensor *>(b), 1, C));
    return ggml_reshape_4d(ctx, y, F, T, C, N);
}

ggml_tensor * prelu(ggml_context * ctx, ggml_tensor * x, const ggml_tensor * a) {
    ggml_tensor * pos = ggml_relu(ctx, x);
    ggml_tensor * neg = ggml_relu(ctx, ggml_neg(ctx, x));
    return ggml_sub(ctx, pos, ggml_mul(ctx, neg, per_channel(ctx, a)));
}

ggml_tensor * conv_norm_prelu(ggml_context * ctx, const Weights & w, const std::string & prefix, ggml_tensor * x,
                              int stride_t, int stride_f, int pad_t, int pad_f, int dil_t, int dil_f) {
    ggml_tensor * y = conv2d(ctx, w, prefix + ".0", x, stride_t, stride_f, pad_t, pad_f, dil_t, dil_f);
    y = instance_norm_affine(ctx, y, w.get(prefix + ".1.weight"), w.get(prefix + ".1.bias"));
    return prelu(ctx, y, w.get(prefix + ".2.weight"));
}

ggml_tensor * dense_block(ggml_context * ctx, const Weights & w, const std::string & prefix, ggml_tensor * x) {
    const int depth = 4;
    ggml_tensor * skip = x;
    for (int i = 0; i < depth; ++i) {
        const int dil = 1 << i;
        // kernel (3,3), dilation (dil, 1), padding get_padding_2d((3,3), (dil,1)) = (dil, 1)
        x = conv_norm_prelu(ctx, w, prefix + ".dense_block." + std::to_string(i), skip,
                            1, 1, dil, 1, dil, 1);
        if (i + 1 < depth) skip = ggml_concat(ctx, x, skip, 2);  // torch.cat([x, skip], dim=1)
    }
    return x;
}

ggml_tensor * encoder(ggml_context * ctx, const Weights & w, ggml_tensor * mag_pha) {
    // "Prevent unpredictable errors": 2 zero frequency bins and 2 zero frames appended.
    ggml_tensor * x = ggml_pad(ctx, mag_pha, 2, 2, 0, 0);
    x = conv_norm_prelu(ctx, w, "dense_encoder.dense_conv_1", x, 1, 1, 0, 0, 1, 1);
    x = dense_block(ctx, w, "dense_encoder.dense_block", x);
    // kernel (1,3), stride (time 4, freq 2), no padding
    return conv_norm_prelu(ctx, w, "dense_encoder.dense_conv_2", x, 4, 2, 0, 0, 1, 1);
}
