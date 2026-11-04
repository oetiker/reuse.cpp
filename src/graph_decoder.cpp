// Magnitude (mask_decoder) and phase (phase_decoder) decoders and the full forward graph
// (see graph.h).
#include "graph.h"

#include <string>

// SPConvTranspose2d(kernel (1,3), r) + InstanceNorm2d + PReLU, the nn.Sequential `prefix`
// ("<decoder>.up_conv1" / "<decoder>.up_conv2") on x [F, T, C, 1].
//
// torch: pad the conv's width axis by 1/1, conv to C*r channels, then
//   out.view(B, r, C, H, W).permute(0, 2, 3, 4, 1).view(B, C, H, W*r)
// i.e. out[c, h, w*r + k] = conv[k*C + c, h, w]: the r sub-channels interleave into the width.
// up_conv1 runs on [B, C, T, F] (width = frequency); up_conv2 runs on the T/F-permuted tensor
// (width = time). Here the time variant reinterprets the {3, 1} kernel as {1, 3} (same memory
// order) and convolves along time directly instead of permuting the activations.
static ggml_tensor * up_conv(ggml_context * ctx, const Weights & w, const std::string & prefix, ggml_tensor * x,
                             int r, bool along_time) {
    const std::string cp = prefix + ".0.conv";
    ggml_tensor * k = wt(w, cp + ".weight");  // {3, 1, C, C*r}
    if (along_time) k = ggml_reshape_4d(ctx, k, 1, 3, k->ne[2], k->ne[3]);
    ggml_tensor * y = conv_2d_f32(ctx, k, x, 1, 1, along_time ? 0 : 1, along_time ? 1 : 0, 1, 1);
    const ggml_tensor * bias = w.get(cp + ".bias");
    y = ggml_add(ctx, y, ggml_reshape_4d(ctx, const_cast<ggml_tensor *>(bias), 1, 1, bias->ne[0], 1));

    const int64_t F = y->ne[0], T = y->ne[1], C = y->ne[2] / r;
    y = ggml_reshape_4d(ctx, y, F, T, C, r);  // channel k*C + c -> (c, k)
    if (along_time) {
        // {F, T, C, r} -> {F, r, T, C} -> [F, T*r, C]
        y = ggml_cont(ctx, ggml_permute(ctx, y, 0, 2, 3, 1));
        y = ggml_reshape_4d(ctx, y, F, T * r, C, 1);
    } else {
        // {F, T, C, r} -> {r, F, T, C} -> [F*r, T, C]
        y = ggml_cont(ctx, ggml_permute(ctx, y, 1, 2, 3, 0));
        y = ggml_reshape_4d(ctx, y, F * r, T, C, 1);
    }
    y = instance_norm_affine(ctx, y, w.get(prefix + ".1.weight"), w.get(prefix + ".1.bias"));
    return prelu(ctx, y, w.get(prefix + ".2.weight"));
}

// Shared decoder trunk: dense_block, up_conv1 (r=2 along frequency), up_conv2 (r=4 along time).
// [F', T', C] -> [2F', 4T', C].
static ggml_tensor * decoder_trunk(ggml_context * ctx, const Weights & w, const std::string & p, ggml_tensor * x) {
    x = dense_block(ctx, w, p + ".dense_block", x);
    x = up_conv(ctx, w, p + ".up_conv1", x, 2, false);
    return up_conv(ctx, w, p + ".up_conv2", x, 4, true);
}

DecoderOut decoders(ggml_context * ctx, const Weights & w, ggml_tensor * x) {
    DecoderOut o;
    ggml_tensor * m = decoder_trunk(ctx, w, "mask_decoder", x);
    o.mag = conv2d(ctx, w, "mask_decoder.final_conv", m, 1, 1, 0, 0, 1, 1);
    ggml_tensor * p = decoder_trunk(ctx, w, "phase_decoder", x);
    o.pha_r = conv2d(ctx, w, "phase_decoder.phase_conv_r", p, 1, 1, 0, 0, 1, 1);
    o.pha_i = conv2d(ctx, w, "phase_decoder.phase_conv_i", p, 1, 1, 0, 0, 1, 1);
    return o;
}

DecoderOut forward(ggml_context * ctx, const Weights & w, ggml_tensor * mag_pha) {
    ggml_tensor * x = encoder(ctx, w, mag_pha);
    for (int i = 0; i < w.hp.num_tfmamba; ++i) x = tf_mamba_block(ctx, w, i, x);
    return decoders(ctx, w, x);
}
