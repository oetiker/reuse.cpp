#pragma once
// Internal graph-building helpers for the RE-USE (SEMamba) model.
//
// Layout: activations are 4-D ggml tensors [F, T, C, 1] (frequency innermost), i.e. the
// reverse of torch's [B, C, T, F]. Torch conv weights [OC, IC, kh(time), kw(freq)] arrive as
// ggml ne {kw, kh, IC, OC}, so ggml dimension 0 (conv "width") is frequency and dimension 1
// (conv "height") is time: torch (time, freq) stride/padding/dilation map to (s1/p1/d1, s0/p0/d0).
//
// The builders below only create graph nodes with standard ggml ops; they contain no
// backend-specific code. Execution happens through GraphRun on the context's backend.
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "weights.h"

#include <cstdint>
#include <string>
#include <vector>

struct reuse_ctx;

// Non-const view of a weight tensor (ggml op builders take non-const pointers; weights are
// never written by the graph).
inline ggml_tensor * wt(const Weights & w, const std::string & name) {
    return const_cast<ggml_tensor *>(w.get(name));
}

// torch.nn.Conv2d with bias on x [F, T, IC, 1]; kernel/stride/padding/dilation are given in
// torch order (time, freq). `prefix` names the conv module ("<prefix>.weight", "<prefix>.bias").
ggml_tensor * conv2d(ggml_context * ctx, const Weights & w, const std::string & prefix, ggml_tensor * x,
                     int stride_t, int stride_f, int pad_t, int pad_f, int dil_t, int dil_f);

// nn.InstanceNorm2d(affine=True), eps 1e-5: normalise each channel over F*T, then per-channel affine.
ggml_tensor * instance_norm_affine(ggml_context * ctx, ggml_tensor * x, const ggml_tensor * w, const ggml_tensor * b);

// nn.PReLU(C): relu(x) - a * relu(-x), per-channel a.
ggml_tensor * prelu(ggml_context * ctx, ggml_tensor * x, const ggml_tensor * a);

// Conv2d -> InstanceNorm2d -> PReLU as an nn.Sequential named `prefix` (sub-modules .0/.1/.2).
ggml_tensor * conv_norm_prelu(ggml_context * ctx, const Weights & w, const std::string & prefix, ggml_tensor * x,
                              int stride_t, int stride_f, int pad_t, int pad_f, int dil_t, int dil_f);

// DenseBlock(depth 4, kernel (3,3)): layer i uses dilation (2^i, 1), padding (2^i, 1); its input
// is the channel concat of all previous outputs, newest first (torch.cat([x, skip], dim=1)).
// `prefix` is the DenseBlock module, e.g. "dense_encoder.dense_block". Returns the last output.
ggml_tensor * dense_block(ggml_context * ctx, const Weights & w, const std::string & prefix, ggml_tensor * x);

// DenseEncoder on mag_pha [F, T, 2] (channel 0 magnitude, 1 phase): pads 2 zero bins and 2 zero
// frames at the end, dense_conv_1, dense_block, dense_conv_2 (kernel (1,3), stride (4,2)).
// Returns [F', T', 64, 1] with F' = (F+2-3)/2+1, T' = (T+2-1)/4+1.
ggml_tensor * encoder(ggml_context * ctx, const Weights & w, ggml_tensor * mag_pha);

// Graph constants: leaf tensors marked as inputs whose contents GraphRun::alloc generates and
// uploads (no host data is needed at build time). Each is created once per context and shared
// by every user (looked up by name, "const:<kind>:<shape>").
//   const_zeros_f32: all zeros (e.g. the initial SSM state).
//   const_zeros_i32: all zeros (e.g. ssm_scan state ids: every sequence starts from state 0).
//   const_rev_i32:   [L, n] with element (j, s) = L-1-j: get_rows indices that reverse dimension 1
//                    of a [D, L, n] tensor.
ggml_tensor * const_zeros_f32(ggml_context * ctx, int64_t ne0, int64_t ne1, int64_t ne2, int64_t ne3);
ggml_tensor * const_zeros_i32(ggml_context * ctx, int64_t n);
ggml_tensor * const_rev_i32(ggml_context * ctx, int64_t L, int64_t n);

// One TFMambaBlock (torch module "TSMamba.<i>") on x [F', T', C, 1] (C = hid_feature):
// x = time(x) + x with sequences along T (one per frequency bin), then x = freq(x) + x with
// sequences along F (one per frame). time/freq are bidirectional MambaBlocks (Mamba-1 via
// ggml_ssm_conv / ggml_ssm_scan). Returns [F', T', C, 1].
ggml_tensor * tf_mamba_block(ggml_context * ctx, const Weights & w, int i, ggml_tensor * x);

// Decoder outputs before the crop to the STFT size: each [2F', 4T', 1, 1] (torch mask_decoder
// output, phase_decoder.phase_conv_r / phase_conv_i outputs; the phase is atan2(pha_i, pha_r)).
struct DecoderOut { ggml_tensor * mag = nullptr, * pha_r = nullptr, * pha_i = nullptr; };

// MagDecoder and PhaseDecoder on the last TF-Mamba block output x [F', T', C, 1]: dense_block,
// SPConvTranspose2d r=2 along frequency, r=4 along time (each + InstanceNorm2d + PReLU), then
// the 1x1 output convs.
DecoderOut decoders(ggml_context * ctx, const Weights & w, ggml_tensor * x);

// The whole model graph on mag_pha [F, T, 2]: encoder, all TF-Mamba blocks, decoders.
DecoderOut forward(ggml_context * ctx, const Weights & w, ggml_tensor * mag_pha);

// GraphRun node/tensor capacity for forward() with `n_blocks` TF-Mamba blocks, with a
// wide margin (a ggml context overflow aborts the process, so this must never be tight).
size_t graph_max_nodes(int n_blocks);

// Test hook (not part of the public C API): builds the forward graph from STFT inputs
// mag/pha (each n_freq * n_frames floats, frequency innermost) up to the named tap, runs it on
// ctx's backend, and returns the tap's data (ggml order; shape in *ne_out if given).
// Taps: "enc_out", "block<N>_out" (output of TF-Mamba block N, 1-based, N <= num_tfmamba),
// "mag_out", "pha_r", "pha_i" (decoder outputs before the crop, see DecoderOut).
// Throws std::runtime_error on unknown tap or backend failure.
std::vector<float> run_tap(reuse_ctx * ctx, const std::string & tap,
                           const std::vector<float> & mag, const std::vector<float> & pha,
                           int n_freq, int n_frames, std::vector<int64_t> * ne_out = nullptr);

// Test hook: runs only the decoders on x [ne0, ne1, ne2] (a TF-Mamba block output, ggml order)
// and returns tap "mag_out", "pha_r" or "pha_i" (pre-crop). Throws like run_tap.
std::vector<float> run_decoder_tap(reuse_ctx * ctx, const std::string & tap, const std::vector<float> & x,
                                   int64_t ne0, int64_t ne1, int64_t ne2, std::vector<int64_t> * ne_out = nullptr);

// Owns a no-alloc ggml context + graph and the backend compute buffer for one forward pass.
// Usage: build nodes in g.ctx; g.alloc(be, out or {outs...}) (expands the graph, verifies every node is
// supported by `be` — there is no CPU fallback — and allocates); upload inputs with
// ggml_backend_tensor_set; g.compute(be); read outputs with ggml_backend_tensor_get.
// alloc also uploads the contents of all graph constants (const_* above).
struct GraphRun {
    explicit GraphRun(size_t max_nodes = 16384);
    ~GraphRun();
    GraphRun(const GraphRun &) = delete;
    GraphRun & operator=(const GraphRun &) = delete;

    void alloc(ggml_backend_t be, ggml_tensor * out) { alloc(be, std::vector<ggml_tensor *>{out}); }
    void alloc(ggml_backend_t be, const std::vector<ggml_tensor *> & outs);
    void compute(ggml_backend_t be);

    ggml_context * ctx = nullptr;
    ggml_cgraph * gf = nullptr;
    size_t max_nodes = 0;  // capacity of ctx (tensors) and gf (nodes, leafs)

private:
    void upload_consts();
    ggml_gallocr_t galloc_ = nullptr;
};
