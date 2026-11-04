#pragma once
#include <cstdint>
#include <cstring>
#include <cassert>

namespace MicroNAM {

// ─────────────────────────────────────────────────────────────────────────────
// A2 Full WaveNet inference engine
//
// A2 Full architecture (SlimmableContainer submodel at max_value 1.0):
//   23 layers × 8-channel
//   Kernel sizes: 6 everywhere, except layers 14 and 15 which use 15
//   Dilations:    [1,3,7,17,41,101,239] twice, then [1,13], then
//                 [1,3,7,17,41,101,239]
//   Every layer:  conv(K taps, dilation d) + conv_b + mixin_w * input
//                 -> LeakyReLU(0.01)
//                 -> head_sum[ch] += activated[ch]
//                 -> x = x + b1x1 + activated @ w1x1
//   Head:         16-tap causal FIR over the 8-channel head_sum across time,
//                 + head_b, then * head_scale
//
// load_weights expects exactly 12146 floats in the NAM file weight order.
// forward processes exactly NAM_BLOCK samples per call.
// ─────────────────────────────────────────────────────────────────────────────

template<size_t NAM_BLOCK>
class A2FullNet
{
public:
  // ── API ───────────────────────────────────────────────────────────────────
  void reset() { reset_impl(); }
  void load_weights(const float* model_weights) { load_weights_impl(model_weights); }
  void forward(const float* input, float* output) noexcept { forward_impl(input, output); }
  //   ^ Process exactly NAM_BLOCK samples.
  // ──────────────────────────────────────────────────────────────────────────

private:
  // ─────────────────────────────────────────────────────────────────────────────
  // LeakyReLU(0.01) activation
  // ─────────────────────────────────────────────────────────────────────────────
  static inline float nam_leaky_relu(float x) noexcept
  {
    return (x > 0.0f) ? x : 0.01f * x;
  }

  // ── 8-channel layer ───────────────────────────────────────────────────────
  struct Layer8
  {
    int    k;
    int    d;
    int    state_size;
    int    ptr;
    int    offset[15];     // ring delta per tap: (ptr + offset[t]) % ss
    float* state;

    float* wt;             // transposed conv: [tap][in][out]
    float  conv_b[8];
    float  mixin_w[8];
    float  w1x1[8 * 8];    // 1×1 projection, [out][in]
    float  b1x1[8];
  };

  // ── Layer stack: 23 × 8-channel ───────────────────────────────────────────
  float  rechannel_w[8];    // 1→8 input projection
  float  head_w[16][8];     // [tap][channel]
  float  head_b;
  float  head_scale;
  Layer8 layers[23];

  // State: sum((k-1)*d + 1) * 8 = 6354 * 8 = 50832 floats
  alignas(16) float state_pool[50832];
  // Conv weights: sum(k) * 8 * 8 = 156 * 64 = 9984 floats
  alignas(16) float conv_pool[9984];

  alignas(16) float head_hist[16][8];  // ring of head_sum history
  int               head_ptr;

  // ── Block working buffers ─────────────────────────────────────────────────
  alignas(16) float buf_a[NAM_BLOCK][8];
  alignas(16) float buf_b[NAM_BLOCK][8];
  alignas(16) float head_acc[NAM_BLOCK][8];

  void init_state_ptrs()
  {
    static constexpr int ks[23] = {
      6, 6, 6, 6, 6, 6, 6,  6, 6, 6, 6, 6, 6, 6,
      15, 15,
      6, 6, 6, 6, 6, 6, 6
    };
    static constexpr int ds[23] = {
      1, 3, 7, 17, 41, 101, 239,
      1, 3, 7, 17, 41, 101, 239,
      1, 13,
      1, 3, 7, 17, 41, 101, 239
    };

    int soff = 0;
    int coff = 0;
    for (int l = 0; l < 23; ++l)
    {
      Layer8&   L  = layers[l];
      const int k  = ks[l];
      const int d  = ds[l];
      const int ss = (k - 1) * d + 1;

      L.k          = k;
      L.d          = d;
      L.state_size = ss;
      L.ptr        = 0;
      L.state      = state_pool + soff;
      soff        += ss * 8;
      L.wt         = conv_pool + coff;
      coff        += k * 8 * 8;

      // tap t: 0 == oldest, k-1 == current sample
      for (int t = 0; t < k; ++t)
        L.offset[t] = ss - (k - 1 - t) * d;
    }
  }

  void reset_impl()
  {
    for (int i = 0; i < 50832; ++i) state_pool[i] = 0.0f;
    for (int t = 0; t < 16; ++t)
      for (int c = 0; c < 8; ++c) head_hist[t][c] = 0.0f;
    head_ptr = 0;
    for (int l = 0; l < 23; ++l) layers[l].ptr = 0;
  }

  // ─────────────────────────────────────────────────────────────────────────────
  // load_weights — float[12146]
  //
  // File weight order:
  //   rechannel: [8]
  //   per layer: conv[out][in][k], conv_b[8], mixin_w[8], w1x1[out][in], b1x1[8]
  //   head:      head_w[channel][tap] (8 channels outer, 16 taps inner)
  //              head_b
  //              head_scale
  //
  // Weight tally:
  //   rechannel_w:  8
  //   23 layers:    21 layers × (6*8*8 + 8 + 8 + 8*8 + 8)   = 21 × 472  = 9912
  //                + 2 layers × (15*8*8 + 8 + 8 + 8*8 + 8)  =  2 × 1048 = 2096
  //                = 12008
  //   head_w:       8 × 16 = 128
  //   head_b:       1
  //   head_scale:   1
  //   Total:        12146
  // ─────────────────────────────────────────────────────────────────────────────
  void load_weights_impl(const float* model_weights)
  {
    init_state_ptrs();
    reset_impl();

    const float* w = model_weights;

    // ── Rechannel ─────────────────────────────────────────────────────────
    for (int c = 0; c < 8; ++c) rechannel_w[c] = *w++;

    // ── Layers ────────────────────────────────────────────────────────────
    for (int l = 0; l < 23; ++l)
    {
      Layer8&   L  = layers[l];
      const int K  = L.k;
      float*    wt = L.wt;

      // Source is PyTorch [out][in][k]; transpose to [tap][in][out]
      for (int o = 0; o < 8; ++o)
        for (int i = 0; i < 8; ++i)
          for (int k = 0; k < K; ++k)
            wt[(k * 8 + i) * 8 + o] = *w++;

      for (int o = 0; o < 8; ++o) L.conv_b[o]  = *w++;
      for (int o = 0; o < 8; ++o) L.mixin_w[o] = *w++;
      for (int o = 0; o < 8; ++o)
        for (int i = 0; i < 8; ++i)
          L.w1x1[o * 8 + i] = *w++;
      for (int o = 0; o < 8; ++o) L.b1x1[o] = *w++;
    }

    // ── Head FIR + bias + scale ───────────────────────────────────────────
    // File stream is [channel][tap]; untangle to [tap][channel]
    for (int c = 0; c < 8; ++c)
      for (int t = 0; t < 16; ++t)
        head_w[t][c] = *w++;

    head_b     = *w++;
    head_scale = *w++;

    assert((int)(w - model_weights) == 12146);
  }

  // ─────────────────────────────────────────────────────────────────────────────
  // forward — process exactly NAM_BLOCK samples
  //
  // Structure (layer-outer, sample-inner):
  //
  //   1. Rechannel input[n] → buf_a[n][8], zero head_acc
  //   2. For each of 23 layers:
  //        Layer8_process_block(ping, pong, input, head_acc)
  //        swap ping/pong
  //   3. Head: 16-tap causal FIR over 8-channel head_acc across time
  //   4. Output: output[n] = head_scale * (head_b + FIR_output)
  // ─────────────────────────────────────────────────────────────────────────────
  void forward_impl(const float* input, float* output) noexcept
  {
    const int N = (int)NAM_BLOCK;

    // ── Rechannel: 1→8 and zero head accumulator ─────────────────────────
    for (int n = 0; n < N; ++n)
    {
      const float x = input[n];
      for (int c = 0; c < 8; ++c) buf_a[n][c] = rechannel_w[c] * x;
      for (int c = 0; c < 8; ++c) head_acc[n][c] = 0.0f;
    }

    // ── Layer loop: 23 layers ─────────────────────────────────────────────
    // 23 swaps (odd): final output lands in buf_b
    {
      const float (*ping)[8] = buf_a;
      float       (*pong)[8] = buf_b;

      for (int l = 0; l < 23; ++l)
      {
        Layer8_process_block(layers[l], ping, pong, input, head_acc, N);
        float (*tmp)[8] = pong; pong = (float(*)[8])ping; ping = tmp;
      }
      (void)ping;
    }

    // ── Head: 16-tap causal FIR across 8 channels over time ───────────────
    for (int n = 0; n < N; ++n)
    {
      for (int c = 0; c < 8; ++c) head_hist[head_ptr][c] = head_acc[n][c];

      float y = head_b;
      for (int t = 0; t < 16; ++t)
      {
        const int    idx = (head_ptr + 1 + t) & (16 - 1);
        const float* h   = head_hist[idx];
        const float* hw  = head_w[t];
        for (int c = 0; c < 8; ++c)
          y += h[c] * hw[c];
      }
      head_ptr = (head_ptr + 1) & (16 - 1);

      output[n] = y * head_scale;
    }
  }

  // ─────────────────────────────────────────────────────────────────────────────
  // Layer8_process_block  (8-channel)
  //
  // For each sample n:
  //   1. Write ins[n] into circular state buffer.
  //   2. Conv: z[o] = conv_b[o] + mixin_w[o]*cond[n] + sum_t(W[t] @ state[ptr+offset[t]])
  //   3. a[o] = LeakyReLU(z[o])
  //   4. head[n][o] += a[o]
  //   5. outs[n][o] = ins[n][o] + b1x1[o] + W1x1 @ a
  // ─────────────────────────────────────────────────────────────────────────────
  void Layer8_process_block(Layer8& layer,
                            const float ins[][8], float outs[][8],
                            const float* cond,    float head[][8],
                            int N) noexcept
  {
    const int          K   = layer.k;
    const int          ss  = layer.state_size;
    float* const       st  = layer.state;
    const float* const wt  = layer.wt;
    int                ptr = layer.ptr;

    for (int n = 0; n < N; ++n)
    {
      // ── Write input into circular state ───────────────────────────────
      float* const cw = st + ptr * 8;
      for (int c = 0; c < 8; ++c) cw[c] = ins[n][c];

      // ── Conv: bias + mixin ───────────────────────────────────────────
      const float cn = cond[n];
      float z[8];
      for (int o = 0; o < 8; ++o)
        z[o] = layer.conv_b[o] + layer.mixin_w[o] * cn;

      // ── Dilated conv taps ─────────────────────────────────────────────
      for (int t = 0; t < K; ++t)
      {
        int sp = ptr + layer.offset[t];
        if (sp >= ss) sp -= ss;
        const float* s  = st + sp * 8;
        const float* wk = wt + t * 8 * 8;
        for (int i = 0; i < 8; ++i)
        {
          const float  si = s[i];
          const float* wr = wk + i * 8;
          for (int o = 0; o < 8; ++o)
            z[o] += si * wr[o];
        }
      }

      ++ptr;
      if (ptr == ss) ptr = 0;

      // ── Activation + head accumulation ───────────────────────────────
      float a[8];
      for (int c = 0; c < 8; ++c)
      {
        const float act = nam_leaky_relu(z[c]);
        a[c] = act;
        head[n][c] += act;
      }

      // ── 1×1 projection + residual ────────────────────────────────────
      for (int o = 0; o < 8; ++o)
      {
        float acc = ins[n][o] + layer.b1x1[o];
        const float* wr = layer.w1x1 + o * 8;
        for (int i = 0; i < 8; ++i)
          acc += a[i] * wr[i];
        outs[n][o] = acc;
      }
    }

    layer.ptr = ptr;
  }

};

} // namespace MicroNAM
