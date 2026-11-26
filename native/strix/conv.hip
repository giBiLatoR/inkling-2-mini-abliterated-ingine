#include <hip/hip_runtime.h>
#include <hip/hip_bfloat16.h>

// One channel per thread. The history holds four FP32 taps in oldest-first order.
// The upstream convolution rounds to the weight dtype before the FP32 residual.
template<class Weight, class Input, class State>
__global__ void conv4(const Input* input, State* state, const Weight* weight, Input* out, int channels) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= channels) return;
    State* history = state + 4*c;
    const Weight* w = weight + 4*c;
    const State a = history[1], b = history[2], d = history[3];
    const float x = float(input[c]);
    const float sum = float(Weight(float(a))) * float(w[0]) +
                      float(Weight(float(b))) * float(w[1]) +
                      float(Weight(float(d))) * float(w[2]) +
                      float(Weight(x)) * float(w[3]);
    history[0] = State(float(Weight(float(a))));
    history[1] = State(float(Weight(float(b))));
    history[2] = State(float(Weight(float(d))));
    history[3] = State(float(Weight(x)));
    out[c] = Input((float(Weight(sum)) + x) * 0.44721359549995794f);
}

template<class W, class X, class S>
static void launch(const void* x, void* state, const void* w, void* out, int n, hipStream_t stream) {
    conv4<W,X,S><<<(n+127)/128,128,0,stream>>>(static_cast<const X*>(x), static_cast<S*>(state),
                                                static_cast<const W*>(w), static_cast<X*>(out), n);
}
extern "C" int strix_conv(const void* x, void* state, const void* w, void* out, int n,
                          int xf, int sf, int wf, hipStream_t stream) {
    using B = hip_bfloat16;
    #define CALL(W,X) if (sf) launch<W,X,float>(x,state,w,out,n,stream); else launch<W,X,B>(x,state,w,out,n,stream)
    if (wf) { if (xf) {CALL(float,float);} else {CALL(float,B);} }
    else { if (xf) {CALL(B,float);} else {CALL(B,B);} }
    #undef CALL
    return hipGetLastError();
}
