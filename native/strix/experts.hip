#include <hip/hip_runtime.h>
#include <hip/hip_bfloat16.h>
#include <stdint.h>

// Decode two checkpoint codes per byte for E2M1, or one E4M3FN value per byte.
// One wave streams one row of one routed expert; inputs are shared across its lanes.
using B = hip_bfloat16;
__device__ float e4(unsigned char b) {
    unsigned m=b&127, e=m>>3;
    float v = e ? __uint_as_float(((e+120)<<23) | ((m&7)<<20)) : m*(1.f/512.f);
    if (m==127) v=NAN;
    return (b&128) ? -v : v;
}
__device__ float e2(unsigned c) {
    unsigned m=c&7;
    float v=m<5 ? .5f*m : (m==5 ? 3.f : (m==6 ? 4.f : 6.f));
    return (c&8) ? -v : v;
}

template<bool FP4>
__global__ void expert_mv(const B* x, const int64_t* ids, const unsigned char* w, const unsigned char* scales,
                          const float* factors, B* out, int rows, int cols, int top, bool per_route) {
    int row=blockIdx.x*4+threadIdx.x/32, lane=threadIdx.x%32, route=blockIdx.y;
    if (row>=rows) return;
    int expert=ids[route];
    size_t r=size_t(expert)*rows+row;
    int bytes=FP4 ? cols/2 : cols;
    auto words=reinterpret_cast<const uint32_t*>(w+r*bytes);
    auto input=x+size_t(per_route ? route : route/top)*cols;
    auto s=scales+r*(FP4 ? cols/16 : cols/32);
    float factor=FP4 ? factors[expert] : 1.f;
    float sum=0.f;
    for (int i=lane;i<bytes/4;i+=32) {
        const uint32_t word=words[i];
        unsigned char block=s[FP4 ? i/2 : i/8];
        float scale=FP4 ? e4(block) : exp2f(float(block)-127.f);
        constexpr int N=FP4 ? 8 : 4;
        #pragma unroll
        for (int j=0;j<N;++j) {
            float value=FP4 ? e2(word>>(4*j)) : e4(word>>(8*j));
            float weight=float(B((value*scale)*factor));
            sum=fmaf(float(input[N*i+j]),weight,sum);
        }
    }
    for(int d=16;d;d/=2) sum+=__shfl_down(sum,d,32);
    if (!lane) out[size_t(route)*rows+row]=B(sum);
}

extern "C" int strix_experts(const void* x,const int64_t* ids,const unsigned char* w,const unsigned char* scale,
                             const float* factor,void* out,int rows,int cols,int top,int routes,int fp4,
                             int per_route,hipStream_t stream) {
    // grid.y is bounded by the caller when processing large prompts.
    dim3 grid((rows+3)/4,routes);
    #define RUN(F) expert_mv<F><<<grid,128,0,stream>>>((const B*)x,ids,w,scale,factor,(B*)out,rows,cols,top,per_route)
    if(fp4) { RUN(true); } else { RUN(false); }
    #undef RUN
    return hipGetLastError();
}

// The checkpoint interleaves gate/up rows and caps the two linear factors,
// while the sigmoid always uses the original (uncapped) gate.
__global__ void activate(const B* paired, B* out, size_t n, float gate_cap, float up_cap) {
    size_t i = size_t(blockIdx.x)*blockDim.x + threadIdx.x;
    if (i >= n) return;
    float gate = float(paired[2*i]), up = float(paired[2*i+1]);
    float value = (gate_cap*tanhf(gate/gate_cap)) * (1.f/(1.f+expf(-gate)));
    out[i] = B(value*(up_cap*tanhf(up/up_cap)));
}

__global__ void combine(const B* down, const float* weights, B* out, size_t n, int cols, int top) {
    size_t i = size_t(blockIdx.x)*blockDim.x + threadIdx.x;
    if (i >= n) return;
    size_t route = (i/cols)*top;
    float sum = 0.f;
    for (int j=0;j<top;++j)
        sum += float(down[(route+j)*cols+i%cols])*weights[route+j];
    out[i] = B(sum);
}

extern "C" int strix_activate(const B* in, B* out, size_t n, float gate_cap, float up_cap, hipStream_t stream) {
    activate<<<(n+255)/256,256,0,stream>>>(in,out,n,gate_cap,up_cap);
    return hipGetLastError();
}
extern "C" int strix_combine(const B* in,const float* weights,B* out,size_t n,int cols,int top,hipStream_t stream) {
    combine<<<(n+255)/256,256,0,stream>>>(in,weights,out,n,cols,top);
    return hipGetLastError();
}
