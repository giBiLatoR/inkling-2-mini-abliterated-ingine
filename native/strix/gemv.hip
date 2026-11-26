#include <hip/hip_runtime.h>
#include <hip/hip_bfloat16.h>

template<class T> struct alignas(sizeof(T) * 4) Quad { T a, b, c, d; };

// One gfx1151 wave32 computes one weight row, loading four consecutive elements per lane.
template<class T>
__global__ void matvec(const Quad<T>* x, const Quad<T>* weights, T* y, int rows, int groups) {
    const int row = blockIdx.x * 4 + threadIdx.x / 32;
    if (row >= rows) return;
    const int lane = threadIdx.x % 32;
    const Quad<T>* w = weights + size_t(row) * groups;
    float a = 0, b = 0, c = 0, d = 0;
    for (int i = lane; i < groups; i += 32) {
        const auto u = x[i], v = w[i];
        a = fmaf(float(u.a), float(v.a), a);
        b = fmaf(float(u.b), float(v.b), b);
        c = fmaf(float(u.c), float(v.c), c);
        d = fmaf(float(u.d), float(v.d), d);
    }
    float sum = (a+b) + (c+d);
    for (int delta = 16; delta; delta >>= 1)
        sum += __shfl_down(sum, delta, 32);
    if (!lane) y[row] = T(sum);
}

template<class T>
static void launch(const void* x, const void* w, void* y, int rows, int cols, hipStream_t stream) {
    matvec<T><<<(rows+3)/4,128,0,stream>>>(static_cast<const Quad<T>*>(x),
        static_cast<const Quad<T>*>(w), static_cast<T*>(y), rows, cols/4);
}
extern "C" int strix_matvec(const void* x, const void* w, void* y, int rows, int cols, int fp32, hipStream_t stream) {
    if (fp32) launch<float>(x,w,y,rows,cols,stream);
    else launch<hip_bfloat16>(x,w,y,rows,cols,stream);
    return hipGetLastError();
}
