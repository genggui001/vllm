#include <cuda_runtime.h>
#include <cstdint>
#include <cutlass/bfloat16.h>
namespace h20_linear_g32 {
template<class D,int Parts>
__global__ void reduce_split_k_static(float const* partial,D* out,int64_t count) {
  int64_t i=int64_t(blockIdx.x)*blockDim.x+threadIdx.x;
  if(i<count) {
    float value=partial[i];
    #pragma unroll
    for(int p=1;p<Parts;++p) value=__fadd_rn(value,partial[int64_t(p)*count+i]);
    out[i]=D(value);
  }
}

template<class D>
void launch(float const* partial,D* out,int64_t count,int parts,cudaStream_t stream) {
  switch(parts) {
    case 2:reduce_split_k_static<D,2><<<(count+255)/256,256,0,stream>>>(partial,out,count);break;
    case 4:reduce_split_k_static<D,4><<<(count+255)/256,256,0,stream>>>(partial,out,count);break;
    case 8:reduce_split_k_static<D,8><<<(count+255)/256,256,0,stream>>>(partial,out,count);break;
    case 16:reduce_split_k_static<D,16><<<(count+255)/256,256,0,stream>>>(partial,out,count);break;
  }
}
void launch_static_reduce(float const* partial,void* out,int64_t count,int parts,bool bf16,cudaStream_t stream) {
  if(bf16) launch(partial,static_cast<cutlass::bfloat16_t*>(out),count,parts,stream);
  else launch(partial,static_cast<float*>(out),count,parts,stream);
}
}
