/***************************************************************************************************
 * Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 * list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 **************************************************************************************************/
// Modifications copyright contributors to the vLLM project.
#include <limits>
#include <tuple>
#include <torch/csrc/stable/library.h>
#include <torch/csrc/stable/tensor.h>
#include "cutlass/cutlass.h"
#include "cute/tensor.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/group_array_problem_shape.hpp"
#include "cutlass/util/mixed_dtype_utils.hpp"
#include "cutlass/util/packed_stride.hpp"
#include "libtorch_stable/torch_utils.h"
#include "libtorch_stable/cutlass_extensions/common.hpp"
#include "group32_wide_output_epilogue.hpp"
#include "group32_collective.hpp"
#include "group32_cooperative_kernel.hpp"

namespace h20_gemmfollowup_groupwise {
using namespace cute;
using TorchTensor = torch::stable::Tensor;
using Type = torch::headeronly::ScalarType;
using Fp8 = cutlass::float_e4m3_t;
using Int4 = cutlass::int4b_t;
using WeightStride = cute::remove_pointer_t<
    cutlass::detail::TagToStrideA_t<cutlass::layout::RowMajor*>>;
using ReorderAtom = decltype(cutlass::compute_memory_reordering_atom<Fp8>());
using WeightLayout = decltype(tile_to_shape(
    ReorderAtom{}, Layout<Shape<int,int,Int<1>>,WeightStride>{}));

// Same CUTLASS reorder device kernel, launched on the caller's stream.
template <class Src, class Dst>
void reorder_async(Src source, Dst dest, cudaStream_t stream) {
  using L = typename Dst::layout_type;
  auto has_major_mode = [](auto s) {
    return any_of(flatten(s), [](auto a) { return is_constant<1,decltype(a)>{}; });
  };
  auto values = conditional_return<has_major_mode(stride<0>(L{}))>(
      make_layout(make_shape(_2{},_1{}),GenColMajor{}),
      make_layout(make_shape(_1{},_2{}),GenRowMajor{}));
  auto threads = make_layout(make_shape(_1{},_128{}));
  auto copy_atom = make_tiled_copy(Copy_Atom<DefaultCopy,Int4>{},threads,values);
  using ReorderTile = Shape<_16>;
  auto tiled = group_modes<3,rank_v<L>>(tiled_divide(dest,ReorderTile{}));
  dim3 blocks{unsigned(size<1>(tiled)),1u,unsigned(size<3>(tiled))};
  cutlass::reorder_tensor_kernel<ReorderTile><<<blocks,128,0,stream>>>(
      source,dest,copy_atom);
}

std::tuple<TorchTensor,TorchTensor> pack(const TorchTensor& weights) {
  STD_TORCH_CHECK(weights.is_cuda() && weights.is_contiguous() &&
                  weights.dim()==3 && weights.scalar_type()==Type::Int);
  const torch::stable::accelerator::DeviceGuard guard(weights.device().index());
  auto stream=get_current_cuda_stream(weights.device().index());
  const int64_t e=weights.size(0),n=weights.size(1),k=weights.size(2)*8;
  STD_TORCH_CHECK(e>0 && e<=256 && n>0 && n%16==0 && k>0 && k%256==0);
  STD_TORCH_CHECK(n<=std::numeric_limits<int32_t>::max() &&
                  k<=std::numeric_limits<int32_t>::max(), "shape exceeds int32");
  auto packed=torch::stable::empty_like(weights);
  constexpr int width=sizeof(WeightLayout)/sizeof(int32_t);
  static_assert(sizeof(WeightLayout)%sizeof(int32_t)==0);
  auto cpu=torch::stable::empty({e,width},Type::Int,std::nullopt,
      torch::stable::Device(torch::stable::DeviceType::CPU));
  auto shape=make_shape(int(n),int(k),Int<1>{});
  auto src_layout=make_layout(shape,make_stride(int64_t(k),_1{},_0{}));
  WeightLayout dst_layout=tile_to_shape(ReorderAtom{},shape);
  for(int expert=0;expert<e;++expert) {
    const int64_t bytes=int64_t(expert)*n*k/2;
    auto src=reinterpret_cast<const Int4*>(
        static_cast<const char*>(weights.data_ptr())+bytes);
    auto dst=reinterpret_cast<Int4*>(static_cast<char*>(packed.data_ptr())+bytes);
    reorder_async(make_tensor(make_gmem_ptr<Int4>(src),src_layout),
                  make_tensor(make_gmem_ptr<Int4>(dst),dst_layout),stream);
    memcpy(cpu.mutable_data_ptr<int32_t>()+expert*width,&dst_layout,sizeof(dst_layout));
  }
  STD_TORCH_CHECK(cudaGetLastError()==cudaSuccess,"CUTLASS reorder launch failed");
  auto layouts=torch::stable::to(cpu,weights.device(),false);
  return {packed,layouts};
}

#ifndef H20_GROUP32_STAGES
#define H20_GROUP32_STAGES 4
#endif
#ifndef H20_GROUP32_CHANNELS
#define H20_GROUP32_CHANNELS 64
#endif
#ifndef H20_GROUP32_SMS_MULT
#define H20_GROUP32_SMS_MULT 2
#endif

template <class T, int Tokens> struct WideOutputAdapter;
template <class T, int Tokens>
struct WideOutputAdapter<cutlass::epilogue::collective::detail::Sm90TmaWarpSpecializedAdapter<T>, Tokens> {
  using Type = cutlass::epilogue::collective::MacroGroup32WideOutputAdapter<
      cutlass::epilogue::collective::MacroGroup32WideOutputEpilogue<
          typename T::ElementC, typename T::StrideC, typename T::StrideD,
          typename T::ThreadEpilogueOp, typename T::EpilogueSchedule, Tokens>>;
};

struct Prepared { TorchTensor ptrs,strides,problems,starts; };

template <int Tokens,class Output,bool WideOutput = false>
struct Operation {
  using Problem = cutlass::gemm::GroupProblemShape<Shape<int,int,int>>;
  using Tile = Shape<Int<H20_GROUP32_CHANNELS>,Int<Tokens>,_128>;
  using Cluster = Shape<_2,_1,_1>;
  using EpilogueSchedule = cutlass::epilogue::PtrArrayNoSmemWarpSpecialized;
  using MainloopSchedule = cute::conditional_t<H20_GROUP32_CHANNELS == 128,
      cutlass::gemm::KernelPtrArrayTmaWarpSpecializedCooperative,
      cutlass::gemm::KernelPtrArrayTmaWarpSpecializedPingpong>;
  using PlainEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
      cutlass::arch::Sm90,cutlass::arch::OpClassTensorOp,Tile,Cluster,
      cutlass::epilogue::collective::EpilogueTileAuto,float,float,
      void,cutlass::layout::ColumnMajor*,128/cutlass::sizeof_bits<Output>::value,
      Output,cutlass::layout::ColumnMajor*,128/cutlass::sizeof_bits<Output>::value,
      EpilogueSchedule>::CollectiveOp;
  using Epilogue = cute::conditional_t<
      WideOutput && cute::is_same_v<Output, cutlass::bfloat16_t>,
      typename WideOutputAdapter<PlainEpilogue, Tokens>::Type, PlainEpilogue>;
  using Base = typename cutlass::gemm::collective::CollectiveBuilder<
      cutlass::arch::Sm90,cutlass::arch::OpClassTensorOp,
      cute::tuple<Int4>,WeightLayout*,32,
      Fp8,cutlass::layout::ColumnMajor*,16,float,Tile,Cluster,
      cutlass::gemm::collective::StageCount<H20_GROUP32_STAGES>,
      MainloopSchedule>::CollectiveOp;
  using Mainloop=MacroGroup32Mainloop<Base>;
  using Kernel=cutlass::gemm::kernel::MacroGroup32CooperativeKernel<Problem,Mainloop,Epilogue,void>;
  using Gemm=cutlass::gemm::device::GemmUniversalAdapter<Kernel>;
  using BStride=typename Kernel::InternalStrideB;
  using DStride=typename Kernel::InternalStrideD;

  static void run(const TorchTensor& a,const TorchTensor& weight,
                  const TorchTensor& layouts,const TorchTensor& as,
                  const TorchTensor& ws,const TorchTensor& offsets,
                  TorchTensor& output,int ag,int wg,bool group_major,cudaStream_t stream,const Prepared* prepared=nullptr,int segment=0);
};

template <class BStride,class DStride,class Output>
__global__ void prepare(const Fp8* a,const Int4* weights,Output* out,
                        const int* offsets,int e,int rows,int n,int k,
                        const Int4** wa,const Fp8** ab,Output** od,
                        BStride* bs,DStride* ds,int* problems) {
  const int expert=threadIdx.x;
  if(expert>=e)return;
  const int begin=max(0,min(rows,offsets[expert]));
  const int end=max(begin,min(rows,offsets[expert+1]));
  wa[expert]=reinterpret_cast<const Int4*>(
      reinterpret_cast<const char*>(weights)+int64_t(expert)*n*k/2);
  ab[expert]=a+int64_t(begin)*k;
  od[expert]=out+int64_t(begin)*n;
  bs[expert]=make_stride(int64_t(k),_1{},_0{});
  ds[expert]=make_stride(_1{},int64_t(n),_0{});
  problems[expert*3]=n;
  problems[expert*3+1]=end-begin;
  problems[expert*3+2]=k;
}

template <int Tokens,class Output,bool WideOutput>
void Operation<Tokens,Output,WideOutput>::run(
    const TorchTensor& a,const TorchTensor& weight,const TorchTensor& layouts,
    const TorchTensor& as,const TorchTensor& ws,const TorchTensor& offsets,
    TorchTensor& output,int ag,int wg,bool group_major,cudaStream_t stream,const Prepared* prepared,int segment) {
  const int e=weight.size(0),n=output.size(1),k=a.size(1),r=a.size(0);
  auto device=a.device();
  auto ptrs=prepared?prepared->ptrs:torch::stable::empty({3,e},Type::Long,std::nullopt,device);
  auto strides=prepared?prepared->strides:torch::stable::empty({2,e},Type::Long,std::nullopt,device);
  auto problems=prepared?prepared->problems:torch::stable::empty({e,3},Type::Int,std::nullopt,device);
  static_assert(sizeof(BStride)==8 && sizeof(DStride)==8);
  auto wa=reinterpret_cast<const Int4**>(ptrs.data_ptr())+(prepared?segment*3*e:0);
  auto ab=reinterpret_cast<const Fp8**>(ptrs.data_ptr())+e+(prepared?segment*3*e:0);
  auto od=reinterpret_cast<Output**>(ptrs.data_ptr())+2*e+(prepared?segment*3*e:0);
  auto bs=reinterpret_cast<BStride*>(strides.data_ptr())+(prepared?segment*2*e:0);
  auto ds=reinterpret_cast<DStride*>(strides.data_ptr())+e+(prepared?segment*2*e:0);
  auto* problem_ptr=static_cast<int*>(problems.data_ptr())+(prepared?segment*3*e:0);
  auto* current_offsets=prepared?static_cast<const int*>(prepared->starts.data_ptr())+segment*(e+1):static_cast<const int*>(offsets.data_ptr());
  if(!prepared) prepare<<<1,256,0,stream>>>(static_cast<const Fp8*>(a.data_ptr()),
      static_cast<const Int4*>(weight.data_ptr()),static_cast<Output*>(output.data_ptr()),
      static_cast<const int*>(offsets.data_ptr()),e,r,n,k,wa,ab,od,bs,ds,
      static_cast<int*>(problems.data_ptr()));
  typename Mainloop::Arguments mainloop;
  mainloop.ptr_A=wa;
  mainloop.dA=static_cast<WeightLayout*>(layouts.data_ptr());
  mainloop.ptr_B=ab;
  mainloop.dB=bs;
  mainloop.group_scales={static_cast<const float*>(as.data_ptr()),
      static_cast<const cutlass::bfloat16_t*>(ws.data_ptr()),current_offsets,
      n,r,k/ag,k/wg,ag==32,wg==32?0:2,group_major};
  mainloop.group_scales.segment_enabled=prepared && (segment>0 || Tokens==128) ? current_offsets+e : nullptr;
  uint64_t dims[3]={uint64_t(n),uint64_t(k/32),uint64_t(e)};
  uint64_t strides_scale[2]={uint64_t(n)*2,uint64_t(n)*(k/32)*2};
  uint32_t box[3]={64,4,1}, elem_stride[3]={1,1,1};
  auto status=cuTensorMapEncodeTiled(&mainloop.group_scales.weight_map,
      CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,3,ws.data_ptr(),dims,strides_scale,
      box,elem_stride,CU_TENSOR_MAP_INTERLEAVE_NONE,CU_TENSOR_MAP_SWIZZLE_NONE,
      CU_TENSOR_MAP_L2_PROMOTION_NONE,CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  STD_TORCH_CHECK(status==CUDA_SUCCESS,"weight scale TMA encoding failed");
  mainloop.group_scales.activation_tma =
      (reinterpret_cast<uintptr_t>(as.data_ptr()) & 15) == 0;
  if (mainloop.group_scales.activation_tma) {
    uint64_t adims[2]={uint64_t(k/32),uint64_t(r)};
    uint64_t astride[1]={uint64_t(k/32)*4};
    uint32_t abox[2]={4,Tokens}, aes[2]={1,1};
    auto ast=cuTensorMapEncodeTiled(&mainloop.group_scales.activation_map,
        CU_TENSOR_MAP_DATA_TYPE_FLOAT32,2,as.data_ptr(),adims,astride,
        abox,aes,CU_TENSOR_MAP_INTERLEAVE_NONE,CU_TENSOR_MAP_SWIZZLE_NONE,
        CU_TENSOR_MAP_L2_PROMOTION_NONE,CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    STD_TORCH_CHECK(ast==CUDA_SUCCESS,"activation scale TMA encoding failed");
  }
  typename Epilogue::Arguments epilogue{{1.f,0.f},nullptr,ds,od,ds};
  Problem shape{e,reinterpret_cast<typename Problem::UnderlyingProblemShape*>(problem_ptr),nullptr};
  cutlass::KernelHardwareInfo hardware{device.index(),
      H20_GROUP32_SMS_MULT * cutlass::KernelHardwareInfo::query_device_multiprocessor_count(device.index())};
  typename Gemm::Arguments args{cutlass::gemm::GemmUniversalMode::kGrouped,
      shape,mainloop,epilogue,hardware};
  auto workspace=torch::stable::empty(int64_t(Gemm::get_workspace_size(args)),
      Type::Byte,std::nullopt,device);
  Gemm gemm;
  CUTLASS_CHECK(gemm.can_implement(args));
  CUTLASS_CHECK(gemm.initialize(args,workspace.data_ptr(),stream));
  CUTLASS_CHECK(gemm.run(stream));
}


template<class BStride,class DStride,class Output>
__global__ void prepare_mixed(const Fp8* a,const Int4* weights,Output* out,
    const int* offsets,int e,int rows,int n,int k,void** ptrs,
    int64_t* strides,int* problems,int* starts) {
  const int expert=threadIdx.x;
  const int begin=expert<e?max(0,min(rows,offsets[expert])):0;
  const int end=expert<e?max(begin,min(rows,offsets[expert+1])):0;
  const int count=end-begin;
  __shared__ int waste[256];
  waste[expert]=((count+63)/64)*64-count;
  __syncthreads();
  for(int step=128;step;step/=2) {
    if(expert<step)waste[expert]+=waste[expert+step];
    __syncthreads();
  }
  const bool mixed=int64_t(waste[0])*100>=int64_t(rows)*8;
  if(expert>=e)return;
  for(int segment=0;segment<3;++segment) {
    int b=begin,z=end;
    if(!mixed) {if(segment!=0)b=z;}
    else if(segment==0)z=b+(count/64)*64;
    else if(segment==1) {b+=(count/64)*64;z=b+((count%64)/32)*32;}
    else b+=(count/32)*32;
    auto** wa=reinterpret_cast<const Int4**>(static_cast<void*>(ptrs+segment*3*e));
    auto** ab=reinterpret_cast<const Fp8**>(static_cast<void*>(ptrs+segment*3*e+e));
    auto** od=reinterpret_cast<Output**>(ptrs+segment*3*e+2*e);
    auto* bs=reinterpret_cast<BStride*>(strides+segment*2*e);
    auto* ds=reinterpret_cast<DStride*>(strides+segment*2*e+e);
    wa[expert]=reinterpret_cast<const Int4*>(reinterpret_cast<const char*>(weights)+int64_t(expert)*n*k/2);
    ab[expert]=a+int64_t(b)*k; od[expert]=out+int64_t(b)*n;
    bs[expert]=make_stride(int64_t(k),_1{},_0{});
    ds[expert]=make_stride(_1{},int64_t(n),_0{});
    int* shape=problems+segment*e*3+expert*3;
    shape[0]=n;shape[1]=z-b;shape[2]=k;
    starts[segment*(e+1)+expert]=b;
    if(expert==0)starts[segment*(e+1)+e]=mixed;
  }
}

template<class Output,bool Wide64>
void run_mixed(const TorchTensor& a,const TorchTensor& weight,const TorchTensor& layouts,
    const TorchTensor& as,const TorchTensor& ws,const TorchTensor& offsets,
    TorchTensor& output,int ag,int wg,bool group_major,cudaStream_t stream) {
  const int e=weight.size(0),n=output.size(1),k=a.size(1),r=a.size(0);
  auto device=a.device();
  Prepared p{torch::stable::empty({9,e},Type::Long,std::nullopt,device),
      torch::stable::empty({6,e},Type::Long,std::nullopt,device),
      torch::stable::empty({3,e,3},Type::Int,std::nullopt,device),
      torch::stable::empty({3,e+1},Type::Int,std::nullopt,device)};
  using Op=Operation<64,Output,Wide64>;
  prepare_mixed<typename Op::BStride,typename Op::DStride,Output><<<1,256,0,stream>>>(
      static_cast<const Fp8*>(a.data_ptr()),static_cast<const Int4*>(weight.data_ptr()),
      static_cast<Output*>(output.data_ptr()),static_cast<const int*>(offsets.data_ptr()),
      e,r,n,k,reinterpret_cast<void**>(p.ptrs.data_ptr()),
      static_cast<int64_t*>(p.strides.data_ptr()),static_cast<int*>(p.problems.data_ptr()),
      static_cast<int*>(p.starts.data_ptr()));
  Operation<64,Output,Wide64>::run(a,weight,layouts,as,ws,offsets,output,ag,wg,group_major,stream,&p,0);
  Operation<32,Output,true>::run(a,weight,layouts,as,ws,offsets,output,ag,wg,group_major,stream,&p,1);
  Operation<16,Output,true>::run(a,weight,layouts,as,ws,offsets,output,ag,wg,group_major,stream,&p,2);
}

template<class BStride,class DStride,class Output>
__global__ void prepare_macro_mixed(const Fp8* a,const Int4* weights,Output* out,
    const int* offsets,int e,int rows,int n,int k,void** ptrs,
    int64_t* strides,int* problems,int* starts) {
  const int expert=threadIdx.x;
  const int begin=expert<e?max(0,min(rows,offsets[expert])):0;
  const int end=expert<e?max(begin,min(rows,offsets[expert+1])):0;
  const int count=end-begin;
  const int lengths[4]={(count/128)*128,((count%128)/64)*64,
                        ((count%64)/32)*32,count%32};
  __shared__ int enabled[256];
  int mask=0;
  for(int segment=0;segment<4;++segment)mask|=(lengths[segment]>0)<<segment;
  enabled[expert]=mask;
  __syncthreads();
  for(int step=128;step;step/=2) {
    if(expert<step)enabled[expert]|=enabled[expert+step];
    __syncthreads();
  }
  if(expert==0)for(int segment=0;segment<4;++segment)
    starts[segment*(e+1)+e]=(enabled[0]>>segment)&1;
  if(expert>=e)return;
  int b=begin;
  for(int segment=0;segment<4;++segment) {
    const int z=b+lengths[segment];
    auto** wa=reinterpret_cast<const Int4**>(static_cast<void*>(ptrs+segment*3*e));
    auto** ab=reinterpret_cast<const Fp8**>(static_cast<void*>(ptrs+segment*3*e+e));
    auto** od=reinterpret_cast<Output**>(ptrs+segment*3*e+2*e);
    auto* bs=reinterpret_cast<BStride*>(strides+segment*2*e);
    auto* ds=reinterpret_cast<DStride*>(strides+segment*2*e+e);
    wa[expert]=reinterpret_cast<const Int4*>(reinterpret_cast<const char*>(weights)+int64_t(expert)*n*k/2);
    ab[expert]=a+int64_t(b)*k;od[expert]=out+int64_t(b)*n;
    bs[expert]=make_stride(int64_t(k),_1{},_0{});
    ds[expert]=make_stride(_1{},int64_t(n),_0{});
    int* shape=problems+segment*e*3+expert*3;
    shape[0]=n;shape[1]=z-b;shape[2]=k;
    starts[segment*(e+1)+expert]=b;
    b=z;
  }
}

template<class Output,bool WideOutput=false>
void run_macro_mixed(const TorchTensor& a,const TorchTensor& weight,const TorchTensor& layouts,
    const TorchTensor& as,const TorchTensor& ws,const TorchTensor& offsets,
    TorchTensor& output,int ag,int wg,bool group_major,cudaStream_t stream) {
  const int e=weight.size(0),n=output.size(1),k=a.size(1),r=a.size(0);
  auto device=a.device();
  Prepared p{torch::stable::empty({12,e},Type::Long,std::nullopt,device),
      torch::stable::empty({8,e},Type::Long,std::nullopt,device),
      torch::stable::empty({4,e,3},Type::Int,std::nullopt,device),
      torch::stable::empty({4,e+1},Type::Int,std::nullopt,device)};
  using Op=Operation<128,Output>;
  prepare_macro_mixed<typename Op::BStride,typename Op::DStride,Output><<<1,256,0,stream>>>(
      static_cast<const Fp8*>(a.data_ptr()),static_cast<const Int4*>(weight.data_ptr()),
      static_cast<Output*>(output.data_ptr()),static_cast<const int*>(offsets.data_ptr()),
      e,r,n,k,reinterpret_cast<void**>(p.ptrs.data_ptr()),
      static_cast<int64_t*>(p.strides.data_ptr()),static_cast<int*>(p.problems.data_ptr()),
      static_cast<int*>(p.starts.data_ptr()));
  Operation<128,Output,WideOutput>::run(a,weight,layouts,as,ws,offsets,output,ag,wg,group_major,stream,&p,0);
  Operation<64,Output,WideOutput>::run(a,weight,layouts,as,ws,offsets,output,ag,wg,group_major,stream,&p,1);
  Operation<32,Output,true>::run(a,weight,layouts,as,ws,offsets,output,ag,wg,group_major,stream,&p,2);
  Operation<16,Output,true>::run(a,weight,layouts,as,ws,offsets,output,ag,wg,group_major,stream,&p,3);
}

void mm(const TorchTensor& a,const TorchTensor& weights,const TorchTensor& layouts,
        const TorchTensor& as,const TorchTensor& ws,const TorchTensor& offsets,
        TorchTensor& output,int64_t ag,int64_t wg,int64_t tile,bool group_major) {
  STD_TORCH_CHECK(a.is_cuda() && a.is_contiguous() && a.dim()==2 &&
      a.scalar_type()==Type::Float8_e4m3fn);
  const torch::stable::accelerator::DeviceGuard guard(a.device().index());
  const int64_t r=a.size(0),k=a.size(1);
  auto require=[&](const TorchTensor& t,Type dtype){
    STD_TORCH_CHECK(t.is_cuda() && t.device()==a.device() && t.is_contiguous() && t.scalar_type()==dtype);
  };
  require(weights,Type::Int);require(layouts,Type::Int);
  require(as,Type::Float);require(ws,Type::BFloat16);require(offsets,Type::Int);
  STD_TORCH_CHECK(output.scalar_type()==Type::BFloat16 || output.scalar_type()==Type::Float);
  require(output,output.scalar_type());
  STD_TORCH_CHECK(weights.dim()==3 && output.dim()==2 && output.size(0)==r);
  int64_t e=weights.size(0),n=weights.size(1);
  STD_TORCH_CHECK(e>0 && e<=256 && n>0 && n%16==0 && k>0 && k%256==0);
  STD_TORCH_CHECK(r<=std::numeric_limits<int32_t>::max() &&
      n<=std::numeric_limits<int32_t>::max() &&
      k<=std::numeric_limits<int32_t>::max(), "shape exceeds int32");
  STD_TORCH_CHECK(weights.size(2)==k/8 && output.size(1)==n);
  STD_TORCH_CHECK(ag==32 && wg==32 && group_major);
  STD_TORCH_CHECK(as.dim()==2 && as.size(0)==r && as.size(1)==k/ag);
  STD_TORCH_CHECK(ws.dim()==3 && ws.size(0)==e &&
      ws.size(group_major?2:1)==n && ws.size(group_major?1:2)==k/wg);
  STD_TORCH_CHECK(offsets.dim()==1 && offsets.numel()==e+1);
  STD_TORCH_CHECK(layouts.dim()==2 && layouts.size(0)==e &&
      layouts.size(1)==sizeof(WeightLayout)/sizeof(int32_t));
  STD_TORCH_CHECK(tile==32 || tile==64 || tile==128);
  for (const auto* tensor : {&a, &weights, static_cast<const TorchTensor*>(&output)}) {
    STD_TORCH_CHECK((reinterpret_cast<uintptr_t>(tensor->data_ptr()) & 15) == 0,
                    "CUTLASS TMA requires 16-byte alignment");
  }
  const auto out_begin = reinterpret_cast<uintptr_t>(output.data_ptr());
  const auto out_end = out_begin + output.numel() * output.element_size();
  for (const auto* tensor : {&a, &weights, &layouts, &as, &ws, &offsets}) {
    const auto begin = reinterpret_cast<uintptr_t>(tensor->data_ptr());
    const auto end = begin + tensor->numel() * tensor->element_size();
    STD_TORCH_CHECK(out_begin >= end || begin >= out_end,
                    "GEMM output must not overlap inputs");
  }
  if(r==0)return;
  auto stream=get_current_cuda_stream(a.device().index());
  if(tile==128) {
    if(output.scalar_type()==Type::Float)Operation<128,float>::run(a,weights,layouts,as,ws,offsets,output,ag,wg,group_major,stream);
    else if(k==256 || k==2048)Operation<128,cutlass::bfloat16_t,true>::run(a,weights,layouts,as,ws,offsets,output,ag,wg,group_major,stream);
    else Operation<128,cutlass::bfloat16_t,false>::run(a,weights,layouts,as,ws,offsets,output,ag,wg,group_major,stream);
    return;
  }
  if(tile==64 && r>8192 && (k==2048 || (k==256 && r>=e*128))) {
    if(output.scalar_type()==Type::Float)run_macro_mixed<float>(a,weights,layouts,as,ws,offsets,output,ag,wg,group_major,stream);
    else if(k==256 || k==2048)run_macro_mixed<cutlass::bfloat16_t,true>(a,weights,layouts,as,ws,offsets,output,ag,wg,group_major,stream);
    else run_macro_mixed<cutlass::bfloat16_t>(a,weights,layouts,as,ws,offsets,output,ag,wg,group_major,stream);
    return;
  }
  if(tile==64 && r>8192) {
    if(output.scalar_type()==Type::Float)run_mixed<float,false>(a,weights,layouts,as,ws,offsets,output,ag,wg,group_major,stream);
    else if(k==256)run_mixed<cutlass::bfloat16_t,true>(a,weights,layouts,as,ws,offsets,output,ag,wg,group_major,stream);
    else run_mixed<cutlass::bfloat16_t,false>(a,weights,layouts,as,ws,offsets,output,ag,wg,group_major,stream);
    return;
  }
  // Short-K FC2 keeps the same arithmetic and writes contiguous BF16 vectors.
  if((k==256 || tile==32) && output.scalar_type()==Type::BFloat16) {
    if(tile==64)Operation<64,cutlass::bfloat16_t,true>::run(a,weights,layouts,as,ws,offsets,output,ag,wg,group_major,stream);
    else Operation<32,cutlass::bfloat16_t,true>::run(a,weights,layouts,as,ws,offsets,output,ag,wg,group_major,stream);
    return;
  }
  if(output.scalar_type()==Type::Float) {
    if(tile==64)Operation<64,float>::run(a,weights,layouts,as,ws,offsets,output,ag,wg,group_major,stream);
    else Operation<32,float>::run(a,weights,layouts,as,ws,offsets,output,ag,wg,group_major,stream);
  } else {
    if(tile==64)Operation<64,cutlass::bfloat16_t>::run(a,weights,layouts,as,ws,offsets,output,ag,wg,group_major,stream);
    else Operation<32,cutlass::bfloat16_t>::run(a,weights,layouts,as,ws,offsets,output,ag,wg,group_major,stream);
  }
}

STABLE_TORCH_LIBRARY_FRAGMENT(_moe_C_gemmfollowup,m) {
  m.def("h20_w4a8_cutlass_groupwise_pack(Tensor weights) -> (Tensor, Tensor)");
  m.def("h20_w4a8_cutlass_groupwise_mm(Tensor a, Tensor weights, Tensor layouts, Tensor a_scale, Tensor w_scale, Tensor offsets, Tensor(a!) output, int a_group, int w_group, int tile_tokens, bool weight_scale_group_major=False) -> ()");
}
STABLE_TORCH_LIBRARY_IMPL(_moe_C_gemmfollowup,CUDA,m) {
  m.impl("h20_w4a8_cutlass_groupwise_pack",TORCH_BOX(&pack));
  m.impl("h20_w4a8_cutlass_groupwise_mm",TORCH_BOX(&mm));
}
}  // namespace h20_gemmfollowup_groupwise
