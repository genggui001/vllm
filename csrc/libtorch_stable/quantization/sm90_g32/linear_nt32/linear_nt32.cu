// SPDX-License-Identifier: Apache-2.0
// Native G32 Linear with 32-row output tiles, measured on H20.
#include <torch/csrc/stable/library.h>
#include <torch/csrc/stable/tensor.h>
#include "libtorch_stable/torch_utils.h"
#include "cutlass/cutlass.h"
#include "cute/tensor.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/util/packed_stride.hpp"

namespace h20_linear_nt32_g32 {
using namespace cute;
using Tensor=torch::stable::Tensor;
using Type=torch::headeronly::ScalarType;
template<class Main,bool Resident> struct SwappedCollective;
template<int Stages,class Cluster,class Schedule,bool Resident,class...Rest>
struct SwappedCollective<cutlass::gemm::collective::CollectiveMma<
    cutlass::gemm::MainloopSm90TmaGmmaWarpSpecializedBlockwiseFP8<Stages,Cluster,Schedule>,Rest...>,Resident> {
  using type=cutlass::gemm::collective::CollectiveMma<
      cutlass::gemm::MainloopSm90TmaGmmaWarpSpecializedBlockwiseFP8<Stages,Cluster,
          cute::conditional_t<Resident,cutlass::gemm::collective::Sm90G32SwapN32ResidentSchedule,cutlass::gemm::collective::Sm90G32SwapN32Schedule>>,Rest...>;
};
template<class D,bool Resident> struct Kernel {
  using F8=cutlass::float_e4m3_t;
  using Tile=Shape<_128,_32,_128>;
  using Cluster=Shape<_1,_1,_1>;
  using Scale=cutlass::detail::Sm90BlockwiseScaleConfig<1,1,32,GMMA::Major::MN,GMMA::Major::MN>;
  using SA=decltype(Scale::deduce_layoutSFA());
  using SB=decltype(Scale::deduce_layoutSFB());
  using LayoutD=cutlass::layout::ColumnMajor;
  using Epi=typename cutlass::epilogue::collective::CollectiveBuilder<
      cutlass::arch::Sm90,cutlass::arch::OpClassTensorOp,Tile,Cluster,
      cutlass::epilogue::collective::EpilogueTileAuto,
      float,float,void,LayoutD,128/cutlass::sizeof_bits<D>::value,
      D,LayoutD,128/cutlass::sizeof_bits<D>::value,
      cutlass::epilogue::TmaWarpSpecializedCooperative,
      cutlass::epilogue::fusion::LinearCombination<D,float,void,float>>::CollectiveOp;
  using GenericMain=typename cutlass::gemm::collective::CollectiveBuilder<
      cutlass::arch::Sm90,cutlass::arch::OpClassTensorOp,
      F8,tuple<cutlass::layout::RowMajor,SA>,16,
      F8,tuple<cutlass::layout::ColumnMajor,SB>,16,
      float,Tile,Cluster,cutlass::gemm::collective::StageCountAutoCarveout<140000>,
      cutlass::gemm::KernelTmaWarpSpecializedCooperativeFP8BlockScaledAccum>::CollectiveOp;
  using Main=typename SwappedCollective<GenericMain,Resident>::type;
  static_assert(Main::DispatchPolicy::Stages==4);
  using Impl=cutlass::gemm::kernel::GemmUniversal<Shape<int,int,int,int>,Main,Epi>;
  using Gemm=cutlass::gemm::device::GemmUniversalAdapter<Impl>;
  static_assert(Impl::NumMMAThreads==256 && Impl::MaxThreadsPerBlock==384 && Main::NumProducerThreadEvents==33);
  static_assert(!Resident || (Impl::MmaRegisterRequirement==96 && Impl::LoadRegisterRequirement==40 && Impl::MinBlocksPerMultiprocessor==2));
};
template<class D,bool Resident>
void run(Tensor const&a,Tensor const&w,Tensor const&sa,Tensor const&sw,Tensor&out,Tensor const&workspace) {
  using K=Kernel<D,Resident>;using Impl=typename K::Impl;using Gemm=typename K::Gemm;
  int m=w.size(0),n=a.size(0),k=a.size(1);auto shape=make_shape(m,n,k,1);
  typename Impl::MainloopArguments main{};
  main.ptr_A=static_cast<typename K::F8 const*>(w.data_ptr());
  main.ptr_B=static_cast<typename K::F8 const*>(a.data_ptr());
  main.dA=cutlass::make_cute_packed_stride(typename Impl::StrideA{},make_shape(m,k,1));
  main.dB=cutlass::make_cute_packed_stride(typename Impl::StrideB{},make_shape(n,k,1));
  main.ptr_SFA=static_cast<float const*>(sw.data_ptr());main.ptr_SFB=static_cast<float const*>(sa.data_ptr());
  main.layout_SFA=K::Scale::tile_atom_to_shape_SFA(shape);main.layout_SFB=K::Scale::tile_atom_to_shape_SFB(shape);
  // Original FP32 scales; a row view retains its original leading stride.
  get<1,1>(main.layout_SFA.stride())=int(sw.stride(1));
  get<1,1>(main.layout_SFB.stride())=int(sa.stride(1));
  auto ds=cutlass::make_cute_packed_stride(typename Impl::StrideD{},make_shape(m,n,1));
  auto ptr=static_cast<D*>(out.data_ptr());typename Impl::EpilogueArguments epi{{},ptr,ds,ptr,ds};
  cutlass::KernelHardwareInfo hw{};hw.device_id=a.get_device_index();
  STD_TORCH_CHECK(cudaDeviceGetAttribute(&hw.sm_count,cudaDevAttrMultiProcessorCount,hw.device_id)==cudaSuccess);
  if constexpr(Resident) {int capacity=Gemm::maximum_active_blocks();STD_TORCH_CHECK(capacity>0);hw.sm_count*=capacity;}
  typename Gemm::Arguments args{cutlass::gemm::GemmUniversalMode::kGemm,shape,main,epi,hw};
  STD_TORCH_CHECK(Gemm::get_workspace_size(args)<=workspace.numel());
  Gemm gemm;auto status=gemm.can_implement(args);STD_TORCH_CHECK(status==cutlass::Status::kSuccess,cutlassGetStatusString(status));
  auto stream=get_current_cuda_stream(a.get_device_index());status=gemm.initialize(args,workspace.data_ptr(),stream);
  STD_TORCH_CHECK(status==cutlass::Status::kSuccess,cutlassGetStatusString(status));status=gemm.run(stream);
  STD_TORCH_CHECK(status==cutlass::Status::kSuccess,cutlassGetStatusString(status));
}
void gemm(Tensor const&a,Tensor const&w,Tensor const&sa,Tensor const&sw,Tensor out,Tensor const&workspace,bool resident) {
  STD_TORCH_CHECK(a.is_cuda() && w.is_cuda() && sa.is_cuda() && sw.is_cuda() && out.is_cuda() && workspace.is_cuda());
  STD_TORCH_CHECK(a.device()==w.device() && a.device()==sa.device() && a.device()==sw.device() && a.device()==out.device() && a.device()==workspace.device());
  STD_TORCH_CHECK(a.dim()==2 && w.dim()==2 && sa.dim()==2 && sw.dim()==2 && out.dim()==2);
  STD_TORCH_CHECK(a.scalar_type()==Type::Float8_e4m3fn && w.scalar_type()==Type::Float8_e4m3fn && sa.scalar_type()==Type::Float && sw.scalar_type()==Type::Float);
  auto m=a.size(0),n=w.size(0),k=a.size(1);
  STD_TORCH_CHECK(m>16 && m<=2147483647LL && n>0 && n<=2147483647LL && k<=2147483647LL && n%8==0 && k>0 && k%32==0 && k==w.size(1));
  STD_TORCH_CHECK(a.is_contiguous() && w.is_contiguous() && out.is_contiguous() && workspace.is_contiguous() && workspace.scalar_type()==Type::Byte);
  STD_TORCH_CHECK(sa.size(0)==m && sa.size(1)==k/32 && sa.stride(0)==1 && sa.stride(1)>=m && sa.stride(1)<=2147483647LL);
  STD_TORCH_CHECK(sw.size(0)==n && sw.size(1)==k/32 && sw.stride(0)==1 && sw.stride(1)==n);
  STD_TORCH_CHECK(out.size(0)==m && out.size(1)==n && (out.scalar_type()==Type::BFloat16 || out.scalar_type()==Type::Float));
  const torch::stable::accelerator::DeviceGuard guard(a.get_device_index());
  if(out.scalar_type()==Type::BFloat16) {
    if(resident)run<cutlass::bfloat16_t,true>(a,w,sa,sw,out,workspace);else run<cutlass::bfloat16_t,false>(a,w,sa,sw,out,workspace);
  } else {
    if(resident)run<float,true>(a,w,sa,sw,out,workspace);else run<float,false>(a,w,sa,sw,out,workspace);
  }
}
}  // namespace h20_linear_nt32_g32
STABLE_TORCH_LIBRARY(h20_linear_nt32_g32,m) {m.def("gemm(Tensor a, Tensor w, Tensor sa, Tensor sw, Tensor(a!) out, Tensor(b!) workspace, bool resident) -> ()");}
STABLE_TORCH_LIBRARY_IMPL(h20_linear_nt32_g32,CUDA,m) {m.impl("gemm",TORCH_BOX(&h20_linear_nt32_g32::gemm));}
