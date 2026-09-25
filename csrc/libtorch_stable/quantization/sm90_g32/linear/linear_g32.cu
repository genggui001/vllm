// SPDX-License-Identifier: Apache-2.0
// Native G32 execution based on vLLM's CUTLASS SM90 blockwise FP8 dispatch.
#include <climits>
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

namespace h20_linear_g32 {
using namespace cute;
using Tensor = torch::stable::Tensor;
using Type = torch::headeronly::ScalarType;


// Rebind only the schedule tag after the official builder has selected all
// layouts, TMA atoms, stages, and the cooperative TiledMma.
template<class Main> struct SmallMCollective;
template<int Stages, class Cluster, class Schedule, class... Rest>
struct SmallMCollective<cutlass::gemm::collective::CollectiveMma<
    cutlass::gemm::MainloopSm90TmaGmmaWarpSpecializedBlockwiseFP8<Stages,Cluster,Schedule>,Rest...>> {
  using type = cutlass::gemm::collective::CollectiveMma<
      cutlass::gemm::MainloopSm90TmaGmmaWarpSpecializedBlockwiseFP8<
          Stages,Cluster,cutlass::gemm::collective::Sm90G32SmallMPromotionSchedule>,Rest...>;
};

template<class Main> struct M1ResidentCollective;
template<int Stages, class Cluster, class Schedule, class... Rest>
struct M1ResidentCollective<cutlass::gemm::collective::CollectiveMma<
    cutlass::gemm::MainloopSm90TmaGmmaWarpSpecializedBlockwiseFP8<Stages,Cluster,Schedule>,Rest...>> {
  using type = cutlass::gemm::collective::CollectiveMma<
      cutlass::gemm::MainloopSm90TmaGmmaWarpSpecializedBlockwiseFP8<
          Stages,Cluster,cutlass::gemm::collective::Sm90G32M1ResidentSchedule>,Rest...>;
};

template<class D, int BN, int GK, int NT = 128, bool Swap = false, int CM = 0, bool SmallM = false, bool M1Resident = false>
struct Kernel {
  using F8 = cutlass::float_e4m3_t;
  using Tile = Shape<_128,Int<NT>,_128>;
  using Cluster = cute::conditional_t<CM == 2 || (CM == 0 && GK == 32 && NT == 128),
      Shape<_2,_1,_1>,Shape<_1,_1,_1>>;
  using Scale = cutlass::detail::Sm90BlockwiseScaleConfig<
      1,BN,GK,(GK == 32 || !Swap) ? GMMA::Major::MN : GMMA::Major::K,
      (GK == 32 || Swap) ? GMMA::Major::MN : GMMA::Major::K>;
  // The isolated G32 collective uses NT16 to identify transposed operands.
  // Enforce this relation for every instantiation, independent of scale layout.
  static_assert(GK != 32 || (Swap == (NT == 16)));
  using SA = decltype(Scale::deduce_layoutSFA());
  using SB = decltype(Scale::deduce_layoutSFB());
  using LayoutD = cute::conditional_t<Swap,cutlass::layout::ColumnMajor,cutlass::layout::RowMajor>;
  using Epi = typename cutlass::epilogue::collective::CollectiveBuilder<
      cutlass::arch::Sm90,cutlass::arch::OpClassTensorOp,Tile,Cluster,
      cutlass::epilogue::collective::EpilogueTileAuto,
      float,float,void,LayoutD,128/cutlass::sizeof_bits<D>::value,
      D,LayoutD,128/cutlass::sizeof_bits<D>::value,
      cutlass::epilogue::TmaWarpSpecializedCooperative,
      cutlass::epilogue::fusion::LinearCombination<D,float,void,float>
  >::CollectiveOp;
  using GenericMain = typename cutlass::gemm::collective::CollectiveBuilder<
      cutlass::arch::Sm90,cutlass::arch::OpClassTensorOp,
      F8,tuple<cutlass::layout::RowMajor,SA>,16,
      F8,tuple<cutlass::layout::ColumnMajor,SB>,16,
      float,Tile,Cluster,
      // This upstream blockwise builder only accepts AutoCarveout. Reserve
      // extra space for four K32 scale planes; assert the resulting stage count.
      cutlass::gemm::collective::StageCountAutoCarveout<
        (GK == 32 ? (NT == 16 ? 150000 : (NT == 64 ? 114688 : 81920)) : sizeof(typename Epi::SharedStorage))>,
      cutlass::gemm::KernelTmaWarpSpecializedCooperativeFP8BlockScaledAccum
  >::CollectiveOp;
  static_assert(!SmallM || (GK == 32 && NT == 16 && Swap));
  static_assert(!M1Resident || (!SmallM && GK==32 && NT==16 && Swap && CM==0));
  using PreviousMain = cute::conditional_t<SmallM,typename SmallMCollective<GenericMain>::type,GenericMain>;
  using Main = cute::conditional_t<M1Resident,typename M1ResidentCollective<GenericMain>::type,PreviousMain>;
  static_assert(GK != 32 || Main::DispatchPolicy::Stages == 4);
  using Impl = cutlass::gemm::kernel::GemmUniversal<Shape<int,int,int,int>,Main,Epi>;
  static_assert(!SmallM || (Impl::NumMMAThreads == 256 && Impl::MaxThreadsPerBlock == 384 &&
      Impl::IsMainloopAuxiliaryLoadNeeded && Main::NumProducerThreadEvents == 33),
      "small-M tag must retain the cooperative producer/consumer protocol");
  static_assert(!M1Resident || (Impl::NumMMAThreads==256 && Impl::MaxThreadsPerBlock==384 &&
      Impl::IsMainloopAuxiliaryLoadNeeded && Main::NumProducerThreadEvents==33 &&
      Impl::MmaRegisterRequirement==96 && Impl::LoadRegisterRequirement==40),
      "M1 resident tag must preserve the cooperative auxiliary protocol");
  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<Impl>;
};

template<class D, int BN, int GK, int NT = 128, bool Swap = false, int CM = 0, bool SmallM = false, bool M1Resident = false>
void run(Tensor const& a, Tensor const& w, Tensor const& sa, Tensor const& sw,
         Tensor& out, Tensor const& workspace, int parts = 1, int supplied_sms = 0) {
  using K = Kernel<D,BN,GK,NT,Swap,CM,SmallM,M1Resident>;
  using Gemm = typename K::Gemm;
  using Impl = typename K::Impl;
  int m=a.size(0),n=w.size(0),k=a.size(1);
  if constexpr (SmallM) STD_TORCH_CHECK(m>=2 && m<=8,"small-M specialization requires original M2..8");
  if constexpr (M1Resident) STD_TORCH_CHECK(m==1,"resident kernel requires M1");
  if constexpr (Swap) { int tmp=m; m=n; n=tmp; }
  int chunk=k/parts;
  auto shape=make_shape(m,n,chunk,parts);
  auto stride_a=cutlass::make_cute_packed_stride(typename Impl::StrideA{},make_shape(m,k,parts));
  auto stride_b=cutlass::make_cute_packed_stride(typename Impl::StrideB{},make_shape(n,k,parts));
  auto stride_d=cutlass::make_cute_packed_stride(typename Impl::StrideD{},make_shape(m,n,parts));
  if (parts>1) {
    get<2>(stride_a)=chunk;
    get<2>(stride_b)=chunk;
  }
  typename Impl::MainloopArguments main{};
  main.ptr_A=static_cast<typename K::F8 const*>((Swap ? w : a).data_ptr()); main.dA=stride_a;
  main.ptr_B=static_cast<typename K::F8 const*>((Swap ? a : w).data_ptr()); main.dB=stride_b;
  main.ptr_SFA=static_cast<float const*>((Swap ? sw : sa).data_ptr());
  main.ptr_SFB=static_cast<float const*>((Swap ? sa : sw).data_ptr());
  main.layout_SFA=K::Scale::tile_atom_to_shape_SFA(shape);
  main.layout_SFB=K::Scale::tile_atom_to_shape_SFB(shape);
  if constexpr (GK == 32) {
    get<1,1>(main.layout_SFA.stride())=int((Swap ? sw : sa).stride(1));
    get<1,1>(main.layout_SFB.stride())=int((Swap ? sa : sw).stride(1));
  }
  if (parts>1) {
    if constexpr (GK == 32) {
      // Both scales are column-major. L advances by complete global K groups.
      get<2>(main.layout_SFA.stride())=int((Swap ? sw : sa).stride(1))*(chunk/GK);
      get<2>(main.layout_SFB.stride())=int((Swap ? sa : sw).stride(1))*(chunk/GK);
    } else {
      // Preserve the diagnostic G128 layout.
      if constexpr (Swap) {
        get<0,1>(main.layout_SFA.stride())=k/GK;
        get<2>(main.layout_SFA.stride())=chunk/GK;
        get<2>(main.layout_SFB.stride())=int((Swap ? sa : sw).stride(1))*(chunk/GK);
      } else {
        get<2>(main.layout_SFA.stride())=int((Swap ? sw : sa).stride(1))*(chunk/GK);
        get<0,1>(main.layout_SFB.stride())=k/GK;
        get<2>(main.layout_SFB.stride())=chunk/GK;
      }
    }
  }
  int64_t scratch_bytes=parts>1 ? ((int64_t(m)*n*parts*sizeof(D)+255)/256)*256 : 0;
  STD_TORCH_CHECK(scratch_bytes<=workspace.numel(),"split-K scratch exceeds workspace");
  auto workspace_ptr=static_cast<uint8_t*>(workspace.data_ptr());
  auto d=parts>1 ? reinterpret_cast<D*>(workspace_ptr) : static_cast<D*>(out.data_ptr());
  typename Impl::EpilogueArguments epi{{},d,stride_d,d,stride_d};
  cutlass::KernelHardwareInfo hw{};
  hw.device_id=a.get_device_index();
  hw.sm_count=supplied_sms;
  if (hw.sm_count<=0) cudaDeviceGetAttribute(&hw.sm_count,cudaDevAttrMultiProcessorCount,hw.device_id);
  if constexpr (M1Resident) {
    static_assert(cute::size(typename K::Cluster{})==1);
    int resident=Gemm::maximum_active_blocks();
    STD_TORCH_CHECK(resident>0,"cannot query resident cooperative CTA capacity");
    hw.sm_count *= resident;
  }
  typename Gemm::Arguments args{cutlass::gemm::GemmUniversalMode::kGemm,shape,main,epi,hw};
  Gemm gemm;
  STD_TORCH_CHECK(Gemm::get_workspace_size(args)<=workspace.numel()-scratch_bytes,"workspace too small");
  auto status=gemm.can_implement(args);
  STD_TORCH_CHECK(status==cutlass::Status::kSuccess,cutlassGetStatusString(status));
  auto stream=get_current_cuda_stream(a.get_device_index());
  status=gemm.initialize(args,workspace_ptr+scratch_bytes,stream);
  STD_TORCH_CHECK(status==cutlass::Status::kSuccess,cutlassGetStatusString(status),
                  "; dynamic_shared_bytes=",Impl::SharedStorageSize);
  status=gemm.run(stream);
  STD_TORCH_CHECK(status==cutlass::Status::kSuccess,cutlassGetStatusString(status));
}


void launch_static_reduce(float const*,void*,int64_t,int,bool,cudaStream_t);

// mode 0 is the automatic full-matrix path. mode 101 is the validated
// resident single-row kernel used by the narrow tail dispatcher.
void gemm(Tensor const& a, Tensor const& w, Tensor const& sa, Tensor const& sw,
          Tensor out, Tensor const& workspace, int64_t mode) {
  STD_TORCH_CHECK(a.is_cuda() && w.is_cuda() && sa.is_cuda() && sw.is_cuda() && out.is_cuda());
  STD_TORCH_CHECK(a.device()==w.device() && a.device()==sa.device() &&
                  a.device()==sw.device() && a.device()==out.device() && a.device()==workspace.device());
  STD_TORCH_CHECK(a.dim()==2 && w.dim()==2 && sa.dim()==2 && sw.dim()==2 && out.dim()==2);
  STD_TORCH_CHECK(a.scalar_type()==Type::Float8_e4m3fn && w.scalar_type()==Type::Float8_e4m3fn);
  STD_TORCH_CHECK(sa.scalar_type()==Type::Float && sw.scalar_type()==Type::Float);
  STD_TORCH_CHECK(a.is_contiguous() && w.is_contiguous() && out.is_contiguous());
  STD_TORCH_CHECK(workspace.scalar_type()==Type::Byte && workspace.is_contiguous());
  int64_t m=a.size(0),n=w.size(0),k=a.size(1);
  STD_TORCH_CHECK(m>0 && n>0 && n%8==0 && k>0 && k%32==0 && k==w.size(1));
  STD_TORCH_CHECK(m<=INT_MAX && n<=INT_MAX && k<=INT_MAX);
  STD_TORCH_CHECK(mode==0 || (mode==101 && m==1));
  STD_TORCH_CHECK(sa.size(0)==m && sa.size(1)==k/32 && sa.stride(0)==1 &&
                  sa.stride(1)>=m && sa.stride(1)<=INT_MAX/(k/32));
  STD_TORCH_CHECK(sw.size(0)==n && sw.size(1)==k/32 && sw.stride(0)==1 && sw.stride(1)==n);
  STD_TORCH_CHECK(out.size(0)==m && out.size(1)==n);
  STD_TORCH_CHECK(out.scalar_type()==Type::BFloat16 || out.scalar_type()==Type::Float);
  const torch::stable::accelerator::DeviceGuard guard(a.get_device_index());
  int split_parts=0;
  bool use_m1_resident=mode==101;
  if(mode==0 && m<=16 && k>=2048 && k%2048==0 && n<=2048)
    split_parts=n<=512 ? 16 : (n<=1024 ? 8 : 4);
  int sm_count=0;
  bool use_wide_tile=false;
  bool const wider_split_candidate=mode==0 && m<=16 && n>2048 && k>=2048 && k%2048==0;
  bool const medium_candidate=mode==0 && m>16 && m<2048;
  bool const resident_candidate=mode==0 && m==1 && n>1024 && k>=2048 && k%2048==0 && (k>=4096 || n>4992);
  if(wider_split_candidate || medium_candidate || resident_candidate) {
    STD_TORCH_CHECK(cudaDeviceGetAttribute(&sm_count,cudaDevAttrMultiProcessorCount,
                                         a.get_device_index())==cudaSuccess && sm_count>0);
  }
  if(wider_split_candidate) {
    int64_t tiles=(n+127)/128;
    if(4*tiles<=sm_count)split_parts=4;
    else if(2*tiles<=sm_count)split_parts=2;
  }
  if(resident_candidate) {
    using RB=Kernel<cutlass::bfloat16_t,1,32,16,true,0,false,true>;
    using RF=Kernel<float,1,32,16,true,0,false,true>;
    int rb=RB::Gemm::maximum_active_blocks(),rf=RF::Gemm::maximum_active_blocks();
    int resident=rb<rf ? rb : rf;
    int64_t tiles=(n+127)/128,slots=int64_t(sm_count)*resident;
    if(resident>0 && tiles<=slots) {
      int selected=1;
      for(int part : {16,8,4,2}) {
        if(tiles*part<=slots && k%(32*part)==0){selected=part;break;}
      }
      use_m1_resident=true;split_parts=selected>1 ? selected : 0;
    }
  }
  if(medium_candidate) {
    int64_t mt=(m+127)/128;
    int64_t waves64=(mt*((n+63)/64)+sm_count-1)/sm_count;
    int64_t waves128=(mt*((n+127)/128)+sm_count-1)/sm_count;
    use_wide_tile=2*waves128<=waves64;
  }
  if(split_parts>1) {
    STD_TORCH_CHECK(m<=16 && k%(32*split_parts)==0);
    if(use_m1_resident)run<float,1,32,16,true,0,false,true>(a,w,sa,sw,out,workspace,split_parts,sm_count);
    else if(m>=2 && m<=8)run<float,1,32,16,true,0,true>(a,w,sa,sw,out,workspace,split_parts,sm_count);
    else run<float,1,32,16,true>(a,w,sa,sw,out,workspace,split_parts,sm_count);
    launch_static_reduce(static_cast<float const*>(workspace.data_ptr()),out.data_ptr(),m*n,split_parts,
        out.scalar_type()==Type::BFloat16,get_current_cuda_stream(a.get_device_index()));
    STD_TORCH_CHECK(cudaGetLastError()==cudaSuccess,"G32 reduction launch failed");
    return;
  }
  if(out.scalar_type()==Type::BFloat16) {
    if(use_wide_tile)run<cutlass::bfloat16_t,1,32,128,false,1>(a,w,sa,sw,out,workspace,1,sm_count);
    else if(m<=16) {
      if(use_m1_resident)run<cutlass::bfloat16_t,1,32,16,true,0,false,true>(a,w,sa,sw,out,workspace,1,sm_count);
      else if(m>=2 && m<=8)run<cutlass::bfloat16_t,1,32,16,true,0,true>(a,w,sa,sw,out,workspace,1,sm_count);
      else run<cutlass::bfloat16_t,1,32,16,true>(a,w,sa,sw,out,workspace,1,sm_count);
    } else if(m>=2048 && n>=512)run<cutlass::bfloat16_t,1,32,128>(a,w,sa,sw,out,workspace);
    else run<cutlass::bfloat16_t,1,32,64>(a,w,sa,sw,out,workspace);
  } else {
    if(use_wide_tile)run<float,1,32,128,false,1>(a,w,sa,sw,out,workspace,1,sm_count);
    else if(m<=16) {
      if(use_m1_resident)run<float,1,32,16,true,0,false,true>(a,w,sa,sw,out,workspace,1,sm_count);
      else if(m>=2 && m<=8)run<float,1,32,16,true,0,true>(a,w,sa,sw,out,workspace,1,sm_count);
      else run<float,1,32,16,true>(a,w,sa,sw,out,workspace,1,sm_count);
    } else if(m>=2048 && n>=512)run<float,1,32,128>(a,w,sa,sw,out,workspace);
    else run<float,1,32,64>(a,w,sa,sw,out,workspace);
  }
}
}  // namespace h20_linear_g32
STABLE_TORCH_LIBRARY(h20_linear_g32,m) {
  m.def("gemm(Tensor a, Tensor w, Tensor sa, Tensor sw, Tensor(a!) out, Tensor(b!) workspace, int mode=0) -> ()");
}
STABLE_TORCH_LIBRARY_IMPL(h20_linear_g32,CUDA,m) {
  m.impl("gemm",TORCH_BOX(&h20_linear_g32::gemm));
}
