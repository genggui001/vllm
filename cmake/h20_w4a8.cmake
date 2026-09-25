# H20 native W4A8 sources; include after _moe_C_stable_libtorch is defined.
if(VLLM_GPU_LANG STREQUAL "CUDA" AND CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL 12.3)
  cuda_archs_loose_intersection(H20_W4A8_ARCHS "9.0a" "${CUDA_ARCHS}")
  if(H20_W4A8_ARCHS)
    set(H20_W4A8_SRCS
      "${CMAKE_CURRENT_SOURCE_DIR}/csrc/libtorch_stable/moe/h20_w4a8/prepared_gemm.cu"
      "${CMAKE_CURRENT_SOURCE_DIR}/csrc/libtorch_stable/moe/h20_w4a8/pingpong_gemm.cu"
      "${CMAKE_CURRENT_SOURCE_DIR}/csrc/libtorch_stable/moe/h20_w4a8/narrow_prepared_gemm.cu"
      "${CMAKE_CURRENT_SOURCE_DIR}/csrc/libtorch_stable/moe/h20_w4a8/narrow_pingpong_gemm.cu"
      "${CMAKE_CURRENT_SOURCE_DIR}/csrc/libtorch_stable/moe/h20_w4a8/staged_gemm.cu"
      "${CMAKE_CURRENT_SOURCE_DIR}/csrc/libtorch_stable/moe/h20_w4a8/resource_gemm.cu"
      "${CMAKE_CURRENT_SOURCE_DIR}/csrc/libtorch_stable/moe/h20_w4a8/silu_fp8.cu"
      "${CMAKE_CURRENT_SOURCE_DIR}/csrc/libtorch_stable/moe/h20_w4a8/single_prepare.cu"
      "${CMAKE_CURRENT_SOURCE_DIR}/csrc/libtorch_stable/moe/h20_w4a8/finalize.cu"
      "${CMAKE_CURRENT_SOURCE_DIR}/csrc/libtorch_stable/moe/h20_w4a8/batch_prepare.cu"
      "${CMAKE_CURRENT_SOURCE_DIR}/csrc/libtorch_stable/moe/h20_w4a8/fused_fc2.cu"
      "${CMAKE_CURRENT_SOURCE_DIR}/csrc/libtorch_stable/moe/h20_w4a8/slot_group_prepare.cu"
    )
    list(APPEND H20_W4A8_SRCS
      "${CMAKE_CURRENT_SOURCE_DIR}/csrc/libtorch_stable/moe/h20_w4a8/topk.cu"
      "${CMAKE_CURRENT_SOURCE_DIR}/csrc/libtorch_stable/moe/h20_w4a8/prefetch_finalize.cu"
    )
    set(H20_G32_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/csrc/libtorch_stable/quantization/sm90_g32")
    set(H20_G32_SRCS
      "${H20_G32_ROOT}/linear/linear_g32.cu"
      "${H20_G32_ROOT}/linear/reduce_static.cu"
      "${H20_G32_ROOT}/linear_nt32/linear_nt32.cu"
      "${H20_G32_ROOT}/quant_g32.cu"
      "${H20_G32_ROOT}/moe_native.cu"
      "${H20_G32_ROOT}/moe_small/cutlass_groupwise_gemm.cu"
      "${H20_G32_ROOT}/moe_large/cutlass_groupwise_gemm.cu")
    list(APPEND H20_W4A8_SRCS ${H20_G32_SRCS})
    # Only the G32 Linear translation unit sees its specialized collective.
    set_property(SOURCE "${H20_G32_ROOT}/linear/linear_g32.cu" APPEND PROPERTY
      INCLUDE_DIRECTORIES "${H20_G32_ROOT}/linear/include")
    set_property(SOURCE "${H20_G32_ROOT}/linear_nt32/linear_nt32.cu" APPEND PROPERTY
      INCLUDE_DIRECTORIES "${H20_G32_ROOT}/linear_nt32/include")
    set_property(SOURCE "${H20_G32_ROOT}/moe_small/cutlass_groupwise_gemm.cu"
      "${H20_G32_ROOT}/moe_large/cutlass_groupwise_gemm.cu" APPEND PROPERTY COMPILE_DEFINITIONS
      H20_GROUP32_SCALE_PREFETCH=1 H20_GROUP32_OVERLAP=1 H20_GROUP32_STAGES=4
      H20_GROUP32_CHANNELS=64 H20_GROUP32_SMS_MULT=2 H20_GROUP32_MIN_BLOCKS=2
      H20_GROUP32_LOAD_REGS=64 H20_GROUP32_MMA_REGS=192)
    set_gencode_flags_for_srcs(SRCS "${H20_W4A8_SRCS}" CUDA_ARCHS "${H20_W4A8_ARCHS}")
    target_sources(_moe_C_stable_libtorch PRIVATE ${H20_W4A8_SRCS})
    target_include_directories(_moe_C_stable_libtorch PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/csrc")
    set_property(SOURCE ${H20_W4A8_SRCS} APPEND PROPERTY COMPILE_OPTIONS
      "$<$<COMPILE_LANGUAGE:CUDA>:-DENABLE_FP8;-U__CUDA_NO_BFLOAT16_CONVERSIONS__;--expt-relaxed-constexpr;--expt-extended-lambda>")
  endif()
endif()
