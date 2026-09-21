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
    set_gencode_flags_for_srcs(SRCS "${H20_W4A8_SRCS}" CUDA_ARCHS "${H20_W4A8_ARCHS}")
    target_sources(_moe_C_stable_libtorch PRIVATE ${H20_W4A8_SRCS})
    target_include_directories(_moe_C_stable_libtorch PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/csrc")
    set_property(SOURCE ${H20_W4A8_SRCS} APPEND PROPERTY COMPILE_OPTIONS
      "$<$<COMPILE_LANGUAGE:CUDA>:-DENABLE_FP8;-U__CUDA_NO_BFLOAT16_CONVERSIONS__;--expt-relaxed-constexpr;--expt-extended-lambda>")
  endif()
endif()
