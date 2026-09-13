# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# The staged SM80 path uses packed, unpadded LSE. The upstream empty-KV
# branch assumes padded LSE and can write outside the packed allocation.
# Generate a corrected header for this target without modifying the shared
# dependency or changing the stock FA2 target's include path.
function(sm80_configure_fa2_lse_header source destination)
  file(READ "${source}" _source)
  set(_old [=[        const index_t row_offset_lseaccum = ((n_split_idx * params.b + bidb) * params.h + bidh) * params.seqlen_q + m_block * kBlockM;]=])
  set(_new [=[        const index_t row_offset_lseaccum = (Split || !params.unpadded_lse ?
            ((n_split_idx * params.b + bidb) * params.h + bidh) * params.seqlen_q :
            bidh * params.total_q + binfo.q_offset(params.seqlen_q, 1, bidb)
        ) + m_block * kBlockM;]=])
  string(FIND "${_source}" "${_old}" _old_position)
  string(FIND "${_source}" "${_new}" _new_position)
  if(_old_position GREATER_EQUAL 0)
    string(REPLACE "${_old}" "${_new}" _source "${_source}")
  elseif(_new_position LESS 0)
    message(FATAL_ERROR
      "SM80 FA2 empty-KV LSE fix no longer matches flash_fwd_kernel.h; review the dependency update.")
  endif()
  file(CONFIGURE OUTPUT "${destination}" CONTENT "${_source}" @ONLY)
endfunction()
