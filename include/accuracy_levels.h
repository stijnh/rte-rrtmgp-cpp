#pragma once

#ifdef __CUDACC__
#include <cuda_fp16.h>
#include <cuda_bf16.h>
using bfloat16 = __nv_bfloat16;
#elif defined(__HIPCC__)
#include <hip/hip_fp16.h>
#include <hip/hip_bf16.h>
using bfloat16 = __hip_bfloat16;
#else
struct half { int16_t _; };
struct bfloat16 { int16_t _; };
#endif

namespace constants {

#ifndef RTE_ACCURACY
#error "constant RTE_ACCURACY' is undefined"
#elif RTE_ACCURACY == 0
// error: -16.553742706423293, time: 7.212470872061593
struct lw_solver_noscat_kernel {
  static constexpr int block_size_x = 32;
  static constexpr int block_size_y = 8;
  static constexpr int loop_unroll_factor_init = 4;
  static constexpr int loop_unroll_factor_nlay = 4;
  static constexpr int vector_size = 2;
  using tau_type = double;
  using source_type = double;
  using surface_type = double;
  using flux_type = double;
  using compute_type = double;
  using intermediate_type = double;
};

// error: -15.050186589151291, time: 7.435702732631138
struct sw_solver_kernel {
  static constexpr int block_size_x = 64;
  static constexpr int block_size_y = 1;
  static constexpr int vector_size = 1;
  static constexpr int loop_unroll_factor_nlay = 4;
  using stream_type = double;
  using tau_type = double;
  using source_type = double;
  using surface_type = double;
  using flux_type = double;
  using compute_type = double;
  using intermediate_type = double;
};

#elif RTE_ACCURACY == 1
// error: -7.066648869885818, time: 4.67777817589896
struct lw_solver_noscat_kernel {
  static constexpr int block_size_x = 256;
  static constexpr int block_size_y = 1;
  static constexpr int loop_unroll_factor_init = 4;
  static constexpr int loop_unroll_factor_nlay = 8;
  static constexpr int vector_size = 2;
  using compute_type = double;
  using tau_type = double;
  using source_type = float;
  using surface_type = double;
  using flux_type = float;
  using intermediate_type = float;
};

// error: -7.4713481733041345, time: 7.09866053717477
struct sw_solver_kernel {
  static constexpr int block_size_x = 128;
  static constexpr int block_size_y = 1;
  static constexpr int vector_size = 1;
  static constexpr int loop_unroll_factor_nlay = 4;
  using stream_type = double;
  using compute_type = double;
  using tau_type = double;
  using source_type = float;
  using surface_type = double;
  using flux_type = float;
  using intermediate_type = double;
};

#elif RTE_ACCURACY == 2
// error: -5.785829283999813, time: 3.630957671574184
struct lw_solver_noscat_kernel {
  static constexpr int block_size_x = 32;
  static constexpr int block_size_y = 8;
  static constexpr int loop_unroll_factor_init = 4;
  static constexpr int loop_unroll_factor_nlay = 4;
  static constexpr int vector_size = 4;
  using compute_type = float;
  using tau_type = float;
  using source_type = float;
  using surface_type = float;
  using flux_type = float;
  using intermediate_type = float;
};

// error: -4.765881642008952, time: 3.9312823159354076
struct sw_solver_kernel {
  static constexpr int block_size_x = 128;
  static constexpr int block_size_y = 1;
  static constexpr int vector_size = 1;
  static constexpr int loop_unroll_factor_nlay = 4;
  using stream_type = float;
  using compute_type = float;
  using tau_type = float;
  using source_type = float;
  using surface_type = float;
  using flux_type = float;
  using intermediate_type = float;
};

#elif RTE_ACCURACY == 3
// error: -2.6864724477963526, time: 2.5468342985425676
struct lw_solver_noscat_kernel {
  static constexpr int block_size_x = 64;
  static constexpr int block_size_y = 4;
  static constexpr int loop_unroll_factor_init = 4;
  static constexpr int loop_unroll_factor_nlay = 1;
  static constexpr int vector_size = 4;
  using compute_type = float;
  using tau_type = float;
  using source_type = float;
  using surface_type = half;
  using flux_type = bfloat16;
  using intermediate_type = half;
};

// error: -2.6618383226485407, time: 3.895296028682164
struct sw_solver_kernel {
  static constexpr int block_size_x = 256;
  static constexpr int block_size_y = 1;
  static constexpr int vector_size = 2;
  static constexpr int loop_unroll_factor_nlay = 4;
  using stream_type = float;
  using compute_type = float;
  using tau_type = float;
  using source_type = float;
  using surface_type = half;
  using flux_type = bfloat16;
  using intermediate_type = float;
};

#elif RTE_ACCURACY == 4
// error: -1.9619967805291267, time: 2.010843413216727
struct lw_solver_noscat_kernel {
  static constexpr int block_size_x = 64;
  static constexpr int block_size_y = 2;
  static constexpr int loop_unroll_factor_init = 4;
  static constexpr int loop_unroll_factor_nlay = 2;
  static constexpr int vector_size = 4;
  using compute_type = half;
  using tau_type = half;
  using source_type = half;
  using surface_type = double;
  using flux_type = bfloat16;
  using intermediate_type = half;
};

// error: -1.217967715108047, time: 4.081663949148996
struct sw_solver_kernel {
  static constexpr int block_size_x = 64;
  static constexpr int block_size_y = 2;
  static constexpr int vector_size = 4;
  static constexpr int loop_unroll_factor_nlay = 4;
  using stream_type = float;
  using compute_type = float;
  using tau_type = half;
  using source_type = half;
  using surface_type = double;
  using flux_type = bfloat16;
  using intermediate_type = bfloat16;
};

#endif
}