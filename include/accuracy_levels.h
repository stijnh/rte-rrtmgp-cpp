#pragma once

#ifdef __CUDACC__
#include "kernel_float.h"
#include <cuda_fp16.h>
#include <cuda_bf16.h>
using bfloat16 = __nv_bfloat16;
#elif defined(__HIPCC__)
#include "kernel_float.h"
#include <hip/hip_fp16.h>
#include <hip/hip_bf16.h>
using bfloat16 = __hip_bfloat16;
#else
struct half { int16_t raw_value; };
struct bfloat16 { int16_t raw_value; };
namespace kernel_float {
    struct accurate_policy;
    struct approx_policy;
}
#endif

namespace constants {

using accurate_policy = kernel_float::accurate_policy;
using approx_policy = kernel_float::approx_policy;

#ifndef RTE_ACCURACY
#error 'constant RTE_ACCURACY' is undefined
#elif RTE_ACCURACY == 0
// Error threshold: double

// error: -15.954589770191003, time: 2.855789695467268
struct gas_optical_depths_major_kernel {
  static constexpr int block_size_x = 32;
  static constexpr int block_size_y = 1;
  static constexpr int block_size_z = 8;
  static constexpr int vector_size = 1;
  using kmajor_type = double;
  using col_mix_type = double;
  using fmajor_type = double;
  using tau_type = double;
  using compute_type = double;
};

// error: -15.954589770191003, time: 2.253385101045881
struct gas_optical_depths_minor_kernel {
  static constexpr int block_size_x = 64;
  static constexpr int block_size_y = 1;
  static constexpr int block_size_z = 2;
  static constexpr int vector_size = 1;
  static constexpr int use_smem = 0;
  using accuracy_policy = approx_policy;
  using kminor_type = double;
  using pressure_type = double;
  using temperature_type = double;
  using col_gas_type = double;
  using fminor_type = double;
  using tau_type = double;
  using compute_type = double;
};

// error: -15.954589770191003, time: 7.046582835061209
struct lw_solver_noscat_kernel {
  static constexpr int block_size_x = 32;
  static constexpr int block_size_y = 8;
  static constexpr int loop_unroll_factor_init = 4;
  static constexpr int loop_unroll_factor_nlay = 4;
  static constexpr int vector_size = 1;
  using compute_type = double;
  using tau_type = double;
  using source_type = double;
  using surface_type = double;
  using flux_type = double;
  using intermediate_type = double;
};

// error: -14.930882939341231, time: 5.923693725040981
struct sw_solver_kernel {
  static constexpr int block_size_x = 32;
  static constexpr int block_size_y = 4;
  static constexpr int vector_size = 2;
  static constexpr int loop_unroll_factor_nlay = 2;
  using stream_type = double;
  using compute_type = double;
  using tau_type = double;
  using source_type = double;
  using surface_type = double;
  using flux_type = double;
  using intermediate_type = double;
};

#elif RTE_ACCURACY == 1
// Error threshold: float

// error: -7.2461257712028875, time: 1.5428753920963831
struct gas_optical_depths_major_kernel {
  static constexpr int block_size_x = 32;
  static constexpr int block_size_y = 1;
  static constexpr int block_size_z = 4;
  static constexpr int vector_size = 1;
  using kmajor_type = float;
  using col_mix_type = float;
  using fmajor_type = float;
  using tau_type = float;
  using compute_type = float;
};

// error: -7.382772901003457, time: 1.4281874213899886
struct gas_optical_depths_minor_kernel {
  static constexpr int block_size_x = 128;
  static constexpr int block_size_y = 4;
  static constexpr int block_size_z = 2;
  static constexpr int vector_size = 1;
  static constexpr int use_smem = 0;
  using accuracy_policy = approx_policy;
  using kminor_type = float;
  using pressure_type = float;
  using temperature_type = float;
  using col_gas_type = float;
  using fminor_type = float;
  using tau_type = float;
  using compute_type = float;
};

// error: -5.7195739346586905, time: 3.5437714712960378
struct lw_solver_noscat_kernel {
  static constexpr int block_size_x = 32;
  static constexpr int block_size_y = 16;
  static constexpr int loop_unroll_factor_init = 4;
  static constexpr int loop_unroll_factor_nlay = 2;
  static constexpr int vector_size = 4;
  using compute_type = float;
  using tau_type = float;
  using source_type = float;
  using surface_type = float;
  using flux_type = float;
  using intermediate_type = float;
};

// error: -4.717422548489995, time: 3.278555461338588
struct sw_solver_kernel {
  static constexpr int block_size_x = 32;
  static constexpr int block_size_y = 2;
  static constexpr int vector_size = 2;
  static constexpr int loop_unroll_factor_nlay = 2;
  using stream_type = float;
  using compute_type = float;
  using tau_type = float;
  using source_type = float;
  using surface_type = float;
  using flux_type = float;
  using intermediate_type = float;
};

#elif RTE_ACCURACY == 2
// Error threshold: -7

// error: -7.2461257712028875, time: 1.5428753920963831
struct gas_optical_depths_major_kernel {
  static constexpr int block_size_x = 32;
  static constexpr int block_size_y = 1;
  static constexpr int block_size_z = 4;
  static constexpr int vector_size = 1;
  using kmajor_type = float;
  using col_mix_type = float;
  using fmajor_type = float;
  using tau_type = float;
  using compute_type = float;
};

// error: -7.382772901003457, time: 1.4281874213899886
struct gas_optical_depths_minor_kernel {
  static constexpr int block_size_x = 128;
  static constexpr int block_size_y = 4;
  static constexpr int block_size_z = 2;
  static constexpr int vector_size = 1;
  static constexpr int use_smem = 0;
  using accuracy_policy = approx_policy;
  using kminor_type = float;
  using pressure_type = float;
  using temperature_type = float;
  using col_gas_type = float;
  using fminor_type = float;
  using tau_type = float;
  using compute_type = float;
};

// error: -7.567053325293669, time: 5.713334900992257
struct lw_solver_noscat_kernel {
  static constexpr int block_size_x = 128;
  static constexpr int block_size_y = 2;
  static constexpr int loop_unroll_factor_init = 2;
  static constexpr int loop_unroll_factor_nlay = 2;
  static constexpr int vector_size = 4;
  using compute_type = double;
  using tau_type = float;
  using source_type = float;
  using surface_type = float;
  using flux_type = float;
  using intermediate_type = double;
};

// error: -7.418275491778431, time: 5.690221786499023
struct sw_solver_kernel {
  static constexpr int block_size_x = 64;
  static constexpr int block_size_y = 1;
  static constexpr int vector_size = 1;
  static constexpr int loop_unroll_factor_nlay = 3;
  using stream_type = double;
  using compute_type = double;
  using tau_type = float;
  using source_type = float;
  using surface_type = float;
  using flux_type = float;
  using intermediate_type = double;
};

#elif RTE_ACCURACY == 3
// Error threshold: -5

// error: -7.2461257712028875, time: 1.5428753920963831
struct gas_optical_depths_major_kernel {
  static constexpr int block_size_x = 32;
  static constexpr int block_size_y = 1;
  static constexpr int block_size_z = 4;
  static constexpr int vector_size = 1;
  using kmajor_type = float;
  using col_mix_type = float;
  using fmajor_type = float;
  using tau_type = float;
  using compute_type = float;
};

// error: -7.382772901003457, time: 1.4281874213899886
struct gas_optical_depths_minor_kernel {
  static constexpr int block_size_x = 128;
  static constexpr int block_size_y = 4;
  static constexpr int block_size_z = 2;
  static constexpr int vector_size = 1;
  static constexpr int use_smem = 0;
  using accuracy_policy = approx_policy;
  using kminor_type = float;
  using pressure_type = float;
  using temperature_type = float;
  using col_gas_type = float;
  using fminor_type = float;
  using tau_type = float;
  using compute_type = float;
};

// error: -5.7195739346586905, time: 3.5437714712960378
struct lw_solver_noscat_kernel {
  static constexpr int block_size_x = 32;
  static constexpr int block_size_y = 16;
  static constexpr int loop_unroll_factor_init = 4;
  static constexpr int loop_unroll_factor_nlay = 2;
  static constexpr int vector_size = 4;
  using compute_type = float;
  using tau_type = float;
  using source_type = float;
  using surface_type = float;
  using flux_type = float;
  using intermediate_type = float;
};

// error: -6.64285373099915, time: 4.297179562704904
struct sw_solver_kernel {
  static constexpr int block_size_x = 32;
  static constexpr int block_size_y = 2;
  static constexpr int vector_size = 2;
  static constexpr int loop_unroll_factor_nlay = 2;
  using stream_type = double;
  using compute_type = double;
  using tau_type = float;
  using source_type = float;
  using surface_type = float;
  using flux_type = float;
  using intermediate_type = float;
};

#elif RTE_ACCURACY == 4
// Error threshold: -3
// unified types for SURFACE_TYPE: {'lw_solver_noscat_kernel': 'half', 'sw_solver_kernel': 'float'} => float

// error: -3.6169684010024836, time: 1.3329553944723946
struct gas_optical_depths_major_kernel {
  static constexpr int block_size_x = 32;
  static constexpr int block_size_y = 1;
  static constexpr int block_size_z = 4;
  static constexpr int vector_size = 2;
  using kmajor_type = float;
  using col_mix_type = float;
  using fmajor_type = half;
  using tau_type = half;
  using compute_type = float;
};

// error: -3.6019714017113444, time: 1.2671268837792533
struct gas_optical_depths_minor_kernel {
  static constexpr int block_size_x = 128;
  static constexpr int block_size_y = 2;
  static constexpr int block_size_z = 2;
  static constexpr int vector_size = 2;
  static constexpr int use_smem = 0;
  using accuracy_policy = approx_policy;
  using kminor_type = float;
  using pressure_type = float;
  using temperature_type = float;
  using col_gas_type = float;
  using fminor_type = float;
  using tau_type = half;
  using compute_type = float;
};

// error: -4.083155082159606, time: 3.3786148684365407
struct lw_solver_noscat_kernel {
  static constexpr int block_size_x = 64;
  static constexpr int block_size_y = 1;
  static constexpr int loop_unroll_factor_init = 2;
  static constexpr int loop_unroll_factor_nlay = 1;
  static constexpr int vector_size = 4;
  using compute_type = float;
  using tau_type = half;
  using source_type = half;
  using surface_type = float;
  using flux_type = float;
  using intermediate_type = float;
};

// error: -4.604790145237762, time: 3.147629737854004
struct sw_solver_kernel {
  static constexpr int block_size_x = 128;
  static constexpr int block_size_y = 1;
  static constexpr int vector_size = 1;
  static constexpr int loop_unroll_factor_nlay = 4;
  using stream_type = float;
  using compute_type = float;
  using tau_type = half;
  using source_type = half;
  using surface_type = float;
  using flux_type = float;
  using intermediate_type = float;
};

#elif RTE_ACCURACY == 5
// Error threshold: -2

// error: -2.790702657785704, time: 1.248256002153669
struct gas_optical_depths_major_kernel {
  static constexpr int block_size_x = 32;
  static constexpr int block_size_y = 1;
  static constexpr int block_size_z = 4;
  static constexpr int vector_size = 2;
  using kmajor_type = float;
  using col_mix_type = bfloat16;
  using fmajor_type = half;
  using tau_type = half;
  using compute_type = float;
};

// error: -3.6019714017113444, time: 1.2671268837792533
struct gas_optical_depths_minor_kernel {
  static constexpr int block_size_x = 128;
  static constexpr int block_size_y = 2;
  static constexpr int block_size_z = 2;
  static constexpr int vector_size = 2;
  static constexpr int use_smem = 0;
  using accuracy_policy = approx_policy;
  using kminor_type = float;
  using pressure_type = float;
  using temperature_type = float;
  using col_gas_type = float;
  using fminor_type = float;
  using tau_type = half;
  using compute_type = float;
};

// error: -2.5808328460451424, time: 2.079451424734933
struct lw_solver_noscat_kernel {
  static constexpr int block_size_x = 64;
  static constexpr int block_size_y = 16;
  static constexpr int loop_unroll_factor_init = 4;
  static constexpr int loop_unroll_factor_nlay = 8;
  static constexpr int vector_size = 4;
  using compute_type = float;
  using tau_type = half;
  using source_type = half;
  using surface_type = float;
  using flux_type = bfloat16;
  using intermediate_type = half;
};

// error: -2.569379263282942, time: 3.007634231022426
struct sw_solver_kernel {
  static constexpr int block_size_x = 128;
  static constexpr int block_size_y = 1;
  static constexpr int vector_size = 1;
  static constexpr int loop_unroll_factor_nlay = 4;
  using stream_type = float;
  using compute_type = float;
  using tau_type = half;
  using source_type = half;
  using surface_type = float;
  using flux_type = bfloat16;
  using intermediate_type = float;
};

#elif RTE_ACCURACY == 6
// Error threshold: None
// unified types for SURFACE_TYPE: {'lw_solver_noscat_kernel': 'float', 'sw_solver_kernel': 'half'} => float

// error: -2.790702657785704, time: 1.248256002153669
struct gas_optical_depths_major_kernel {
  static constexpr int block_size_x = 32;
  static constexpr int block_size_y = 1;
  static constexpr int block_size_z = 4;
  static constexpr int vector_size = 2;
  using kmajor_type = float;
  using col_mix_type = bfloat16;
  using fmajor_type = half;
  using tau_type = half;
  using compute_type = float;
};

// error: -3.6019714017113444, time: 1.2671268837792533
struct gas_optical_depths_minor_kernel {
  static constexpr int block_size_x = 128;
  static constexpr int block_size_y = 2;
  static constexpr int block_size_z = 2;
  static constexpr int vector_size = 2;
  static constexpr int use_smem = 0;
  using accuracy_policy = approx_policy;
  using kminor_type = float;
  using pressure_type = float;
  using temperature_type = float;
  using col_gas_type = float;
  using fminor_type = float;
  using tau_type = half;
  using compute_type = float;
};

// error: -1.861116552184259, time: 1.9492571353912354
struct lw_solver_noscat_kernel {
  static constexpr int block_size_x = 512;
  static constexpr int block_size_y = 1;
  static constexpr int loop_unroll_factor_init = 4;
  static constexpr int loop_unroll_factor_nlay = 8;
  static constexpr int vector_size = 4;
  using compute_type = half;
  using tau_type = half;
  using source_type = half;
  using surface_type = float;
  using flux_type = bfloat16;
  using intermediate_type = half;
};

// error: 0.03455186166798564, time: 2.2832273755754744
struct sw_solver_kernel {
  static constexpr int block_size_x = 128;
  static constexpr int block_size_y = 8;
  static constexpr int vector_size = 2;
  static constexpr int loop_unroll_factor_nlay = 4;
  using stream_type = half;
  using compute_type = half;
  using tau_type = half;
  using source_type = half;
  using surface_type = float;
  using flux_type = bfloat16;
  using intermediate_type = bfloat16;
};

#endif


}
