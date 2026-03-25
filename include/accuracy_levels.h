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

// error: -15.954589770191003, time: 5.066766057695661
struct gas_optical_depths_major_kernel {
  static constexpr int block_size_x = 32;
  static constexpr int block_size_y = 2;
  static constexpr int block_size_z = 8;
  static constexpr int vector_size = 1;
  using kmajor_type = double;
  using col_mix_type = double;
  using fmajor_type = double;
  using tau_type = double;
  using compute_type = double;
};

// error: -15.954589770191003, time: 5.6369927270071845
struct gas_optical_depths_minor_kernel {
  static constexpr int block_size_x = 128;
  static constexpr int block_size_y = 2;
  static constexpr int block_size_z = 4;
  static constexpr int vector_size = 1;
  static constexpr int use_smem = 1;
  using accuracy_policy = accurate_policy;
  using kminor_type = double;
  using pressure_type = double;
  using temperature_type = double;
  using col_gas_type = double;
  using fminor_type = double;
  using tau_type = double;
  using compute_type = double;
};

// error: -15.954589770191003, time: 8.400064877101354
struct lw_solver_noscat_kernel {
  static constexpr int block_size_x = 256;
  static constexpr int block_size_y = 2;
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

// error: -12.843672070511605, time: 7.290224892752511
struct sw_solver_kernel {
  static constexpr int block_size_x = 256;
  static constexpr int block_size_y = 1;
  static constexpr int vector_size = 1;
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

// error: -7.2461257712028875, time: 2.715643746512277
struct gas_optical_depths_major_kernel {
  static constexpr int block_size_x = 64;
  static constexpr int block_size_y = 1;
  static constexpr int block_size_z = 8;
  static constexpr int vector_size = 1;
  using kmajor_type = float;
  using col_mix_type = float;
  using fmajor_type = float;
  using tau_type = float;
  using compute_type = float;
};

// error: -7.382772897995827, time: 2.6160715648106168
struct gas_optical_depths_minor_kernel {
  static constexpr int block_size_x = 128;
  static constexpr int block_size_y = 2;
  static constexpr int block_size_z = 4;
  static constexpr int vector_size = 1;
  static constexpr int use_smem = 1;
  using accuracy_policy = approx_policy;
  using kminor_type = float;
  using pressure_type = float;
  using temperature_type = float;
  using col_gas_type = float;
  using fminor_type = float;
  using tau_type = float;
  using compute_type = float;
};

// error: -5.97838179062345, time: 4.370594773973737
struct lw_solver_noscat_kernel {
  static constexpr int block_size_x = 128;
  static constexpr int block_size_y = 4;
  static constexpr int loop_unroll_factor_init = 2;
  static constexpr int loop_unroll_factor_nlay = 4;
  static constexpr int vector_size = 1;
  using compute_type = float;
  using tau_type = float;
  using source_type = float;
  using surface_type = float;
  using flux_type = float;
  using intermediate_type = float;
};

// error: -4.81495620865894, time: 3.8390676975250244
struct sw_solver_kernel {
  static constexpr int block_size_x = 256;
  static constexpr int block_size_y = 1;
  static constexpr int vector_size = 1;
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

// error: -7.513451999317735, time: 2.590318134852818
struct gas_optical_depths_major_kernel {
  static constexpr int block_size_x = 64;
  static constexpr int block_size_y = 1;
  static constexpr int block_size_z = 8;
  static constexpr int vector_size = 1;
  using kmajor_type = float;
  using col_mix_type = double;
  using fmajor_type = float;
  using tau_type = float;
  using compute_type = double;
};

// error: -7.382772897995827, time: 2.6160715648106168
struct gas_optical_depths_minor_kernel {
  static constexpr int block_size_x = 128;
  static constexpr int block_size_y = 2;
  static constexpr int block_size_z = 4;
  static constexpr int vector_size = 1;
  static constexpr int use_smem = 1;
  using accuracy_policy = approx_policy;
  using kminor_type = float;
  using pressure_type = float;
  using temperature_type = float;
  using col_gas_type = float;
  using fminor_type = float;
  using tau_type = float;
  using compute_type = float;
};

// error: -7.567053325293669, time: 6.515485899788993
struct lw_solver_noscat_kernel {
  static constexpr int block_size_x = 512;
  static constexpr int block_size_y = 1;
  static constexpr int loop_unroll_factor_init = 1;
  static constexpr int loop_unroll_factor_nlay = 2;
  static constexpr int vector_size = 1;
  using compute_type = double;
  using tau_type = float;
  using source_type = float;
  using surface_type = float;
  using flux_type = float;
  using intermediate_type = double;
};

// error: -7.418275491778431, time: 5.890273911612375
struct sw_solver_kernel {
  static constexpr int block_size_x = 256;
  static constexpr int block_size_y = 1;
  static constexpr int vector_size = 1;
  static constexpr int loop_unroll_factor_nlay = 1;
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
// unified types for TAU_TYPE: {'gas_optical_depths_major_kernel': 'float', 'gas_optical_depths_minor_kernel': 'float', 'lw_solver_noscat_kernel': 'float', 'sw_solver_kernel': 'half'} => float
// unified types for SOURCE_TYPE: {'lw_solver_noscat_kernel': 'float', 'sw_solver_kernel': 'half'} => float
// unified types for SURFACE_TYPE: {'lw_solver_noscat_kernel': 'float', 'sw_solver_kernel': 'double'} => double

// error: -7.513451999317735, time: 2.590318134852818
struct gas_optical_depths_major_kernel {
  static constexpr int block_size_x = 64;
  static constexpr int block_size_y = 1;
  static constexpr int block_size_z = 8;
  static constexpr int vector_size = 1;
  using kmajor_type = float;
  using col_mix_type = double;
  using fmajor_type = float;
  using tau_type = float;
  using compute_type = double;
};

// error: -7.382772897995827, time: 2.6160715648106168
struct gas_optical_depths_minor_kernel {
  static constexpr int block_size_x = 128;
  static constexpr int block_size_y = 2;
  static constexpr int block_size_z = 4;
  static constexpr int vector_size = 1;
  static constexpr int use_smem = 1;
  using accuracy_policy = approx_policy;
  using kminor_type = float;
  using pressure_type = float;
  using temperature_type = float;
  using col_gas_type = float;
  using fminor_type = float;
  using tau_type = float;
  using compute_type = float;
};

// error: -5.97838179062345, time: 4.370594773973737
struct lw_solver_noscat_kernel {
  static constexpr int block_size_x = 128;
  static constexpr int block_size_y = 4;
  static constexpr int loop_unroll_factor_init = 2;
  static constexpr int loop_unroll_factor_nlay = 4;
  static constexpr int vector_size = 1;
  using compute_type = float;
  using tau_type = float;
  using source_type = float;
  using surface_type = double;
  using flux_type = float;
  using intermediate_type = float;
};

// error: -5.053038007523458, time: 3.851040840148926
struct sw_solver_kernel {
  static constexpr int block_size_x = 128;
  static constexpr int block_size_y = 2;
  static constexpr int vector_size = 2;
  static constexpr int loop_unroll_factor_nlay = 2;
  using stream_type = double;
  using compute_type = double;
  using tau_type = float;
  using source_type = float;
  using surface_type = double;
  using flux_type = float;
  using intermediate_type = float;
};

#elif RTE_ACCURACY == 4
// Error threshold: -3

// error: -3.6156039435823915, time: 2.3162197385515486
struct gas_optical_depths_major_kernel {
  static constexpr int block_size_x = 32;
  static constexpr int block_size_y = 1;
  static constexpr int block_size_z = 8;
  static constexpr int vector_size = 1;
  using kmajor_type = float;
  using col_mix_type = float;
  using fmajor_type = half;
  using tau_type = half;
  using compute_type = float;
};

// error: -3.6019709768292585, time: 2.2411708491189137
struct gas_optical_depths_minor_kernel {
  static constexpr int block_size_x = 128;
  static constexpr int block_size_y = 2;
  static constexpr int block_size_z = 4;
  static constexpr int vector_size = 1;
  static constexpr int use_smem = 1;
  using accuracy_policy = approx_policy;
  using kminor_type = float;
  using pressure_type = float;
  using temperature_type = float;
  using col_gas_type = float;
  using fminor_type = float;
  using tau_type = half;
  using compute_type = float;
};

// error: -4.083328304181949, time: 3.8956217084612166
struct lw_solver_noscat_kernel {
  static constexpr int block_size_x = 256;
  static constexpr int block_size_y = 1;
  static constexpr int loop_unroll_factor_init = 2;
  static constexpr int loop_unroll_factor_nlay = 2;
  static constexpr int vector_size = 2;
  using compute_type = float;
  using tau_type = half;
  using source_type = half;
  using surface_type = half;
  using flux_type = float;
  using intermediate_type = float;
};

// error: -3.9418684717561643, time: 3.5053672790527344
struct sw_solver_kernel {
  static constexpr int block_size_x = 128;
  static constexpr int block_size_y = 2;
  static constexpr int vector_size = 2;
  static constexpr int loop_unroll_factor_nlay = 1;
  using stream_type = float;
  using compute_type = float;
  using tau_type = half;
  using source_type = half;
  using surface_type = half;
  using flux_type = float;
  using intermediate_type = float;
};

#elif RTE_ACCURACY == 5
// Error threshold: -2
// unified types for FLUX_TYPE: {'lw_solver_noscat_kernel': 'bfloat16', 'sw_solver_kernel': 'float'} => float

// error: -2.790702657785704, time: 2.114256041390555
struct gas_optical_depths_major_kernel {
  static constexpr int block_size_x = 32;
  static constexpr int block_size_y = 1;
  static constexpr int block_size_z = 8;
  static constexpr int vector_size = 1;
  using kmajor_type = float;
  using col_mix_type = bfloat16;
  using fmajor_type = half;
  using tau_type = half;
  using compute_type = float;
};

// error: -3.6019709768292585, time: 2.2411708491189137
struct gas_optical_depths_minor_kernel {
  static constexpr int block_size_x = 128;
  static constexpr int block_size_y = 2;
  static constexpr int block_size_z = 4;
  static constexpr int vector_size = 1;
  static constexpr int use_smem = 1;
  using accuracy_policy = approx_policy;
  using kminor_type = float;
  using pressure_type = float;
  using temperature_type = float;
  using col_gas_type = float;
  using fminor_type = float;
  using tau_type = half;
  using compute_type = float;
};

// error: -2.581038819143616, time: 2.547574554170881
struct lw_solver_noscat_kernel {
  static constexpr int block_size_x = 256;
  static constexpr int block_size_y = 1;
  static constexpr int loop_unroll_factor_init = 1;
  static constexpr int loop_unroll_factor_nlay = 8;
  static constexpr int vector_size = 2;
  using compute_type = float;
  using tau_type = half;
  using source_type = half;
  using surface_type = half;
  using flux_type = float;
  using intermediate_type = half;
};

// error: -3.9418684717561643, time: 3.5053672790527344
struct sw_solver_kernel {
  static constexpr int block_size_x = 128;
  static constexpr int block_size_y = 2;
  static constexpr int vector_size = 2;
  static constexpr int loop_unroll_factor_nlay = 1;
  using stream_type = float;
  using compute_type = float;
  using tau_type = half;
  using source_type = half;
  using surface_type = half;
  using flux_type = float;
  using intermediate_type = float;
};

#elif RTE_ACCURACY == 6
// Error threshold: -1
// unified types for SURFACE_TYPE: {'lw_solver_noscat_kernel': 'half', 'sw_solver_kernel': 'float'} => float
// unified types for FLUX_TYPE: {'lw_solver_noscat_kernel': 'bfloat16', 'sw_solver_kernel': 'float'} => float

// error: -2.790702657785704, time: 2.114256041390555
struct gas_optical_depths_major_kernel {
  static constexpr int block_size_x = 32;
  static constexpr int block_size_y = 1;
  static constexpr int block_size_z = 8;
  static constexpr int vector_size = 1;
  using kmajor_type = float;
  using col_mix_type = bfloat16;
  using fmajor_type = half;
  using tau_type = half;
  using compute_type = float;
};

// error: -3.6019709768292585, time: 2.2411708491189137
struct gas_optical_depths_minor_kernel {
  static constexpr int block_size_x = 128;
  static constexpr int block_size_y = 2;
  static constexpr int block_size_z = 4;
  static constexpr int vector_size = 1;
  static constexpr int use_smem = 1;
  using accuracy_policy = approx_policy;
  using kminor_type = half;
  using pressure_type = float;
  using temperature_type = float;
  using col_gas_type = bfloat16;
  using fminor_type = half;
  using tau_type = half;
  using compute_type = float;

};

// error: -1.8611426273023153, time: 2.454885619027274
struct lw_solver_noscat_kernel {
  static constexpr int block_size_x = 256;
  static constexpr int block_size_y = 1;
  static constexpr int loop_unroll_factor_init = 1;
  static constexpr int loop_unroll_factor_nlay = 1;
  static constexpr int vector_size = 2;
  using compute_type = half;
  using tau_type = half;
  using source_type = half;
  using surface_type = float;
  using flux_type = float;
  using intermediate_type = half;
};

// error: -1.2663246287427383, time: 2.8399551595960344
struct sw_solver_kernel {
  static constexpr int block_size_x = 256;
  static constexpr int block_size_y = 1;
  static constexpr int vector_size = 2;
  static constexpr int loop_unroll_factor_nlay = 1;
  using stream_type = float;
  using compute_type = half;
  using tau_type = half;
  using source_type = half;
  using surface_type = float;
  using flux_type = float;
  using intermediate_type = bfloat16;
};

#endif

}

