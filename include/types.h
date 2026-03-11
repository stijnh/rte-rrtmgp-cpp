#ifndef TYPES_H
#define TYPES_H

#include <map>
#include <float.h>
#include "accuracy_levels.h"


// CvH Temporary, crash on this flag to avoid trouble.
#ifdef RTE_RRTMGP_USE_CBOOL
#error "RTE_RRTMGP_USE_CBOOL is deprecated, use RTE_USE_CBOOL instead!"
#endif
// CvH End temporary

#ifdef RTE_USE_CBOOL
using Bool = signed char;
#else
using Bool = int;
#endif

using Int = unsigned long long;
const Int Atomic_reduce_const = (Int)(-1LL);


#if RTE_RTE_ACCURACY == 1 || ACCURACY >= 4 
using Float = float;
const Float Float_epsilon = FLT_EPSILON;
#else
using Float = double;
const Float Float_epsilon = DBL_EPSILON;
#endif

using FloatFlux = constants::lw_solver_noscat_kernel::flux_type;
using FloatSurface = constants::lw_solver_noscat_kernel::surface_type;
using FloatTau = constants::lw_solver_noscat_kernel::tau_type;
using FloatSource = constants::lw_solver_noscat_kernel::source_type;
using FloatTemperature = constants::gas_optical_depths_minor_kernel::temperature_type;
using FloatPressure = constants::gas_optical_depths_minor_kernel::pressure_type;
using FloatWeight = Float;
using FloatCol = Float;
using FloatOptical = FloatTau;

using FloatColDry = FloatCol;
using FloatColMix = constants::gas_optical_depths_major_kernel::col_mix_type;
using FloatColGas = constants::gas_optical_depths_minor_kernel::col_gas_type;

using FloatFMinor = constants::gas_optical_depths_minor_kernel::fminor_type;
using FloatFMajor = constants::gas_optical_depths_major_kernel::fmajor_type;
using FloatKMinor = constants::gas_optical_depths_minor_kernel::kminor_type;
using FloatKMajor = constants::gas_optical_depths_major_kernel::kmajor_type;

#endif
