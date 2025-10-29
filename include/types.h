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


#if RTE_ACCURACY >= 2
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
using FloatTemperature = Float;
using FloatPressure = Float;
using FloatWeight = Float;
using FloatCol = Float;
using FloatOptical = FloatTau;

using FloatColDry = FloatCol;
using FloatColMix = FloatCol;
using FloatColGas = FloatCol;

using FloatFMinor = FloatWeight;
using FloatFMajor = FloatWeight;
using FloatKMinor = FloatWeight;
using FloatKMajor = FloatWeight;

#endif
