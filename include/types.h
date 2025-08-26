#ifndef TYPES_H
#define TYPES_H

#include <map>
#include <float.h>

#ifdef __CUDACC__
#include <cuda_fp16.h>
#else
struct half;
#endif

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

#ifdef RTE_USE_SP
using Float = float;
const Float Float_epsilon = FLT_EPSILON;
#else
using Float = double;
const Float Float_epsilon = DBL_EPSILON;
#endif

using Int = unsigned long long;
const Int Atomic_reduce_const = (Int)(-1LL);

using FloatIntermediate = Float;
using FloatFlux = Float;
using FloatSurface = Float;
using FloatTemperature = Float;
using FloatPressure = Float;
using FloatWeight = Float;
using FloatCol = Float;
using FloatTau = Float;
using FloatSource = Float;
using FloatOptical = Float;

using FloatColDry = FloatCol;
using FloatColMix = FloatCol;
using FloatColGas = FloatCol;

using FloatFMinor = FloatWeight;
using FloatFMajor = FloatWeight;
using FloatKMinor = FloatWeight;
using FloatKMajor = FloatWeight;

#endif
