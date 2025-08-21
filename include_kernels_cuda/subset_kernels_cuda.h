/*
 * This file is part of a C++ interface to the Radiative Transfer for Energetics (RTE)
 * and Rapid Radiative Transfer Model for GCM applications Parallel (RRTMGP).
 *
 * The original code is found at https://github.com/RobertPincus/rte-rrtmgp.
 *
 * Contacts: Robert Pincus and Eli Mlawer
 * email: rrtmgp@aer.com
 *
 * Copyright 2015-2020,  Atmospheric and Environmental Research and
 * Regents of the University of Colorado.  All right reserved.
 *
 * This C++ interface can be downloaded from https://github.com/microhh/rte-rrtmgp-cpp
 *
 * Contact: Chiel van Heerwaarden
 * email: chiel.vanheerwaarden@wur.nl
 *
 * Copyright 2020, Wageningen University & Research.
 *
 * Use and duplication is permitted under the terms of the
 * BSD 3-clause license, see http://opensource.org/licenses/BSD-3-Clause
 *
 */

#ifndef RTE_KERNELS_CUDA_H
#define RTE_KERNELS_CUDA_H

#include "types.h"


namespace Subset_kernels_cuda
{
    void get_from_subset(
            const int ncol, const int nlay, const int nbnd, const int ncol_in, const int col_s_in,
            double* var1_full, const double* var1_sub);

    void get_from_subset(
            const int ncol, const int nlay, const int nbnd, const int ncol_in, const int col_s_in,
            float* var1_full, const float* var1_sub);

    void get_from_subset(
            const int ncol, const int nlay, const int nbnd, const int ncol_in, const int col_s_in,
            half* var1_full, const half* var1_sub);

    template <typename T1, typename T2>
    void get_from_subset(
            const int ncol, const int nlay, const int nbnd, const int ncol_in, const int col_s_in,
            T1* var1_full, T2* var2_full,
            const T1* var1_sub, const T2* var2_sub) {
        get_from_subset(ncol, nlay, nbnd, ncol_in, col_s_in, var1_full, var1_sub);
        get_from_subset(ncol, nlay, nbnd, ncol_in, col_s_in, var2_full, var2_sub);
    }

    template <typename T1, typename T2, typename T3>
    void get_from_subset(
            const int ncol, const int nlay, const int nbnd, const int ncol_in, const int col_s_in,
            T1* var1_full, T2* var2_full, T3* var3_full,
            const T1* var1_sub, const T2* var2_sub, const T3* var3_sub
    ) {
        get_from_subset(ncol, nlay, nbnd, ncol_in, col_s_in, var1_full, var1_sub);
        get_from_subset(ncol, nlay, nbnd, ncol_in, col_s_in, var2_full, var2_sub);
        get_from_subset(ncol, nlay, nbnd, ncol_in, col_s_in, var3_full, var3_sub);
    }

    template <typename T1, typename T2, typename T3, typename T4>
    void get_from_subset(
            const int ncol, const int nlay, const int nbnd, const int ncol_in, const int col_s_in,
            T1* var1_full, T2* var2_full, T3* var3_full, T4* var4_full,
            const T1* var1_sub, const T2* var2_sub, const T3* var3_sub, const T4* var4_sub
    ) {
        get_from_subset(ncol, nlay, nbnd, ncol_in, col_s_in, var1_full, var1_sub);
        get_from_subset(ncol, nlay, nbnd, ncol_in, col_s_in, var2_full, var2_sub);
        get_from_subset(ncol, nlay, nbnd, ncol_in, col_s_in, var3_full, var3_sub);
        get_from_subset(ncol, nlay, nbnd, ncol_in, col_s_in, var4_full, var4_sub);
    }

    template <typename T>
    void get_from_subset(
            const int ncol, const int nlay, const int ncol_in, const int col_s_in,
            T* var1_full, const T* var1_sub) {
        get_from_subset(ncol, nlay, 1, ncol_in, col_s_in, var1_full, var1_sub);
    }

    template <typename T1, typename T2>
    void get_from_subset(
            const int ncol, const int nlay, const int ncol_in, const int col_s_in,
            T1* var1_full, T2* var2_full,
            const T1* var1_sub, const T2* var2_sub) {
        get_from_subset(ncol, nlay, ncol_in, col_s_in, var1_full, var1_sub);
        get_from_subset(ncol, nlay, ncol_in, col_s_in, var2_full, var2_sub);
    }

    template <typename T1, typename T2, typename T3>
    void get_from_subset(
            const int ncol, const int nlay, const int ncol_in, const int col_s_in,
            T1* var1_full, T2* var2_full, T3* var3_full,
            const T1* var1_sub, const T2* var2_sub, const T3* var3_sub
    ) {
        get_from_subset(ncol, nlay, ncol_in, col_s_in, var1_full, var1_sub);
        get_from_subset(ncol, nlay, ncol_in, col_s_in, var2_full, var2_sub);
        get_from_subset(ncol, nlay, ncol_in, col_s_in, var3_full, var3_sub);
    }

    template <typename T1, typename T2, typename T3, typename T4>
    void get_from_subset(
            const int ncol, const int nlay, const int ncol_in, const int col_s_in,
            T1* var1_full, T2* var2_full, T3* var3_full, T4* var4_full,
            const T1* var1_sub, const T2* var2_sub, const T3* var3_sub, const T4* var4_sub
    ) {
        get_from_subset(ncol, nlay, ncol_in, col_s_in, var1_full, var1_sub);
        get_from_subset(ncol, nlay, ncol_in, col_s_in, var2_full, var2_sub);
        get_from_subset(ncol, nlay, ncol_in, col_s_in, var3_full, var3_sub);
        get_from_subset(ncol, nlay, ncol_in, col_s_in, var4_full, var4_sub);
    }
}
#endif
