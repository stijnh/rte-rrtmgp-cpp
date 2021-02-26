/*
 * This file is part of a C++ interface to the Radiative Transfer for Energetics (RTE)
 * and Rapid Radiative Transfer Model for GCM applications Parallel (RRTMGP).
 *
 * The original code is found at https://github.com/earth-system-radiation/rte-rrtmgp.
 *
 * Contacts: Robert Pincus and Eli Mlawer
 * email: rrtmgp@aer.com
 *
 * Copyright 2015-2020,  Atmospheric and Environmental Research and
 * Regents of the University of Colorado.  All right reserved.
 *
 * This C++ interface can be downloaded from https://github.com/earth-system-radiation/rte-rrtmgp-cpp
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

#ifndef WORK_ARRAY_OWNER_H_
#define WORK_ARRAY_OWNER_H_
#include <memory>

template<typename TW> class Work_array_owner
{
    public:

        virtual std::unique_ptr<TW> create_work_arrays(
            const int n_cols, 
            const int n_levs, 
            const int n_lays, 
            const int b_bnds) const=0;

        void set_work_arrays(std::shared_ptr<TW>& work_arrays_)
        {
            work_arrays = work_arrays_;
        }

    protected:

        std::shared_ptr<TW> work_arrays;
};

#endif