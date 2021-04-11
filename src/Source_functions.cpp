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
 * email: chiel.vanheerwaarden@wur.nl
 * Contact: Chiel van Heerwaarden
 *
 * Copyright 2020, Wageningen University & Research.
 *
 * Use and duplication is permitted under the terms of the
 * BSD 3-clause license, see http://opensource.org/licenses/BSD-3-Clause
 *
 */

#include <vector>
#include "Source_functions.h"
#include "Array.h"
#include "Optical_props.h"

template<typename TF>
Source_func_lw<TF>::Source_func_lw(
        const int n_col,
        const int n_lay,
        const Optical_props<TF>& optical_props,
        Pool_base<std::vector<TF>>* pool) :
    Optical_props<TF>(optical_props), Pool_client_group<std::vector<TF>>(pool),
    sfc_source({n_col, optical_props.get_ngpt()}, pool),
    sfc_source_jac({n_col, optical_props.get_ngpt()}, pool),
    lay_source({n_col, n_lay, optical_props.get_ngpt()}, pool),
    lev_source_inc({n_col, n_lay, optical_props.get_ngpt()}, pool),
    lev_source_dec({n_col, n_lay, optical_props.get_ngpt()}, pool)
{
    this->add_client(sfc_source);
    this->add_client(sfc_source_jac);
    this->add_client(lay_source);
    this->add_client(lev_source_inc);
    this->add_client(lev_source_dec);
}

template<typename TF>
void Source_func_lw<TF>::set_subset(
        const Source_func_lw<TF>& sources_sub,
        const int col_s, const int col_e)
{
    for (int igpt=1; igpt<=lay_source.dim(3); ++igpt)
        for (int icol=col_s; icol<=col_e; ++icol)
            sfc_source({icol, igpt}) = sources_sub.get_sfc_source()({icol-col_s+1, igpt});

    for (int igpt=1; igpt<=lay_source.dim(3); ++igpt)
        for (int ilay=1; ilay<=lay_source.dim(2); ++ilay)
            for (int icol=col_s; icol<=col_e; ++icol)
            {
                lay_source    ({icol, ilay, igpt}) = sources_sub.get_lay_source()    ({icol-col_s+1, ilay, igpt});
                lev_source_inc({icol, ilay, igpt}) = sources_sub.get_lev_source_inc()({icol-col_s+1, ilay, igpt});
                lev_source_dec({icol, ilay, igpt}) = sources_sub.get_lev_source_dec()({icol-col_s+1, ilay, igpt});
            }
}

template<typename TF>
void Source_func_lw<TF>::get_subset(
        const Source_func_lw<TF>& sources_sub,
        const int col_s, const int col_e)
{
    for (int igpt=1; igpt<=lay_source.dim(3); ++igpt)
        for (int icol=col_s; icol<=col_e; ++icol)
            sfc_source({icol-col_s+1, igpt}) = sources_sub.get_sfc_source()({icol, igpt});

    for (int igpt=1; igpt<=lay_source.dim(3); ++igpt)
        for (int ilay=1; ilay<=lay_source.dim(2); ++ilay)
            for (int icol=col_s; icol<=col_e; ++icol)
            {
                lay_source    ({icol-col_s+1, ilay, igpt}) = sources_sub.get_lay_source()    ({icol, ilay, igpt});
                lev_source_inc({icol-col_s+1, ilay, igpt}) = sources_sub.get_lev_source_inc()({icol, ilay, igpt});
                lev_source_dec({icol-col_s+1, ilay, igpt}) = sources_sub.get_lev_source_dec()({icol, ilay, igpt});
            }
}

#ifdef FLOAT_SINGLE_RRTMGP
template class Source_func_lw<float>;
#else
template class Source_func_lw<double>;
#endif
