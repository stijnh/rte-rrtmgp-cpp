/*
 * This file is imported from MicroHH (https://github.com/earth-system-radiation/earth-system-radiation)
 * and is adapted for the testing of the C++ interface to the
 * RTE+RRTMGP radiation code.
 *
 * MicroHH is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * MicroHH is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with MicroHH.  If not, see <http://www.gnu.org/licenses/>.
 */

#include <boost/algorithm/string.hpp>
#include <cmath>
#include <numeric>

#include "Radiation_solver.h"
#include "Status.h"
#include "Netcdf_interface.h"

#include "Array.h"
#include "Gas_concs.h"
#include "Gas_optics_rrtmgp.h"
#include "Optical_props.h"
#include "Source_functions.h"
#include "Fluxes.h"
#include "Rte_lw.h"
#include "Rte_sw.h"
#include "subset_kernel_launcher_cuda.h"
#include "Array_pool_gpu.h"

#include <cuda_profiler_api.h>
namespace
{
    template<typename TF>__global__
    void scaling_to_subset_kernel(
            const int ncol, const int ngpt, TF* __restrict__ toa_src, const TF* __restrict__ tsi_scaling)
    {
        const int icol = blockIdx.x*blockDim.x + threadIdx.x;
        const int igpt = blockIdx.y*blockDim.y + threadIdx.y;
        if ( ( icol < ncol) && (igpt < ngpt) )
        {
            const int idx = icol + igpt*ncol;
            toa_src[idx] *= tsi_scaling[icol];
        }
    }

    template<typename TF>
    void scaling_to_subset(
            const int ncol, const int ngpt, Array_gpu<TF,2>& toa_src, const Array_gpu<TF,1>& tsi_scaling)
    {
        const int block_col = 16;
        const int block_gpt = 16;
        const int grid_col  = ncol/block_col + (ncol%block_col > 0);
        const int grid_gpt  = ngpt/block_gpt + (ngpt%block_gpt > 0);
        dim3 grid_gpu(grid_col, grid_gpt);
        dim3 block_gpu(block_col, block_gpt);
        scaling_to_subset_kernel<<<grid_gpu, block_gpu>>>(
            ncol, ngpt, toa_src.ptr(), tsi_scaling.ptr());
    }


    std::vector<std::string> get_variable_string(
            const std::string& var_name,
            std::vector<int> i_count,
            Netcdf_handle& input_nc,
            const int string_len,
            bool trim=true)
    {
        // Multiply all elements in i_count.
        int total_count = std::accumulate(i_count.begin(), i_count.end(), 1, std::multiplies<>());

        // Add the string length as the rightmost dimension.
        i_count.push_back(string_len);

        // Read the entire char array;
        std::vector<char> var_char;
        var_char = input_nc.get_variable<char>(var_name, i_count);

        std::vector<std::string> var;

        for (int n=0; n<total_count; ++n)
        {
            std::string s(var_char.begin()+n*string_len, var_char.begin()+(n+1)*string_len);
            if (trim)
                boost::trim(s);
            var.push_back(s);
        }

        return var;
    }

    template<typename TF>
    Gas_optics_rrtmgp_gpu<TF> load_and_init_gas_optics(
            const Gas_concs_gpu<TF>& gas_concs,
            const std::string& coef_file)
    {
        // READ THE COEFFICIENTS FOR THE OPTICAL SOLVER.
        Netcdf_file coef_nc(coef_file, Netcdf_mode::Read);

        // Read k-distribution information.
        int n_temps = coef_nc.get_dimension_size("temperature");
        int n_press = coef_nc.get_dimension_size("pressure");
        int n_absorbers = coef_nc.get_dimension_size("absorber");

        // CvH: I hardcode the value to 32 now, because coef files
        // CvH: changed dimension name inconsistently.
        // int n_char = coef_nc.get_dimension_size("string_len");
        constexpr int n_char = 32;

        int n_minorabsorbers = coef_nc.get_dimension_size("minor_absorber");
        int n_extabsorbers = coef_nc.get_dimension_size("absorber_ext");
        int n_mixingfracs = coef_nc.get_dimension_size("mixing_fraction");
        int n_layers = coef_nc.get_dimension_size("atmos_layer");
        int n_bnds = coef_nc.get_dimension_size("bnd");
        int n_gpts = coef_nc.get_dimension_size("gpt");
        int n_pairs = coef_nc.get_dimension_size("pair");
        int n_minor_absorber_intervals_lower = coef_nc.get_dimension_size("minor_absorber_intervals_lower");
        int n_minor_absorber_intervals_upper = coef_nc.get_dimension_size("minor_absorber_intervals_upper");
        int n_contributors_lower = coef_nc.get_dimension_size("contributors_lower");
        int n_contributors_upper = coef_nc.get_dimension_size("contributors_upper");

        // Read gas names.
        Array<std::string,1> gas_names(
                get_variable_string("gas_names", {n_absorbers}, coef_nc, n_char, true), {n_absorbers});

        Array<int,3> key_species(
                coef_nc.get_variable<int>("key_species", {n_bnds, n_layers, 2}),
                {2, n_layers, n_bnds});
        Array<TF,2> band_lims(coef_nc.get_variable<TF>("bnd_limits_wavenumber", {n_bnds, 2}), {2, n_bnds});
        Array<int,2> band2gpt(coef_nc.get_variable<int>("bnd_limits_gpt", {n_bnds, 2}), {2, n_bnds});
        Array<TF,1> press_ref(coef_nc.get_variable<TF>("press_ref", {n_press}), {n_press});
        Array<TF,1> temp_ref(coef_nc.get_variable<TF>("temp_ref", {n_temps}), {n_temps});

        TF temp_ref_p = coef_nc.get_variable<TF>("absorption_coefficient_ref_P");
        TF temp_ref_t = coef_nc.get_variable<TF>("absorption_coefficient_ref_T");
        TF press_ref_trop = coef_nc.get_variable<TF>("press_ref_trop");

        Array<TF,3> kminor_lower(
                coef_nc.get_variable<TF>("kminor_lower", {n_temps, n_mixingfracs, n_contributors_lower}),
                {n_contributors_lower, n_mixingfracs, n_temps});
        Array<TF,3> kminor_upper(
                coef_nc.get_variable<TF>("kminor_upper", {n_temps, n_mixingfracs, n_contributors_upper}),
                {n_contributors_upper, n_mixingfracs, n_temps});

        Array<std::string,1> gas_minor(get_variable_string("gas_minor", {n_minorabsorbers}, coef_nc, n_char),
                                       {n_minorabsorbers});

        Array<std::string,1> identifier_minor(
                get_variable_string("identifier_minor", {n_minorabsorbers}, coef_nc, n_char), {n_minorabsorbers});

        Array<std::string,1> minor_gases_lower(
                get_variable_string("minor_gases_lower", {n_minor_absorber_intervals_lower}, coef_nc, n_char),
                {n_minor_absorber_intervals_lower});
        Array<std::string,1> minor_gases_upper(
                get_variable_string("minor_gases_upper", {n_minor_absorber_intervals_upper}, coef_nc, n_char),
                {n_minor_absorber_intervals_upper});

        Array<int,2> minor_limits_gpt_lower(
                coef_nc.get_variable<int>("minor_limits_gpt_lower", {n_minor_absorber_intervals_lower, n_pairs}),
                {n_pairs, n_minor_absorber_intervals_lower});
        Array<int,2> minor_limits_gpt_upper(
                coef_nc.get_variable<int>("minor_limits_gpt_upper", {n_minor_absorber_intervals_upper, n_pairs}),
                {n_pairs, n_minor_absorber_intervals_upper});

        Array<BOOL_TYPE,1> minor_scales_with_density_lower(
                coef_nc.get_variable<BOOL_TYPE>("minor_scales_with_density_lower", {n_minor_absorber_intervals_lower}),
                {n_minor_absorber_intervals_lower});
        Array<BOOL_TYPE,1> minor_scales_with_density_upper(
                coef_nc.get_variable<BOOL_TYPE>("minor_scales_with_density_upper", {n_minor_absorber_intervals_upper}),
                {n_minor_absorber_intervals_upper});

        Array<BOOL_TYPE,1> scale_by_complement_lower(
                coef_nc.get_variable<BOOL_TYPE>("scale_by_complement_lower", {n_minor_absorber_intervals_lower}),
                {n_minor_absorber_intervals_lower});
        Array<BOOL_TYPE,1> scale_by_complement_upper(
                coef_nc.get_variable<BOOL_TYPE>("scale_by_complement_upper", {n_minor_absorber_intervals_upper}),
                {n_minor_absorber_intervals_upper});

        Array<std::string,1> scaling_gas_lower(
                get_variable_string("scaling_gas_lower", {n_minor_absorber_intervals_lower}, coef_nc, n_char),
                {n_minor_absorber_intervals_lower});
        Array<std::string,1> scaling_gas_upper(
                get_variable_string("scaling_gas_upper", {n_minor_absorber_intervals_upper}, coef_nc, n_char),
                {n_minor_absorber_intervals_upper});

        Array<int,1> kminor_start_lower(
                coef_nc.get_variable<int>("kminor_start_lower", {n_minor_absorber_intervals_lower}),
                {n_minor_absorber_intervals_lower});
        Array<int,1> kminor_start_upper(
                coef_nc.get_variable<int>("kminor_start_upper", {n_minor_absorber_intervals_upper}),
                {n_minor_absorber_intervals_upper});

        Array<TF,3> vmr_ref(
                coef_nc.get_variable<TF>("vmr_ref", {n_temps, n_extabsorbers, n_layers}),
                {n_layers, n_extabsorbers, n_temps});

        Array<TF,4> kmajor(
                coef_nc.get_variable<TF>("kmajor", {n_temps, n_press+1, n_mixingfracs, n_gpts}),
                {n_gpts, n_mixingfracs, n_press+1, n_temps});

        // Keep the size at zero, if it does not exist.
        Array<TF,3> rayl_lower;
        Array<TF,3> rayl_upper;

        if (coef_nc.variable_exists("rayl_lower"))
        {
            rayl_lower.set_dims({n_gpts, n_mixingfracs, n_temps});
            rayl_upper.set_dims({n_gpts, n_mixingfracs, n_temps});
            rayl_lower = coef_nc.get_variable<TF>("rayl_lower", {n_temps, n_mixingfracs, n_gpts});
            rayl_upper = coef_nc.get_variable<TF>("rayl_upper", {n_temps, n_mixingfracs, n_gpts});
        }

        // Is it really LW if so read these variables as well.
        if (coef_nc.variable_exists("totplnk"))
        {
            int n_internal_sourcetemps = coef_nc.get_dimension_size("temperature_Planck");

            Array<TF,2> totplnk(
                    coef_nc.get_variable<TF>( "totplnk", {n_bnds, n_internal_sourcetemps}),
                    {n_internal_sourcetemps, n_bnds});
            Array<TF,4> planck_frac(
                    coef_nc.get_variable<TF>("plank_fraction", {n_temps, n_press+1, n_mixingfracs, n_gpts}),
                    {n_gpts, n_mixingfracs, n_press+1, n_temps});

            // Construct the k-distribution.
            return Gas_optics_rrtmgp_gpu<TF>(
                    gas_concs,
                    gas_names,
                    key_species,
                    band2gpt,
                    band_lims,
                    press_ref,
                    press_ref_trop,
                    temp_ref,
                    temp_ref_p,
                    temp_ref_t,
                    vmr_ref,
                    kmajor,
                    kminor_lower,
                    kminor_upper,
                    gas_minor,
                    identifier_minor,
                    minor_gases_lower,
                    minor_gases_upper,
                    minor_limits_gpt_lower,
                    minor_limits_gpt_upper,
                    minor_scales_with_density_lower,
                    minor_scales_with_density_upper,
                    scaling_gas_lower,
                    scaling_gas_upper,
                    scale_by_complement_lower,
                    scale_by_complement_upper,
                    kminor_start_lower,
                    kminor_start_upper,
                    totplnk,
                    planck_frac,
                    rayl_lower,
                    rayl_upper);
        }
        else
        {
            Array<TF,1> solar_src_quiet(
                    coef_nc.get_variable<TF>("solar_source_quiet", {n_gpts}), {n_gpts});
            Array<TF,1> solar_src_facular(
                    coef_nc.get_variable<TF>("solar_source_facular", {n_gpts}), {n_gpts});
            Array<TF,1> solar_src_sunspot(
                    coef_nc.get_variable<TF>("solar_source_sunspot", {n_gpts}), {n_gpts});

            TF tsi = coef_nc.get_variable<TF>("tsi_default");
            TF mg_index = coef_nc.get_variable<TF>("mg_default");
            TF sb_index = coef_nc.get_variable<TF>("sb_default");

            return Gas_optics_rrtmgp_gpu<TF>(
                    gas_concs,
                    gas_names,
                    key_species,
                    band2gpt,
                    band_lims,
                    press_ref,
                    press_ref_trop,
                    temp_ref,
                    temp_ref_p,
                    temp_ref_t,
                    vmr_ref,
                    kmajor,
                    kminor_lower,
                    kminor_upper,
                    gas_minor,
                    identifier_minor,
                    minor_gases_lower,
                    minor_gases_upper,
                    minor_limits_gpt_lower,
                    minor_limits_gpt_upper,
                    minor_scales_with_density_lower,
                    minor_scales_with_density_upper,
                    scaling_gas_lower,
                    scaling_gas_upper,
                    scale_by_complement_lower,
                    scale_by_complement_upper,
                    kminor_start_lower,
                    kminor_start_upper,
                    solar_src_quiet,
                    solar_src_facular,
                    solar_src_sunspot,
                    tsi,
                    mg_index,
                    sb_index,
                    rayl_lower,
                    rayl_upper);
        }
        // End reading of k-distribution.
    }

    template<typename TF>
    Cloud_optics_gpu<TF> load_and_init_cloud_optics(
            const std::string& coef_file)
    {
        // READ THE COEFFICIENTS FOR THE OPTICAL SOLVER.
        Netcdf_file coef_nc(coef_file, Netcdf_mode::Read);

        // Read look-up table coefficient dimensions
        int n_band     = coef_nc.get_dimension_size("nband");
        int n_rghice   = coef_nc.get_dimension_size("nrghice");
        int n_size_liq = coef_nc.get_dimension_size("nsize_liq");
        int n_size_ice = coef_nc.get_dimension_size("nsize_ice");

        Array<TF,2> band_lims_wvn(coef_nc.get_variable<TF>("bnd_limits_wavenumber", {n_band, 2}), {2, n_band});

        // Read look-up table constants.
        TF radliq_lwr = coef_nc.get_variable<TF>("radliq_lwr");
        TF radliq_upr = coef_nc.get_variable<TF>("radliq_upr");
        TF radliq_fac = coef_nc.get_variable<TF>("radliq_fac");

        TF radice_lwr = coef_nc.get_variable<TF>("radice_lwr");
        TF radice_upr = coef_nc.get_variable<TF>("radice_upr");
        TF radice_fac = coef_nc.get_variable<TF>("radice_fac");

        Array<TF,2> lut_extliq(
                coef_nc.get_variable<TF>("lut_extliq", {n_band, n_size_liq}), {n_size_liq, n_band});
        Array<TF,2> lut_ssaliq(
                coef_nc.get_variable<TF>("lut_ssaliq", {n_band, n_size_liq}), {n_size_liq, n_band});
        Array<TF,2> lut_asyliq(
                coef_nc.get_variable<TF>("lut_asyliq", {n_band, n_size_liq}), {n_size_liq, n_band});

        Array<TF,3> lut_extice(
                coef_nc.get_variable<TF>("lut_extice", {n_rghice, n_band, n_size_ice}), {n_size_ice, n_band, n_rghice});
        Array<TF,3> lut_ssaice(
                coef_nc.get_variable<TF>("lut_ssaice", {n_rghice, n_band, n_size_ice}), {n_size_ice, n_band, n_rghice});
        Array<TF,3> lut_asyice(
                coef_nc.get_variable<TF>("lut_asyice", {n_rghice, n_band, n_size_ice}), {n_size_ice, n_band, n_rghice});

        return Cloud_optics_gpu<TF>(
                band_lims_wvn,
                radliq_lwr, radliq_upr, radliq_fac,
                radice_lwr, radice_upr, radice_fac,
                lut_extliq, lut_ssaliq, lut_asyliq,
                lut_extice, lut_ssaice, lut_asyice);
    }
}


template<typename TF>
std::unique_ptr<Pool_base<TF*>> radiation_block_work_arrays_gpu<TF>::create_pool(
        const int ncols, 
        const int nlevs,
        const int nlays,
        const bool switch_fluxes,
        const bool switch_cloud_optics,
        const Radiation_solver_longwave<TF>* lws,
        const Radiation_solver_shortwave<TF>* sws)
{
    const int ngptmax = std::max(   lws == nullptr? 1 : lws->kdist_gpu->get_ngpt(), 
                                    sws == nullptr? 1 : sws->kdist_gpu->get_ngpt());
    const int nflavmax = std::max(  lws == nullptr? 1 : lws->kdist_gpu->get_nflav(), 
                                    sws == nullptr? 1 : sws->kdist_gpu->get_nflav());
    const int ngasmax = std::max(   lws == nullptr? 1 : lws->kdist_gpu->get_ngas(), 
                                    sws == nullptr? 1 : sws->kdist_gpu->get_ngas());

    const int nlevmax = std::max(nlevs, nlays);
    return std::unique_ptr<Pool_base<TF*>>(new Array_pool_gpu<TF>({
        ncols * ngptmax,
        ngptmax * nlevmax, 
        ncols * ngptmax * nlevmax, 
        2 * nflavmax * ncols * nlays,
        4 * nflavmax * ncols * nlays,
        8 * nflavmax * ncols * nlays,
        ncols * nlays * ngasmax,
        ncols * nlays * (ngasmax + 1)}));
}


template<typename TF>
radiation_block_work_arrays_gpu<TF>::radiation_block_work_arrays_gpu(
        const int ncols,
        const int nlevs,
        const int nlays,
        const bool switch_fluxes,
        const bool switch_bnd_fluxes,
        const bool switch_cloud_optics,
        const Gas_concs_gpu<TF>& gas_concs,
        const Radiation_solver_longwave<TF>* lws,
        const Radiation_solver_shortwave<TF>* sws,
        const bool recursive):
//        memory_pool(create_pool(ncols, nlevs, nlays, switch_fluxes, switch_cloud_optics, lws, sws)),
        col_dry_subset({ncols, nlays}),
        delta_plev_subset({ncols, nlays}),
        m_air_subset({ncols, nlays}),
        p_lev_subset({ncols, nlevs}),
        p_lay_subset({ncols, nlays}),
        t_lev_subset({ncols, nlevs}),
        t_lay_subset({ncols, nlays}),
        t_sfc_subset({ncols}),
        gas_concs_subset(std::make_unique<Gas_concs_gpu<TF>>(gas_concs, 1, ncols))
{
    if(switch_cloud_optics)
    {
        lwp_lay_subset = Array_gpu<TF,2>({ncols, nlays});
        iwp_lay_subset = Array_gpu<TF,2>({ncols, nlays});
        rel_lay_subset = Array_gpu<TF,2>({ncols, nlays});
        rei_lay_subset = Array_gpu<TF,2>({ncols, nlays});
    }
    
    fluxes_subset = std::make_unique<Fluxes_broadband_gpu<TF>>(ncols, nlevs);

    if(sws == nullptr)
    {
        if(lws != nullptr)
        {
            allocate_lw_data(ncols, nlevs, nlays, switch_fluxes, switch_bnd_fluxes, 
                            switch_cloud_optics, lws, recursive);
        }
    }
    else
    {
        if(lws == nullptr)
        {
            allocate_sw_data(ncols, nlevs, nlays, switch_fluxes, switch_bnd_fluxes,
                            switch_cloud_optics, sws, recursive);
        }
        else
        {
            allocate_lw_data(ncols, nlevs, nlays, switch_fluxes, switch_bnd_fluxes, 
                            switch_cloud_optics, lws, recursive);
            allocate_sw_data(ncols, nlevs, nlays, switch_fluxes, switch_bnd_fluxes, 
                            switch_cloud_optics, sws, recursive);
        }
    }
    
    if(memory_pool)
    {
        static_cast<Array_pool_gpu<TF>*>(memory_pool.get())->lock = true;
    }
}

template<typename TF>
void radiation_block_work_arrays_gpu<TF>::allocate_lw_data(
        const int ncols, 
        const int nlevs,
        const int nlays,
        const bool switch_fluxes,
        const bool switch_bnd_fluxes,
        const bool switch_cloud_optics,
        const Radiation_solver_longwave<TF>* lws,
        const bool recursive)
{
    int ngpt = lws->kdist_gpu->get_ngpt();
    int nbnd = lws->kdist_gpu->get_nband();
    int ngas = lws->kdist_gpu->get_ngas();
    int nflav = lws->kdist_gpu->get_nflav();
    auto pool = memory_pool.get();

    emis_sfc_subset = Array_gpu<TF,2>({nbnd, ncols});
    lw_optical_props_subset = std::make_unique<Optical_props_1scl_gpu<TF>>(ncols, nlays, *(lws->kdist_gpu), pool);
    sources_subset = std::make_unique<Source_func_lw_gpu<TF>>(ncols, nlays, *(lws->kdist_gpu), pool);

    if(recursive)
    {
        lw_gas_optics_work = std::make_unique<gas_optics_work_arrays_gpu<TF>>(ncols, nlays, nflav, pool);
        lw_gas_optics_work->tau_work_arrays = 
            std::make_unique<gas_taus_work_arrays_gpu<TF>>(ncols, nlays, ngpt, ngas, nflav, pool);
        lw_gas_optics_work->tau_work_arrays->release_memory();
        lw_gas_optics_work->source_work_arrays = 
            std::make_unique<gas_source_work_arrays_gpu<TF>>(ncols, nlays, ngpt, pool);
        lw_gas_optics_work->source_work_arrays->release_memory();
        lw_gas_optics_work->release_memory();
    }
    //TODO: Make use of pool
    if(switch_cloud_optics)
    {
        lw_cloud_optical_props_subset = 
                std::make_unique<Optical_props_1scl_gpu<TF>>(ncols, nlays, *(lws->cloud_optics_gpu));
    }

    if(switch_fluxes)
    {
        lw_gpt_flux_up = std::make_unique<Array_gpu<TF,3>>(std::array<int,3>({ncols, nlevs, ngpt}), pool);
        lw_gpt_flux_dn = std::make_unique<Array_gpu<TF,3>>(std::array<int,3>({ncols, nlevs, ngpt}), pool);
        lw_gpt_flux_up->release_memory();
        lw_gpt_flux_dn->release_memory();

        if(recursive)
        {
            rte_lw_work = std::make_unique<rte_lw_work_arrays_gpu<TF>>(ncols, nlevs, nlays, ngpt, pool);
            rte_lw_work->release_memory();
        }
    }

    if(switch_bnd_fluxes)
    {
        lw_bnd_fluxes_subset = std::make_unique<Fluxes_byband_gpu<TF>>(ncols, nlevs, nbnd, pool);
        lw_bnd_fluxes_subset->release_memory();
    }

    lw_optical_props_subset->release_memory();
    sources_subset->release_memory();
}


template<typename TF>
void radiation_block_work_arrays_gpu<TF>::allocate_sw_data(
        const int ncols, 
        const int nlevs,
        const int nlays,
        const bool switch_fluxes,
        const bool switch_bnd_fluxes,
        const bool switch_cloud_optics,
        const Radiation_solver_shortwave<TF>* sws,
        const bool recursive)
{
    int ngpt = sws->kdist_gpu->get_ngpt();
    int nbnd = sws->kdist_gpu->get_nband();
    int ngas = sws->kdist_gpu->get_ngas();
    int nflav = sws->kdist_gpu->get_nflav();
    auto pool = memory_pool.get();

    sw_optical_props_subset = std::make_unique<Optical_props_2str_gpu<TF>>(ncols, nlays, *(sws->kdist_gpu), pool);
    toa_src_subset = std::make_unique<Array_gpu<TF,2>>(std::array<int,2>({ncols, ngpt}), pool);
    tsi_scaling_subset = Array<TF,1>({ncols});

    if(recursive)
    {
        sw_gas_optics_work = std::make_unique<gas_optics_work_arrays_gpu<TF>>(ncols, nlays, nflav, pool);
        sw_gas_optics_work->tau_work_arrays = 
            std::make_unique<gas_taus_work_arrays_gpu<TF>>(ncols, nlays, ngpt, ngas, nflav, pool);
        sw_gas_optics_work->tau_work_arrays->release_memory();
        sw_gas_optics_work->release_memory();        
    }

    //TODO: Make use of pool
    if(switch_cloud_optics)
    {
        sw_cloud_optical_props_subset = 
                std::make_unique<Optical_props_2str_gpu<TF>>(ncols, nlays, *(sws->cloud_optics_gpu));
    }

    if(switch_fluxes)
    {
        mu0_subset = Array_gpu<TF,1>({ncols});
        sfc_alb_dir_subset = Array_gpu<TF,2>({nbnd, ncols}, pool);
        sfc_alb_dif_subset = Array_gpu<TF,2>({nbnd, ncols}, pool);

        sw_gpt_flux_up = std::make_unique<Array_gpu<TF,3>>(std::array<int,3>({ncols, nlevs, ngpt}), pool);
        sw_gpt_flux_dn = std::make_unique<Array_gpu<TF,3>>(std::array<int,3>({ncols, nlevs, ngpt}), pool);
        sw_gpt_flux_dn_dir = std::make_unique<Array_gpu<TF,3>>(std::array<int,3>({ncols, nlevs, ngpt}), pool);
        sw_gpt_flux_up->release_memory();
        sw_gpt_flux_dn->release_memory();
        sw_gpt_flux_dn_dir->release_memory();

        if(recursive)
        {
            rte_sw_work = std::make_unique<rte_sw_work_arrays_gpu<TF>>(ncols, ngpt, pool);
            rte_sw_work->release_memory();
        }
    }

    if(switch_bnd_fluxes)
    {
        sw_bnd_fluxes_subset = std::make_unique<Fluxes_byband_gpu<TF>>(ncols, nlevs, nbnd, pool);
        sw_bnd_fluxes_subset->release_memory();
    }

    sw_optical_props_subset->release_memory();
    toa_src_subset->release_memory();
}


template<typename TF>
radiation_solver_work_arrays_gpu<TF>::radiation_solver_work_arrays_gpu(){}


template<typename TF>
radiation_solver_work_arrays_gpu<TF>::radiation_solver_work_arrays_gpu(
        const int ncols, 
        const int nlevs, 
        const int nlays,
        const bool switch_fluxes,
        const bool switch_bnd_fluxes,
        const bool switch_cloud_optics,
        const Gas_concs_gpu<TF>& gas_concs,
        const Radiation_solver_longwave<TF>* lws,
        const Radiation_solver_shortwave<TF>* sws)
{
    blocks_work_arrays = std::make_unique<radiation_block_work_arrays_gpu<TF>>(
        Radiation_solver_longwave<TF>::n_col_block, nlevs, nlays,
        switch_fluxes, switch_bnd_fluxes, switch_cloud_optics,
        gas_concs, lws, sws);
    int n_col_block_residual = ncols % Radiation_solver_longwave<TF>::n_col_block;
    if(n_col_block_residual > 0)
    {    
        residual_work_arrays = std::make_unique<radiation_block_work_arrays_gpu<TF>>(
        n_col_block_residual, nlevs, nlays,
        switch_fluxes, switch_bnd_fluxes, switch_cloud_optics,
        gas_concs, lws, sws);
    }
}


template<typename TF>
Radiation_solver_longwave<TF>::Radiation_solver_longwave(
        const Gas_concs_gpu<TF>& gas_concs,
        const std::string& file_name_gas,
        const std::string& file_name_cloud)
{
    // Construct the gas optics classes for the solver.
    this->kdist_gpu = std::make_unique<Gas_optics_rrtmgp_gpu<TF>>(
            load_and_init_gas_optics<TF>(gas_concs, file_name_gas));

    this->cloud_optics_gpu = std::make_unique<Cloud_optics_gpu<TF>>(
            load_and_init_cloud_optics<TF>(file_name_cloud));
}


template<typename TF>
void Radiation_solver_longwave<TF>::solve_gpu(
        const bool switch_fluxes,
        const bool switch_cloud_optics,
        const bool switch_output_optical,
        const bool switch_output_bnd_fluxes,
        const Gas_concs_gpu<TF>& gas_concs,
        const Array_gpu<TF,2>& p_lay, const Array_gpu<TF,2>& p_lev,
        const Array_gpu<TF,2>& t_lay, const Array_gpu<TF,2>& t_lev,
        const Array_gpu<TF,2>& col_dry,
        const Array_gpu<TF,1>& t_sfc, const Array_gpu<TF,2>& emis_sfc,
        const Array_gpu<TF,2>& lwp, const Array_gpu<TF,2>& iwp,
        const Array_gpu<TF,2>& rel, const Array_gpu<TF,2>& rei,
        Array_gpu<TF,3>& tau, Array_gpu<TF,3>& lay_source,
        Array_gpu<TF,3>& lev_source_inc, Array_gpu<TF,3>& lev_source_dec, Array_gpu<TF,2>& sfc_source,
        Array_gpu<TF,2>& lw_flux_up, Array_gpu<TF,2>& lw_flux_dn, Array_gpu<TF,2>& lw_flux_net,
        Array_gpu<TF,3>& lw_bnd_flux_up, Array_gpu<TF,3>& lw_bnd_flux_dn, Array_gpu<TF,3>& lw_bnd_flux_net,
        radiation_solver_work_arrays_gpu<TF>* work_arrays) const
{
    const int n_col = p_lay.dim(1);
    const int n_lay = p_lay.dim(2);
    const int n_lev = p_lev.dim(2);
    const int n_gpt = this->kdist_gpu->get_ngpt();
    const int n_bnd = this->kdist_gpu->get_nband();

    const BOOL_TYPE top_at_1 = p_lay({1, 1}) < p_lay({1, n_lay});

    // Lambda function for solving optical properties subset.
    auto call_kernels = [&](
        const int col_s_in, const int col_e_in,
        radiation_block_work_arrays_gpu<TF>* work_block)
    {
        const int n_col_in = col_e_in - col_s_in + 1;

        auto work = work_block;
        std::unique_ptr<radiation_block_work_arrays_gpu<TF>> allocated_work_arrays;
        if(work_block == nullptr)
        {
            allocated_work_arrays = std::make_unique<radiation_block_work_arrays_gpu<TF>>(
                    n_col_in, n_lev, n_lay, switch_fluxes, switch_output_bnd_fluxes, 
                    switch_cloud_optics, gas_concs, this, nullptr, false);
            work = allocated_work_arrays.get();
        }

        work->gas_concs_subset->subset_copy(gas_concs, col_s_in);

        p_lev.subset_copy(work->p_lev_subset, {col_s_in, 1});
        p_lay.subset_copy(work->p_lay_subset, {col_s_in, 1});
        t_lev.subset_copy(work->t_lev_subset, {col_s_in, 1});
        t_lay.subset_copy(work->t_lay_subset, {col_s_in, 1});
        t_sfc.subset_copy(work->t_sfc_subset, {col_s_in});
        emis_sfc.subset_copy(work->emis_sfc_subset, {1, col_s_in});

        if (col_dry.size() == 0)
            Gas_optics_rrtmgp_gpu<TF>::get_col_dry(
                    work->col_dry_subset, 
                    work->delta_plev_subset,
                    work->m_air_subset,    
                    work->gas_concs_subset->get_vmr("h2o"), 
                    work->p_lev_subset);
        else
            col_dry.subset_copy(work->col_dry_subset, {col_s_in, 1});

        work->lw_optical_props_subset->acquire_memory();
        work->sources_subset->acquire_memory();
        if(work->lw_gas_optics_work)
        {
            work->lw_gas_optics_work->acquire_memory();
        }
    
        kdist_gpu->gas_optics(
                work->p_lay_subset,
                work->p_lev_subset,
                work->t_lay_subset,
                work->t_sfc_subset,
                *(work->gas_concs_subset),
                work->lw_optical_props_subset,
                *(work->sources_subset),
                work->col_dry_subset,
                work->t_lev_subset,
                work->lw_gas_optics_work.get());

        if(work->lw_gas_optics_work)
        {
            work->lw_gas_optics_work->release_memory();
        }
        
        if (switch_cloud_optics)
        {
            lwp.subset_copy(work->lwp_lay_subset, {col_s_in, 1});
            iwp.subset_copy(work->iwp_lay_subset, {col_s_in, 1});
            rel.subset_copy(work->rel_lay_subset, {col_s_in, 1});
            rei.subset_copy(work->rei_lay_subset, {col_s_in, 1});
            cloud_optics_gpu->cloud_optics(
                work->lwp_lay_subset,
                work->iwp_lay_subset,
                work->rel_lay_subset,
                work->rei_lay_subset,
                *(work->lw_cloud_optical_props_subset));

            // cloud->delta_scale();

            // Add the cloud optical props to the gas optical properties.
            add_to(
                    dynamic_cast<Optical_props_1scl_gpu<TF>&>(*(work->lw_optical_props_subset)),
                    dynamic_cast<Optical_props_1scl_gpu<TF>&>(*(work->lw_cloud_optical_props_subset)));
        }

        // Store the optical properties, if desired.
        if (switch_output_optical)
        {
            subset_kernel_launcher_cuda::get_from_subset(
                    n_col, n_lay, n_gpt, n_col_in, col_s_in, tau, lay_source, lev_source_inc, lev_source_dec,
                    work->lw_optical_props_subset->get_tau(), work->sources_subset->get_lay_source(),
                    work->sources_subset->get_lev_source_inc(), work->sources_subset->get_lev_source_dec());

            subset_kernel_launcher_cuda::get_from_subset(
                    n_col, n_gpt, n_col_in, col_s_in, sfc_source, work->sources_subset->get_sfc_source());
        }

        if (!switch_fluxes)
            return;

        constexpr int n_ang = 1;

        work->lw_gpt_flux_up->acquire_memory();
        work->lw_gpt_flux_dn->acquire_memory();
        if(work->rte_lw_work)
        {
            work->rte_lw_work->acquire_memory();
        }

        Rte_lw_gpu<TF>::rte_lw(
                work->lw_optical_props_subset,
                top_at_1,
                *(work->sources_subset),
                work->emis_sfc_subset,
                Array_gpu<TF,2>(), // Add an empty array, no inc_flux.
                *(work->lw_gpt_flux_up), *(work->lw_gpt_flux_dn),
                n_ang, work->rte_lw_work.get());

        work->fluxes_subset->reduce(*(work->lw_gpt_flux_up), *(work->lw_gpt_flux_dn), 
                                        work->lw_optical_props_subset, top_at_1);

        // Copy the data to the output.
        subset_kernel_launcher_cuda::get_from_subset(
                n_col, n_lev, n_col_in, col_s_in, lw_flux_up, lw_flux_dn, lw_flux_net,
                work->fluxes_subset->get_flux_up(), work->fluxes_subset->get_flux_dn(), 
                work->fluxes_subset->get_flux_net());


        if (switch_output_bnd_fluxes)
        {
            work->lw_bnd_fluxes_subset->acquire_memory();

            work->lw_bnd_fluxes_subset->reduce(*(work->lw_gpt_flux_up), *(work->lw_gpt_flux_dn), 
                                                work->lw_optical_props_subset, top_at_1);

            subset_kernel_launcher_cuda::get_from_subset(
                    n_col, n_lev, n_bnd, n_col_in, col_s_in, lw_bnd_flux_up, lw_bnd_flux_dn, lw_bnd_flux_net,
                    work->lw_bnd_fluxes_subset->get_bnd_flux_up(), 
                    work->lw_bnd_fluxes_subset->get_bnd_flux_dn(), 
                    work->lw_bnd_fluxes_subset->get_bnd_flux_net());

            work->lw_bnd_fluxes_subset->release_memory();
        }

        work->lw_gpt_flux_up->release_memory();
        work->lw_gpt_flux_dn->release_memory();
        work->lw_optical_props_subset->release_memory();
    };

    // Read the sources and create containers for the substeps.
    int n_blocks = n_col / n_col_block;
    
    for (int b=1; b<=n_blocks; ++b)
    {
        const int col_s = (b-1) * n_col_block + 1;
        const int col_e =  b    * n_col_block;

        call_kernels(col_s, col_e, work_arrays == nullptr ? nullptr : work_arrays->blocks_work_arrays.get());
    }

    int n_col_block_residual = n_col % n_col_block;

    if (n_col_block_residual > 0)
    {
        const int col_s = n_col - n_col_block_residual + 1;
        const int col_e = n_col;

        call_kernels(col_s, col_e, work_arrays == nullptr? nullptr : work_arrays->residual_work_arrays.get());
    }
}

template<typename TF>
Radiation_solver_shortwave<TF>::Radiation_solver_shortwave(
        const Gas_concs_gpu<TF>& gas_concs,
        const std::string& file_name_gas,
        const std::string& file_name_cloud)
{
    // Construct the gas optics classes for the solver.
    this->kdist_gpu = std::make_unique<Gas_optics_rrtmgp_gpu<TF>>(
            load_and_init_gas_optics<TF>(gas_concs, file_name_gas));

    this->cloud_optics_gpu = std::make_unique<Cloud_optics_gpu<TF>>(
            load_and_init_cloud_optics<TF>(file_name_cloud));
}


template<typename TF>
void Radiation_solver_shortwave<TF>::solve_gpu(
        const bool switch_fluxes,
        const bool switch_cloud_optics,
        const bool switch_output_optical,
        const bool switch_output_bnd_fluxes,
        const Gas_concs_gpu<TF>& gas_concs,
        const Array_gpu<TF,2>& p_lay, const Array_gpu<TF,2>& p_lev,
        const Array_gpu<TF,2>& t_lay, const Array_gpu<TF,2>& t_lev,
        const Array_gpu<TF,2>& col_dry,
        const Array_gpu<TF,2>& sfc_alb_dir, const Array_gpu<TF,2>& sfc_alb_dif,
        const Array_gpu<TF,1>& tsi_scaling, const Array_gpu<TF,1>& mu0,
        const Array_gpu<TF,2>& lwp, const Array_gpu<TF,2>& iwp,
        const Array_gpu<TF,2>& rel, const Array_gpu<TF,2>& rei,
        Array_gpu<TF,3>& tau, Array_gpu<TF,3>& ssa, Array_gpu<TF,3>& g,
        Array_gpu<TF,2>& toa_src,
        Array_gpu<TF,2>& sw_flux_up, Array_gpu<TF,2>& sw_flux_dn,
        Array_gpu<TF,2>& sw_flux_dn_dir, Array_gpu<TF,2>& sw_flux_net,
        Array_gpu<TF,3>& sw_bnd_flux_up, Array_gpu<TF,3>& sw_bnd_flux_dn,
        Array_gpu<TF,3>& sw_bnd_flux_dn_dir, Array_gpu<TF,3>& sw_bnd_flux_net,
        radiation_solver_work_arrays_gpu<TF>* work_arrays) const
{
    const int n_col = p_lay.dim(1);
    const int n_lay = p_lay.dim(2);
    const int n_lev = p_lev.dim(2);
    const int n_gpt = this->kdist_gpu->get_ngpt();
    const int n_bnd = this->kdist_gpu->get_nband();

    const BOOL_TYPE top_at_1 = p_lay({1, 1}) < p_lay({1, n_lay});

    // Lambda function for solving optical properties subset.
    auto call_kernels = [&](
        const int col_s_in, const int col_e_in,
        radiation_block_work_arrays_gpu<TF>* work_block)
    {
        const int n_col_in = col_e_in - col_s_in + 1;

        auto work = work_block;
        std::unique_ptr<radiation_block_work_arrays_gpu<TF>> allocated_work_arrays;
        if(work_block == nullptr)
        {
            allocated_work_arrays = std::make_unique<radiation_block_work_arrays_gpu<TF>>(
                    n_col_in, n_lev, n_lay, switch_fluxes, switch_output_bnd_fluxes, 
                    switch_cloud_optics, gas_concs, nullptr, this, false);
            work = allocated_work_arrays.get();
        }

        work->gas_concs_subset->subset_copy(gas_concs, col_s_in);

        p_lev.subset_copy(work->p_lev_subset, {col_s_in, 1});
        p_lay.subset_copy(work->p_lay_subset, {col_s_in, 1});
        t_lay.subset_copy(work->t_lay_subset, {col_s_in, 1});

        if (col_dry.size() == 0)
            Gas_optics_rrtmgp_gpu<TF>::get_col_dry(
                    work->col_dry_subset, 
                    work->delta_plev_subset,
                    work->m_air_subset,    
                    work->gas_concs_subset->get_vmr("h2o"), 
                    work->p_lev_subset);
        else
            col_dry.subset_copy(work->col_dry_subset, {col_s_in, 1});

        work->sw_optical_props_subset->acquire_memory();
        work->toa_src_subset->acquire_memory();
        if(work->sw_gas_optics_work)
        {
            work->sw_gas_optics_work->acquire_memory();
        }

        kdist_gpu->gas_optics(
                work->p_lay_subset,
                work->p_lev_subset,
                work->t_lay_subset,
                *(work->gas_concs_subset),
                work->sw_optical_props_subset,
                *(work->toa_src_subset),
                work->col_dry_subset,
                work->sw_gas_optics_work.get());

        if(work->sw_gas_optics_work)
        {
                work->sw_gas_optics_work->release_memory();
        }

        auto tsi_scaling_subset = tsi_scaling.subset_copy(work->tsi_scaling_subset, {col_s_in});

        scaling_to_subset(n_col_in, n_gpt, *(work->toa_src_subset), tsi_scaling_subset);

        if (switch_cloud_optics)
        {
            lwp.subset_copy(work->lwp_lay_subset, {col_s_in, 1});
            iwp.subset_copy(work->iwp_lay_subset, {col_s_in, 1});
            rel.subset_copy(work->rel_lay_subset, {col_s_in, 1});
            rei.subset_copy(work->rei_lay_subset, {col_s_in, 1});
            
            cloud_optics_gpu->cloud_optics(
                    work->lwp_lay_subset,
                    work->iwp_lay_subset,
                    work->rel_lay_subset,
                    work->rei_lay_subset,
                    *(work->sw_cloud_optical_props_subset));

            work->sw_cloud_optical_props_subset->delta_scale();

            // Add the cloud optical props to the gas optical properties.
            add_to(
                dynamic_cast<Optical_props_2str<TF>&>(*(work->sw_optical_props_subset)),
                dynamic_cast<Optical_props_2str<TF>&>(*(work->sw_cloud_optical_props_subset)));
        }

        // Store the optical properties, if desired.
        if (switch_output_optical)
        {
            subset_kernel_launcher_cuda::get_from_subset(
                    n_col, n_lay, n_gpt, n_col_in, col_s_in, tau, ssa, g,
                    work->sw_optical_props_subset->get_tau(),
                    work->sw_optical_props_subset->get_ssa(),
                    work->sw_optical_props_subset->get_g());

            subset_kernel_launcher_cuda::get_from_subset(
                    n_col, n_gpt, n_col_in, col_s_in, toa_src, *work->toa_src_subset);
        }
        if (!switch_fluxes)
            return;

        mu0.subset_copy(work->mu0_subset, {col_s_in});
        sfc_alb_dir.subset_copy(work->sfc_alb_dir_subset, {1, col_s_in});
        sfc_alb_dif.subset_copy(work->sfc_alb_dif_subset, {1, col_s_in});
            
        work->sw_gpt_flux_up->acquire_memory();
        work->sw_gpt_flux_dn->acquire_memory();
        work->sw_gpt_flux_dn_dir->acquire_memory();
    
        if(work->rte_sw_work)
        {
            work->rte_sw_work->acquire_memory();
        }
        Rte_sw_gpu<TF>::rte_sw(
                work->sw_optical_props_subset,
                top_at_1,
                work->mu0_subset,
                *(work->toa_src_subset),
                work->sfc_alb_dir_subset,
                work->sfc_alb_dif_subset,
                Array_gpu<TF,2>(), // Add an empty array, no inc_flux.
                *(work->sw_gpt_flux_up),
                *(work->sw_gpt_flux_dn),
                *(work->sw_gpt_flux_dn_dir),
                work->rte_sw_work.get());
        
        if(work->rte_sw_work)
        {
            work->rte_sw_work->release_memory();
        }
        
        work->fluxes_subset->reduce(
                *(work->sw_gpt_flux_up), 
                *(work->sw_gpt_flux_dn), 
                *(work->sw_gpt_flux_dn_dir), 
                work->sw_optical_props_subset, 
                top_at_1);

        // Copy the data to the output.
        subset_kernel_launcher_cuda::get_from_subset(
                n_col, n_lev, n_col_in, col_s_in, sw_flux_up, sw_flux_dn, sw_flux_dn_dir, sw_flux_net,
                work->fluxes_subset->get_flux_up(), 
                work->fluxes_subset->get_flux_dn(),
                work->fluxes_subset->get_flux_dn_dir(),
                work->fluxes_subset->get_flux_net());

        if (switch_output_bnd_fluxes)
        {
            work->sw_bnd_fluxes_subset->acquire_memory();

            work->sw_bnd_fluxes_subset->reduce(
                *(work->sw_gpt_flux_up), 
                *(work->sw_gpt_flux_dn), 
                *(work->sw_gpt_flux_dn_dir), 
                work->sw_optical_props_subset, 
                top_at_1);

            //work->sw_bnd_fluxes_subset->reduce(gpt_flux_up, gpt_flux_dn, optical_props_subset_in, top_at_1);

            subset_kernel_launcher_cuda::get_from_subset(
                    n_col, n_lev, n_bnd, n_col_in, col_s_in, 
                    sw_bnd_flux_up, 
                    sw_bnd_flux_dn, 
                    sw_bnd_flux_dn_dir, 
                    sw_bnd_flux_net,
                    work->sw_bnd_fluxes_subset->get_bnd_flux_up(), 
                    work->sw_bnd_fluxes_subset->get_bnd_flux_dn(), 
                    work->sw_bnd_fluxes_subset->get_bnd_flux_dn_dir(),
                    work->sw_bnd_fluxes_subset->get_bnd_flux_net());
                    
            work->sw_bnd_fluxes_subset->release_memory();
        }

        work->toa_src_subset->release_memory();
        work->sw_gpt_flux_up->release_memory();
        work->sw_gpt_flux_dn->release_memory();
        work->sw_gpt_flux_dn_dir->release_memory();
        work->sw_optical_props_subset->release_memory();
    };


    // Read the sources and create containers for the substeps.
    int n_blocks = n_col / n_col_block;
    
    for (int b=1; b<=n_blocks; ++b)
    {
        const int col_s = (b-1) * n_col_block + 1;
        const int col_e =  b    * n_col_block;

        call_kernels(col_s, col_e, work_arrays == nullptr ? nullptr : work_arrays->blocks_work_arrays.get());
    }

    int n_col_block_residual = n_col % n_col_block;

    if (n_col_block_residual > 0)
    {
        const int col_s = n_col - n_col_block_residual + 1;
        const int col_e = n_col;

        call_kernels(col_s, col_e, work_arrays == nullptr? nullptr : work_arrays->residual_work_arrays.get());
    }
}

#ifdef FLOAT_SINGLE_RRTMGP
template class Radiation_solver_longwave<float>;
template class Radiation_solver_shortwave<float>;
template class radiation_block_work_arrays_gpu<float>;
template class radiation_solver_work_arrays_gpu<float>;
#else
template class Radiation_solver_longwave<double>;
template class Radiation_solver_shortwave<double>;
template class radiation_block_work_arrays_gpu<double>;
template class radiation_solver_work_arrays_gpu<double>;
#endif
