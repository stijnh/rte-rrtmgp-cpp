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


namespace
{
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

    Gas_optics_rrtmgp load_and_init_gas_optics(
            const Gas_concs& gas_concs,
            const std::string& coef_file)
    {
        // READ THE COEFFICIENTS FOR THE OPTICAL SOLVER.
        Netcdf_file coef_nc(coef_file, Netcdf_mode::Read);

        // Read k-distribution information.
        const int n_temps = coef_nc.get_dimension_size("temperature");
        const int n_press = coef_nc.get_dimension_size("pressure");
        const int n_absorbers = coef_nc.get_dimension_size("absorber");
        const int n_char = coef_nc.get_dimension_size("string_len");
        const int n_minorabsorbers = coef_nc.get_dimension_size("minor_absorber");
        const int n_extabsorbers = coef_nc.get_dimension_size("absorber_ext");
        const int n_mixingfracs = coef_nc.get_dimension_size("mixing_fraction");
        const int n_layers = coef_nc.get_dimension_size("atmos_layer");
        const int n_bnds = coef_nc.get_dimension_size("bnd");
        const int n_gpts = coef_nc.get_dimension_size("gpt");
        const int n_pairs = coef_nc.get_dimension_size("pair");
        const int n_minor_absorber_intervals_lower = coef_nc.get_dimension_size("minor_absorber_intervals_lower");
        const int n_minor_absorber_intervals_upper = coef_nc.get_dimension_size("minor_absorber_intervals_upper");
        const int n_contributors_lower = coef_nc.get_dimension_size("contributors_lower");
        const int n_contributors_upper = coef_nc.get_dimension_size("contributors_upper");

        // Read gas names.
        Array<std::string,1> gas_names(
                get_variable_string("gas_names", {n_absorbers}, coef_nc, n_char, true), {n_absorbers});

        Array<int,3> key_species(
                coef_nc.get_variable<int>("key_species", {n_bnds, n_layers, 2}),
                {2, n_layers, n_bnds});
        Array<Float,2> band_lims(coef_nc.get_variable<Float>("bnd_limits_wavenumber", {n_bnds, 2}), {2, n_bnds});
        Array<int,2> band2gpt(coef_nc.get_variable<int>("bnd_limits_gpt", {n_bnds, 2}), {2, n_bnds});
        Array<Float,1> press_ref(coef_nc.get_variable<Float>("press_ref", {n_press}), {n_press});
        Array<Float,1> temp_ref(coef_nc.get_variable<Float>("temp_ref", {n_temps}), {n_temps});

        Float temp_ref_p = coef_nc.get_variable<Float>("absorption_coefficient_ref_P");
        Float temp_ref_t = coef_nc.get_variable<Float>("absorption_coefficient_ref_T");
        Float press_ref_trop = coef_nc.get_variable<Float>("press_ref_trop");

        Array<Float,3> kminor_lower(
                coef_nc.get_variable<Float>("kminor_lower", {n_temps, n_mixingfracs, n_contributors_lower}),
                {n_contributors_lower, n_mixingfracs, n_temps});
        Array<Float,3> kminor_upper(
                coef_nc.get_variable<Float>("kminor_upper", {n_temps, n_mixingfracs, n_contributors_upper}),
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

        Array<Bool,1> minor_scales_with_density_lower(
                coef_nc.get_variable<Bool>("minor_scales_with_density_lower", {n_minor_absorber_intervals_lower}),
                {n_minor_absorber_intervals_lower});
        Array<Bool,1> minor_scales_with_density_upper(
                coef_nc.get_variable<Bool>("minor_scales_with_density_upper", {n_minor_absorber_intervals_upper}),
                {n_minor_absorber_intervals_upper});

        Array<Bool,1> scale_by_complement_lower(
                coef_nc.get_variable<Bool>("scale_by_complement_lower", {n_minor_absorber_intervals_lower}),
                {n_minor_absorber_intervals_lower});
        Array<Bool,1> scale_by_complement_upper(
                coef_nc.get_variable<Bool>("scale_by_complement_upper", {n_minor_absorber_intervals_upper}),
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

        Array<Float,3> vmr_ref(
                coef_nc.get_variable<Float>("vmr_ref", {n_temps, n_extabsorbers, n_layers}),
                {n_layers, n_extabsorbers, n_temps});

        Array<Float,4> kmajor(
                coef_nc.get_variable<Float>("kmajor", {n_temps, n_press+1, n_mixingfracs, n_gpts}),
                {n_gpts, n_mixingfracs, n_press+1, n_temps});

        // Keep the size at zero, if it does not exist.
        Array<Float,3> rayl_lower;
        Array<Float,3> rayl_upper;

        if (coef_nc.variable_exists("rayl_lower"))
        {
            rayl_lower.set_dims({n_gpts, n_mixingfracs, n_temps});
            rayl_upper.set_dims({n_gpts, n_mixingfracs, n_temps});
            rayl_lower = coef_nc.get_variable<Float>("rayl_lower", {n_temps, n_mixingfracs, n_gpts});
            rayl_upper = coef_nc.get_variable<Float>("rayl_upper", {n_temps, n_mixingfracs, n_gpts});
        }

        // Is it really LW if so read these variables as well.
        if (coef_nc.variable_exists("totplnk"))
        {
            int n_internal_sourcetemps = coef_nc.get_dimension_size("temperature_Planck");

            Array<Float,2> totplnk(
                    coef_nc.get_variable<Float>( "totplnk", {n_bnds, n_internal_sourcetemps}),
                    {n_internal_sourcetemps, n_bnds});
            Array<Float,4> planck_frac(
                    coef_nc.get_variable<Float>("plank_fraction", {n_temps, n_press+1, n_mixingfracs, n_gpts}),
                    {n_gpts, n_mixingfracs, n_press+1, n_temps});

            // Construct the k-distribution.
            return Gas_optics_rrtmgp(
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
            Array<Float,1> solar_src_quiet(
                    coef_nc.get_variable<Float>("solar_source_quiet", {n_gpts}), {n_gpts});
            Array<Float,1> solar_src_facular(
                    coef_nc.get_variable<Float>("solar_source_facular", {n_gpts}), {n_gpts});
            Array<Float,1> solar_src_sunspot(
                    coef_nc.get_variable<Float>("solar_source_sunspot", {n_gpts}), {n_gpts});

            Float tsi = coef_nc.get_variable<Float>("tsi_default");
            Float mg_index = coef_nc.get_variable<Float>("mg_default");
            Float sb_index = coef_nc.get_variable<Float>("sb_default");

            return Gas_optics_rrtmgp(
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

    Cloud_optics load_and_init_cloud_optics(
            const std::string& coef_file)
    {
        // READ THE COEFFICIENTS FOR THE OPTICAL SOLVER.
        Netcdf_file coef_nc(coef_file, Netcdf_mode::Read);

        // Read look-up table coefficient dimensions
        int n_band     = coef_nc.get_dimension_size("nband");
        int n_rghice   = coef_nc.get_dimension_size("nrghice");
        int n_size_liq = coef_nc.get_dimension_size("nsize_liq");
        int n_size_ice = coef_nc.get_dimension_size("nsize_ice");

        Array<Float,2> band_lims_wvn(coef_nc.get_variable<Float>("bnd_limits_wavenumber", {n_band, 2}), {2, n_band});

        // Read look-up table constants.
        Float radliq_lwr = coef_nc.get_variable<Float>("radliq_lwr");
        Float radliq_upr = coef_nc.get_variable<Float>("radliq_upr");
        Float radliq_fac = coef_nc.get_variable<Float>("radliq_fac");

        Float radice_lwr = coef_nc.get_variable<Float>("radice_lwr");
        Float radice_upr = coef_nc.get_variable<Float>("radice_upr");
        Float radice_fac = coef_nc.get_variable<Float>("radice_fac");

        Array<Float,2> lut_extliq(
                coef_nc.get_variable<Float>("lut_extliq", {n_band, n_size_liq}), {n_size_liq, n_band});
        Array<Float,2> lut_ssaliq(
                coef_nc.get_variable<Float>("lut_ssaliq", {n_band, n_size_liq}), {n_size_liq, n_band});
        Array<Float,2> lut_asyliq(
                coef_nc.get_variable<Float>("lut_asyliq", {n_band, n_size_liq}), {n_size_liq, n_band});

        Array<Float,3> lut_extice(
                coef_nc.get_variable<Float>("lut_extice", {n_rghice, n_band, n_size_ice}), {n_size_ice, n_band, n_rghice});
        Array<Float,3> lut_ssaice(
                coef_nc.get_variable<Float>("lut_ssaice", {n_rghice, n_band, n_size_ice}), {n_size_ice, n_band, n_rghice});
        Array<Float,3> lut_asyice(
                coef_nc.get_variable<Float>("lut_asyice", {n_rghice, n_band, n_size_ice}), {n_size_ice, n_band, n_rghice});

        return Cloud_optics(
                band_lims_wvn,
                radliq_lwr, radliq_upr, radliq_fac,
                radice_lwr, radice_upr, radice_fac,
                lut_extliq, lut_ssaliq, lut_asyliq,
                lut_extice, lut_ssaice, lut_asyice);
    }

    Aerosol_optics load_and_init_aerosol_optics(
            const std::string& coef_file)
    {
        // READ THE COEFFICIENTS FOR THE OPTICAL SOLVER.
        Netcdf_file coef_nc(coef_file, Netcdf_mode::Read);

        // Read look-up table coefficient dimensions
        int n_band     = coef_nc.get_dimension_size("band_sw");
        int n_hum      = coef_nc.get_dimension_size("relative_humidity");
        int n_philic = coef_nc.get_dimension_size("hydrophilic");
        int n_phobic = coef_nc.get_dimension_size("hydrophobic");

        Array<Float,2> band_lims_wvn({2, n_band});

        Array<Float,2> mext_phobic(
                coef_nc.get_variable<Float>("mass_ext_sw_hydrophobic", {n_phobic, n_band}), {n_band, n_phobic});
        Array<Float,2> ssa_phobic(
                coef_nc.get_variable<Float>("ssa_sw_hydrophobic", {n_phobic, n_band}), {n_band, n_phobic});
        Array<Float,2> g_phobic(
                coef_nc.get_variable<Float>("asymmetry_sw_hydrophobic", {n_phobic, n_band}), {n_band, n_phobic});

        Array<Float,3> mext_philic(
                coef_nc.get_variable<Float>("mass_ext_sw_hydrophilic", {n_philic, n_hum, n_band}), {n_band, n_hum, n_philic});
        Array<Float,3> ssa_philic(
                coef_nc.get_variable<Float>("ssa_sw_hydrophilic", {n_philic, n_hum, n_band}), {n_band, n_hum, n_philic});
        Array<Float,3> g_philic(
                coef_nc.get_variable<Float>("asymmetry_sw_hydrophilic", {n_philic, n_hum, n_band}), {n_band, n_hum, n_philic});

        Array<Float,1> rh_upper(
                coef_nc.get_variable<Float>("relative_humidity2", {n_hum}), {n_hum});

        return Aerosol_optics(
                band_lims_wvn, rh_upper,
                mext_phobic, ssa_phobic, g_phobic,
                mext_philic, ssa_philic, g_philic);
    }
}


Radiation_solver_longwave::Radiation_solver_longwave(
        const Gas_concs& gas_concs,
        const std::string& file_name_gas,
        const std::string& file_name_cloud)
{
    // Construct the gas optics classes for the solver.
    this->kdist = std::make_unique<Gas_optics_rrtmgp>(
            load_and_init_gas_optics(gas_concs, file_name_gas));

    this->cloud_optics = std::make_unique<Cloud_optics>(
            load_and_init_cloud_optics(file_name_cloud));
}


void Radiation_solver_longwave::solve(
        const bool switch_fluxes,
        const bool switch_cloud_optics,
        const bool switch_output_optical,
        const bool switch_output_bnd_fluxes,
        const Gas_concs& gas_concs,
        const Array<PressureType,2>& p_lay, const Array<PressureType,2>& p_lev,
        const Array<TempType,2>& t_lay, const Array<TempType,2>& t_lev,
        const Array<Float,2>& col_dry,
        const Array<TempType, 1>& t_sfc, const Array<Float,2>& emis_sfc,
        const Array<Float,2>& lwp, const Array<Float,2>& iwp,
        const Array<Float,2>& rel, const Array<Float,2>& rei,
        Array<Float,3>& tau, Array<Float,3>& lay_source,
        Array<Float,3>& lev_source_inc, Array<Float,3>& lev_source_dec, Array<Float,2>& sfc_source,
        Array<Float,2>& lw_flux_up, Array<Float,2>& lw_flux_dn, Array<Float,2>& lw_flux_net,
        Array<Float,3>& lw_bnd_flux_up, Array<Float,3>& lw_bnd_flux_dn, Array<Float,3>& lw_bnd_flux_net) const
{
    const int n_col = p_lay.dim(1);
    const int n_lay = p_lay.dim(2);
    const int n_lev = p_lev.dim(2);
    const int n_gpt = this->kdist->get_ngpt();
    const int n_bnd = this->kdist->get_nband();

    const Bool top_at_1 = p_lay({1, 1}) < p_lay({1, n_lay});

    constexpr int n_col_block = 12;

    // Read the sources and create containers for the substeps.
    int n_blocks = n_col / n_col_block;
    int n_col_block_residual = n_col % n_col_block;

    std::unique_ptr<Optical_props_arry> optical_props_subset;
    std::unique_ptr<Optical_props_arry> optical_props_residual;

    optical_props_subset = std::make_unique<Optical_props_1scl>(n_col_block, n_lay, *kdist);

    std::unique_ptr<Source_func_lw> sources_subset;
    std::unique_ptr<Source_func_lw> sources_residual;

    sources_subset = std::make_unique<Source_func_lw>(n_col_block, n_lay, *kdist);

    if (n_col_block_residual > 0)
    {
        optical_props_residual = std::make_unique<Optical_props_1scl>(n_col_block_residual, n_lay, *kdist);
        sources_residual = std::make_unique<Source_func_lw>(n_col_block_residual, n_lay, *kdist);
    }

    std::unique_ptr<Optical_props_1scl> cloud_optical_props_subset;
    std::unique_ptr<Optical_props_1scl> cloud_optical_props_residual;

    if (switch_cloud_optics)
    {
        cloud_optical_props_subset = std::make_unique<Optical_props_1scl>(n_col_block, n_lay, *cloud_optics);
        if (n_col_block_residual > 0)
            cloud_optical_props_residual = std::make_unique<Optical_props_1scl>(n_col_block_residual, n_lay, *cloud_optics);
    }

    // Lambda function for solving optical properties subset.
    auto call_kernels = [&](
            const int col_s_in, const int col_e_in,
            std::unique_ptr<Optical_props_arry>& optical_props_subset_in,
            std::unique_ptr<Optical_props_1scl>& cloud_optical_props_subset_in,
            Source_func_lw& sources_subset_in,
            const Array<Float,2>& emis_sfc_subset_in,
            Fluxes_broadband& fluxes,
            Fluxes_broadband& bnd_fluxes)
    {
        const int n_col_in = col_e_in - col_s_in + 1;
        Gas_concs gas_concs_subset(gas_concs, col_s_in, n_col_in);

        auto p_lev_subset = p_lev.subset({{ {col_s_in, col_e_in}, {1, n_lev} }});

        Array<Float,2> col_dry_subset({n_col_in, n_lay});
        if (col_dry.size() == 0)
            Gas_optics_rrtmgp::get_col_dry(col_dry_subset, gas_concs_subset.get_vmr("h2o"), p_lev_subset);
        else
            col_dry_subset = std::move(col_dry.subset({{ {col_s_in, col_e_in}, {1, n_lay} }}));

        kdist->gas_optics(
                p_lay.subset({{ {col_s_in, col_e_in}, {1, n_lay} }}),
                p_lev_subset,
                t_lay.subset({{ {col_s_in, col_e_in}, {1, n_lay} }}),
                t_sfc.subset({{ {col_s_in, col_e_in} }}),
                gas_concs_subset,
                optical_props_subset_in,
                sources_subset_in,
                col_dry_subset,
                t_lev.subset({{ {col_s_in, col_e_in}, {1, n_lev} }}) );

        if (switch_cloud_optics)
        {
            cloud_optics->cloud_optics(
                    lwp.subset({{ {col_s_in, col_e_in}, {1, n_lay} }}),
                    iwp.subset({{ {col_s_in, col_e_in}, {1, n_lay} }}),
                    rel.subset({{ {col_s_in, col_e_in}, {1, n_lay} }}),
                    rei.subset({{ {col_s_in, col_e_in}, {1, n_lay} }}),
                    *cloud_optical_props_subset_in);

            // cloud->delta_scale();

            // Add the cloud optical props to the gas optical properties.
            add_to(
                    dynamic_cast<Optical_props_1scl&>(*optical_props_subset_in),
                    dynamic_cast<Optical_props_1scl&>(*cloud_optical_props_subset_in));
        }

        // Store the optical properties, if desired.
        if (switch_output_optical)
        {
            for (int igpt=1; igpt<=n_gpt; ++igpt)
                for (int ilay=1; ilay<=n_lay; ++ilay)
                    for (int icol=1; icol<=n_col_in; ++icol)
                    {
                        tau           ({icol+col_s_in-1, ilay, igpt}) = optical_props_subset_in->get_tau()    ({icol, ilay, igpt});
                        lay_source    ({icol+col_s_in-1, ilay, igpt}) = sources_subset_in.get_lay_source()    ({icol, ilay, igpt});
                        lev_source_inc({icol+col_s_in-1, ilay, igpt}) = sources_subset_in.get_lev_source_inc()({icol, ilay, igpt});
                        lev_source_dec({icol+col_s_in-1, ilay, igpt}) = sources_subset_in.get_lev_source_dec()({icol, ilay, igpt});
                    }

            for (int igpt=1; igpt<=n_gpt; ++igpt)
                for (int icol=1; icol<=n_col_in; ++icol)
                    sfc_source({icol+col_s_in-1, igpt}) = sources_subset_in.get_sfc_source()({icol, igpt});
        }

        if (!switch_fluxes)
            return;

        Array<Float,3> gpt_flux_up;
        Array<Float,3> gpt_flux_dn;

        // Save the output per gpt if postprocessing is desired.
        if (switch_output_bnd_fluxes)
        {
            gpt_flux_up.set_dims({n_col_in, n_lev, n_gpt});
            gpt_flux_dn.set_dims({n_col_in, n_lev, n_gpt});
        }
        else
        {
            gpt_flux_up.set_dims({n_col_in, n_lev, 1});
            gpt_flux_dn.set_dims({n_col_in, n_lev, 1});
        }

        constexpr int n_ang = 1;

        Rte_lw::rte_lw(
                optical_props_subset_in,
                top_at_1,
                sources_subset_in,
                emis_sfc_subset_in,
                Array<Float,2>({n_col_in, n_gpt}), // Add an empty array, no inc_flux.
                gpt_flux_up, gpt_flux_dn,
                n_ang);

        if (switch_output_bnd_fluxes)
        {
            // Aggegated fluxes.
            fluxes.reduce(gpt_flux_up, gpt_flux_dn, optical_props_subset_in, top_at_1);

            for (int ilev=1; ilev<=n_lev; ++ilev)
                for (int icol=1; icol<=n_col_in; ++icol)
                {
                    lw_flux_up ({icol+col_s_in-1, ilev}) = fluxes.get_flux_up ()({icol, ilev});
                    lw_flux_dn ({icol+col_s_in-1, ilev}) = fluxes.get_flux_dn ()({icol, ilev});
                    lw_flux_net({icol+col_s_in-1, ilev}) = fluxes.get_flux_net()({icol, ilev});
                }

            // Aggegated fluxes per band
            bnd_fluxes.reduce(gpt_flux_up, gpt_flux_dn, optical_props_subset_in, top_at_1);

            for (int ibnd=1; ibnd<=n_bnd; ++ibnd)
                for (int ilev=1; ilev<=n_lev; ++ilev)
                    for (int icol=1; icol<=n_col_in; ++icol)
                    {
                        lw_bnd_flux_up ({icol+col_s_in-1, ilev, ibnd}) = bnd_fluxes.get_bnd_flux_up ()({icol, ilev, ibnd});
                        lw_bnd_flux_dn ({icol+col_s_in-1, ilev, ibnd}) = bnd_fluxes.get_bnd_flux_dn ()({icol, ilev, ibnd});
                        lw_bnd_flux_net({icol+col_s_in-1, ilev, ibnd}) = bnd_fluxes.get_bnd_flux_net()({icol, ilev, ibnd});
                    }
        }
        else
        {
            // Copy the data to the output.
            for (int ilev=1; ilev<=n_lev; ++ilev)
                for (int icol=1; icol<=n_col_in; ++icol)
                {
                    lw_flux_up ({icol+col_s_in-1, ilev}) = gpt_flux_up({icol, ilev, 1});
                    lw_flux_dn ({icol+col_s_in-1, ilev}) = gpt_flux_dn({icol, ilev, 1});
                    lw_flux_net({icol+col_s_in-1, ilev}) = gpt_flux_dn({icol, ilev, 1}) - gpt_flux_up({icol, ilev, 1});
                }
        }
    };

    for (int b=1; b<=n_blocks; ++b)
    {
        const int col_s = (b-1) * n_col_block + 1;
        const int col_e =  b    * n_col_block;

        Array<Float,2> emis_sfc_subset = emis_sfc.subset({{ {1, n_bnd}, {col_s, col_e} }});

        std::unique_ptr<Fluxes_broadband> fluxes_subset =
                std::make_unique<Fluxes_broadband>(n_col_block, n_lev);
        std::unique_ptr<Fluxes_broadband> bnd_fluxes_subset =
                std::make_unique<Fluxes_byband>(n_col_block, n_lev, n_bnd);

        call_kernels(
                col_s, col_e,
                optical_props_subset,
                cloud_optical_props_subset,
                *sources_subset,
                emis_sfc_subset,
                *fluxes_subset,
                *bnd_fluxes_subset);
    }

    if (n_col_block_residual > 0)
    {
        const int col_s = n_col - n_col_block_residual + 1;
        const int col_e = n_col;

        Array<Float,2> emis_sfc_residual = emis_sfc.subset({{ {1, n_bnd}, {col_s, col_e} }});
        std::unique_ptr<Fluxes_broadband> fluxes_residual =
                std::make_unique<Fluxes_broadband>(n_col_block_residual, n_lev);
        std::unique_ptr<Fluxes_broadband> bnd_fluxes_residual =
                std::make_unique<Fluxes_byband>(n_col_block_residual, n_lev, n_bnd);

        call_kernels(
                col_s, col_e,
                optical_props_residual,
                cloud_optical_props_residual,
                *sources_residual,
                emis_sfc_residual,
                *fluxes_residual,
                *bnd_fluxes_residual);
    }
}


Radiation_solver_shortwave::Radiation_solver_shortwave(
        const Gas_concs& gas_concs,
        const bool switch_cloud_optics,
        const bool switch_aerosol_optics,
        const std::string& file_name_gas,
        const std::string& file_name_cloud,
        const std::string& file_name_aerosol)
{
    // Construct the gas optics classes for the solver.
    this->kdist = std::make_unique<Gas_optics_rrtmgp>(
            load_and_init_gas_optics(gas_concs, file_name_gas));

    if (switch_cloud_optics)
        this->cloud_optics = std::make_unique<Cloud_optics>(
                load_and_init_cloud_optics(file_name_cloud));

    if (switch_aerosol_optics)
        this->aerosol_optics = std::make_unique<Aerosol_optics>(
                load_and_init_aerosol_optics(file_name_aerosol));
}


void Radiation_solver_shortwave::solve(
        const bool switch_fluxes,
        const bool switch_cloud_optics,
        const bool switch_aerosol_optics,
        const bool switch_output_optical,
        const bool switch_output_bnd_fluxes,
        const bool switch_delta_cloud,
        const bool switch_delta_aerosol,
        const Gas_concs& gas_concs,
        const Array<Float,2>& p_lay, const Array<Float,2>& p_lev,
        const Array<Float,2>& t_lay, const Array<Float,2>& t_lev,
        const Array<Float,2>& col_dry,
        const Array<Float,2>& sfc_alb_dir, const Array<Float,2>& sfc_alb_dif,
        const Array<Float,1>& tsi_scaling, const Array<Float,1>& mu0,
        const Array<Float,2>& lwp, const Array<Float,2>& iwp,
        const Array<Float,2>& rel, const Array<Float,2>& rei,
        const Array<Float,2>& rh,
        const Aerosol_concs& aerosol_concs,
        Array<Float,3>& tau, Array<Float,3>& ssa, Array<Float,3>& g,
        Array<Float,2>& toa_src,
        Array<Float,2>& sw_flux_up, Array<Float,2>& sw_flux_dn,
        Array<Float,2>& sw_flux_dn_dir, Array<Float,2>& sw_flux_net,
        Array<Float,3>& sw_bnd_flux_up, Array<Float,3>& sw_bnd_flux_dn,
        Array<Float,3>& sw_bnd_flux_dn_dir, Array<Float,3>& sw_bnd_flux_net) const
{
    const int n_col = p_lay.dim(1);
    const int n_lay = p_lay.dim(2);
    const int n_lev = p_lev.dim(2);
    const int n_gpt = this->kdist->get_ngpt();
    const int n_bnd = this->kdist->get_nband();

    const Bool top_at_1 = p_lay({1, 1}) < p_lay({1, n_lay});

    constexpr int n_col_block = 12;

    // Read the sources and create containers for the substeps.
    int n_blocks = n_col / n_col_block;
    int n_col_block_residual = n_col % n_col_block;

    std::unique_ptr<Optical_props_arry> optical_props_subset;
    std::unique_ptr<Optical_props_arry> optical_props_residual;

    optical_props_subset = std::make_unique<Optical_props_2str>(n_col_block, n_lay, *kdist);
    if (n_col_block_residual > 0)
        optical_props_residual = std::make_unique<Optical_props_2str>(n_col_block_residual, n_lay, *kdist);

    std::unique_ptr<Optical_props_2str> cloud_optical_props_subset;
    std::unique_ptr<Optical_props_2str> cloud_optical_props_residual;

    std::unique_ptr<Optical_props_2str> aerosol_optical_props_subset;
    std::unique_ptr<Optical_props_2str> aerosol_optical_props_residual;

    if (switch_cloud_optics)
    {
        cloud_optical_props_subset = std::make_unique<Optical_props_2str>(n_col_block, n_lay, *cloud_optics);
        if (n_col_block_residual > 0)
            cloud_optical_props_residual = std::make_unique<Optical_props_2str>(n_col_block_residual, n_lay, *cloud_optics);
    }

    if (switch_aerosol_optics)
    {
        aerosol_optical_props_subset = std::make_unique<Optical_props_2str>(n_col_block, n_lay, *aerosol_optics);
        if (n_col_block_residual > 0)
            aerosol_optical_props_residual = std::make_unique<Optical_props_2str>(n_col_block_residual, n_lay, *aerosol_optics);
    }

    // Lambda function for solving optical properties subset.
    auto call_kernels = [&](
            const int col_s_in, const int col_e_in,
            std::unique_ptr<Optical_props_arry>& optical_props_subset_in,
            std::unique_ptr<Optical_props_2str>& cloud_optical_props_subset_in,
            std::unique_ptr<Optical_props_2str>& aerosol_optical_props_subset_in,
            Fluxes_broadband& fluxes,
            Fluxes_broadband& bnd_fluxes)
    {
        const int n_col_in = col_e_in - col_s_in + 1;
        Gas_concs gas_concs_subset(gas_concs, col_s_in, n_col_in);

        auto p_lev_subset = p_lev.subset({{ {col_s_in, col_e_in}, {1, n_lev} }});

        Array<Float,2> col_dry_subset({n_col_in, n_lay});
        if (col_dry.size() == 0)
            Gas_optics_rrtmgp::get_col_dry(col_dry_subset, gas_concs_subset.get_vmr("h2o"), p_lev_subset);
        else
            col_dry_subset = std::move(col_dry.subset({{ {col_s_in, col_e_in}, {1, n_lay} }}));

        Array<Float,2> toa_src_subset({n_col_in, n_gpt});

        kdist->gas_optics(
                p_lay.subset({{ {col_s_in, col_e_in}, {1, n_lay} }}),
                p_lev_subset,
                t_lay.subset({{ {col_s_in, col_e_in}, {1, n_lay} }}),
                gas_concs_subset,
                optical_props_subset_in,
                toa_src_subset,
                col_dry_subset);

        auto tsi_scaling_subset = tsi_scaling.subset({{ {col_s_in, col_e_in} }});

        for (int igpt=1; igpt<=n_gpt; ++igpt)
            for (int icol=1; icol<=n_col_in; ++icol)
                toa_src_subset({icol, igpt}) *= tsi_scaling_subset({icol});


        if (switch_cloud_optics)
        {
            Array<int,2> cld_mask_liq({n_col_in, n_lay});
            Array<int,2> cld_mask_ice({n_col_in, n_lay});

            cloud_optics->cloud_optics(
                    lwp.subset({{ {col_s_in, col_e_in}, {1, n_lay} }}),
                    iwp.subset({{ {col_s_in, col_e_in}, {1, n_lay} }}),
                    rel.subset({{ {col_s_in, col_e_in}, {1, n_lay} }}),
                    rei.subset({{ {col_s_in, col_e_in}, {1, n_lay} }}),
                    *cloud_optical_props_subset_in);

            if (switch_delta_cloud)
                cloud_optical_props_subset_in->delta_scale();

            // Add the cloud optical props to the gas optical properties.
            add_to(
                    dynamic_cast<Optical_props_2str&>(*optical_props_subset_in),
                    dynamic_cast<Optical_props_2str&>(*cloud_optical_props_subset_in));
        }

        if (switch_aerosol_optics)
        {
            Aerosol_concs aerosol_concs_subset(aerosol_concs, 1, n_col_in);
            aerosol_optics->aerosol_optics(
                    aerosol_concs_subset,
                    rh.subset({{ {col_s_in, col_e_in}, {1, n_lay} }}),
                    p_lev_subset,
                    *aerosol_optical_props_subset_in);

            if (switch_delta_aerosol)
                aerosol_optical_props_subset_in->delta_scale();

            // Add the aerosol optical props to the gas optical properties.
            add_to(
                    dynamic_cast<Optical_props_2str&>(*optical_props_subset_in),
                    dynamic_cast<Optical_props_2str&>(*aerosol_optical_props_subset_in));
        }


        // Store the optical properties, if desired.
        if (switch_output_optical)
        {
            for (int igpt=1; igpt<=n_gpt; ++igpt)
                for (int ilay=1; ilay<=n_lay; ++ilay)
                    for (int icol=1; icol<=n_col_in; ++icol)
                    {
                        tau({icol+col_s_in-1, ilay, igpt}) = optical_props_subset_in->get_tau()({icol, ilay, igpt});
                        ssa({icol+col_s_in-1, ilay, igpt}) = optical_props_subset_in->get_ssa()({icol, ilay, igpt});
                        g  ({icol+col_s_in-1, ilay, igpt}) = optical_props_subset_in->get_g  ()({icol, ilay, igpt});
                    }

            for (int igpt=1; igpt<=n_gpt; ++igpt)
                for (int icol=1; icol<=n_col_in; ++icol)
                    toa_src({icol+col_s_in-1, igpt}) = toa_src_subset({icol, igpt});
        }

        if (!switch_fluxes)
            return;

        Array<Float,3> gpt_flux_up;
        Array<Float,3> gpt_flux_dn;
        Array<Float,3> gpt_flux_dn_dir;

        // Save the output per gpt if postprocessing is desired.
        if (switch_output_bnd_fluxes)
        {
            gpt_flux_up.set_dims({n_col_in, n_lev, n_gpt});
            gpt_flux_dn.set_dims({n_col_in, n_lev, n_gpt});
            gpt_flux_dn_dir.set_dims({n_col_in, n_lev, n_gpt});
        }
        else
        {
            gpt_flux_up.set_dims({n_col_in, n_lev, 1});
            gpt_flux_dn.set_dims({n_col_in, n_lev, 1});
            gpt_flux_dn_dir.set_dims({n_col_in, n_lev, 1});
        }

        Rte_sw::rte_sw(
                optical_props_subset_in,
                top_at_1,
                mu0.subset({{ {col_s_in, col_e_in} }}),
                toa_src_subset,
                sfc_alb_dir.subset({{ {1, n_bnd}, {col_s_in, col_e_in} }}),
                sfc_alb_dif.subset({{ {1, n_bnd}, {col_s_in, col_e_in} }}),
                Array<Float,2>(), // Add an empty array, no inc_flux.
                gpt_flux_up,
                gpt_flux_dn,
                gpt_flux_dn_dir);

        if (switch_output_bnd_fluxes)
        {
            // Aggegated fluxes.
            fluxes.reduce(gpt_flux_up, gpt_flux_dn, gpt_flux_dn_dir, optical_props_subset_in, top_at_1);
            for (int ilev=1; ilev<=n_lev; ++ilev)
                for (int icol=1; icol<=n_col_in; ++icol)
                {
                    sw_flux_up    ({icol+col_s_in-1, ilev}) = fluxes.get_flux_up    ()({icol, ilev});
                    sw_flux_dn    ({icol+col_s_in-1, ilev}) = fluxes.get_flux_dn    ()({icol, ilev});
                    sw_flux_dn_dir({icol+col_s_in-1, ilev}) = fluxes.get_flux_dn_dir()({icol, ilev});
                    sw_flux_net   ({icol+col_s_in-1, ilev}) = fluxes.get_flux_net   ()({icol, ilev});
                }

            // Aggegated fluxes per band
            bnd_fluxes.reduce(gpt_flux_up, gpt_flux_dn, gpt_flux_dn_dir, optical_props_subset_in, top_at_1);

            for (int ibnd=1; ibnd<=n_bnd; ++ibnd)
                for (int ilev=1; ilev<=n_lev; ++ilev)
                    for (int icol=1; icol<=n_col_in; ++icol)
                    {
                        sw_bnd_flux_up    ({icol+col_s_in-1, ilev, ibnd}) = bnd_fluxes.get_bnd_flux_up    ()({icol, ilev, ibnd});
                        sw_bnd_flux_dn    ({icol+col_s_in-1, ilev, ibnd}) = bnd_fluxes.get_bnd_flux_dn    ()({icol, ilev, ibnd});
                        sw_bnd_flux_dn_dir({icol+col_s_in-1, ilev, ibnd}) = bnd_fluxes.get_bnd_flux_dn_dir()({icol, ilev, ibnd});
                        sw_bnd_flux_net   ({icol+col_s_in-1, ilev, ibnd}) = bnd_fluxes.get_bnd_flux_net   ()({icol, ilev, ibnd});
                    }
        }
        else
        {
            // Copy the data to the output.
            for (int ilev=1; ilev<=n_lev; ++ilev)
                for (int icol=1; icol<=n_col_in; ++icol)
                {
                    sw_flux_up    ({icol+col_s_in-1, ilev}) = gpt_flux_up    ({icol, ilev, 1});
                    sw_flux_dn    ({icol+col_s_in-1, ilev}) = gpt_flux_dn    ({icol, ilev, 1});
                    sw_flux_dn_dir({icol+col_s_in-1, ilev}) = gpt_flux_dn_dir({icol, ilev, 1});
                    sw_flux_net   ({icol+col_s_in-1, ilev}) = gpt_flux_dn({icol, ilev, 1}) - gpt_flux_up({icol, ilev, 1});
                }
        }
    };

    for (int b=1; b<=n_blocks; ++b)
    {
        const int col_s = (b-1) * n_col_block + 1;
        const int col_e =  b    * n_col_block;

        std::unique_ptr<Fluxes_broadband> fluxes_subset =
                std::make_unique<Fluxes_broadband>(n_col_block, n_lev);
        std::unique_ptr<Fluxes_broadband> bnd_fluxes_subset =
                std::make_unique<Fluxes_byband>(n_col_block, n_lev, n_bnd);

        call_kernels(
                col_s, col_e,
                optical_props_subset,
                cloud_optical_props_subset,
                aerosol_optical_props_subset,
                *fluxes_subset,
                *bnd_fluxes_subset);
    }

    if (n_col_block_residual > 0)
    {
        const int col_s = n_col - n_col_block_residual + 1;
        const int col_e = n_col;

        std::unique_ptr<Fluxes_broadband> fluxes_residual =
                std::make_unique<Fluxes_broadband>(n_col_block_residual, n_lev);
        std::unique_ptr<Fluxes_broadband> bnd_fluxes_residual =
                std::make_unique<Fluxes_byband>(n_col_block_residual, n_lev, n_bnd);

        call_kernels(
                col_s, col_e,
                optical_props_residual,
                cloud_optical_props_residual,
                aerosol_optical_props_residual,
                *fluxes_residual,
                *bnd_fluxes_residual);
    }
}
