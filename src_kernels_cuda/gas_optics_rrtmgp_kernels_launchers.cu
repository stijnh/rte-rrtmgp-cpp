#include <chrono>
#include <functional>
#include <iostream>
#include <iomanip>

#include "gas_optics_rrtmgp_kernels_cuda.h"
#include "tools_gpu.h"
#include "tuner.h"
#include "kernel.h"
#include "kernel_float.h"


namespace
{
    #include "gas_optics_rrtmgp_kernels.cu"

    using Tools_gpu::calc_grid_size;
}


namespace Gas_optics_rrtmgp_kernels_cuda
{
    void reorder123x321(
            const int ni, const int nj, const int nk,
            const Float* arr_in, Float* arr_out)
    {
        Tuner_map& tunings = Tuner::get_map();

        dim3 grid(ni, nj, nk);
        dim3 block;

        if (tunings.count("reorder123x321_kernel") == 0)
        {
            std::tie(grid, block) = tune_kernel(
                "reorder123x321_kernel",
                dim3(ni, nj, nk),
                {1, 2, 4, 8, 16, 24, 32, 48, 64, 96},
                {1, 2, 4, 8, 16, 24, 32, 48, 64, 96},
                {1, 2, 4, 8, 16, 24, 32, 48, 64, 96},
                reorder123x321_kernel,
                ni, nj, nk, arr_in, arr_out);

            tunings["reorder123x321_kernel"].first = grid;
            tunings["reorder123x321_kernel"].second = block;
        }
        else
        {
            block = tunings["reorder123x321_kernel"].second;
        }

        grid = calc_grid_size(block, dim3(ni, nj, nk));

        reorder123x321_kernel<<<grid, block>>>(
                ni, nj, nk, arr_in, arr_out);
    }


    void reorder12x21(
            const int ni, const int nj,
            const Float* arr_in, Float* arr_out)
    {
        dim3 block_gpu(32, 16, 1);
        dim3 grid_gpu = calc_grid_size(block_gpu, dim3(ni, nj));

        reorder12x21_kernel<<<grid_gpu, block_gpu>>>(
                ni, nj, arr_in, arr_out);
    }

    void zero_array_bytes(const size_t nbytes, void* arr) {
        cuda_safe_call(gpuMemsetAsync(arr, 0, nbytes, nullptr));
    }

    void interpolation(
            const int ncol, const int nlay,
            const int ngas, const int nflav, const int neta, const int npres, const int ntemp,
            const int* flavor,
            const FloatPressure* press_ref_log,
            const FloatTemperature * temp_ref,
            FloatPressure press_ref_log_delta,
            FloatTemperature temp_ref_min,
            FloatTemperature temp_ref_delta,
            FloatPressure press_ref_trop_log,
            const Float* vmr_ref,
            const FloatPressure* play,
            const FloatTemperature* tlay,
            FloatColGas* col_gas,
            int* jtemp,
            FloatFMajor* fmajor, FloatFMinor* fminor,
            FloatColMix* col_mix,
            Bool* tropo,
            int* jeta,
            int* jpress)
    {
        dim3 block_gpu(4, 2, 16);
        dim3 grid_gpu = calc_grid_size(block_gpu, dim3(ncol, nlay, nflav));

        Float tmin = std::numeric_limits<Float>::min();
        interpolation_kernel<<<grid_gpu, block_gpu>>>(
                ncol, nlay, ngas, nflav, neta, npres, ntemp, tmin,
                flavor, press_ref_log, temp_ref,
                press_ref_log_delta, temp_ref_min,
                temp_ref_delta, press_ref_trop_log,
                vmr_ref, play, tlay,
                col_gas, jtemp, fmajor,
                fminor, col_mix, tropo,
                jeta, jpress);
    }


    void combine_abs_and_rayleigh(
            const int ncol, const int nlay, const int ngpt,
            const FloatTau* tau_abs, const FloatTau* tau_rayleigh,
            FloatTau* tau, FloatOptical* ssa, FloatOptical* g)
    {
        Tuner_map& tunings = Tuner::get_map();

        Float tmin = std::numeric_limits<Float>::min();

        dim3 grid(ncol, nlay, ngpt);
        dim3 block;

        if (tunings.count("combine_abs_and_rayleigh_kernel") == 0)
        {
            std::tie(grid, block) = tune_kernel(
                "combine_abs_and_rayleigh_kernel",
                dim3(ncol, nlay, ngpt),
                {1, 2, 4, 8, 16, 24, 32, 48, 64, 96}, {1, 2, 4}, {1, 2, 4, 8, 16, 24, 32, 48, 64, 96},
                combine_abs_and_rayleigh_kernel,
                ncol, nlay, ngpt, tmin,
                tau_abs, tau_rayleigh,
                tau, ssa, g);

            tunings["combine_abs_and_rayleigh_kernel"].first = grid;
            tunings["combine_abs_and_rayleigh_kernel"].second = block;
        }
        else
        {
            block = tunings["combine_abs_and_rayleigh_kernel"].second;
        }

        grid = calc_grid_size(block, dim3(ncol, nlay, ngpt));

        combine_abs_and_rayleigh_kernel<<<grid, block>>>(
                ncol, nlay, ngpt, tmin,
                tau_abs, tau_rayleigh,
                tau, ssa, g);
    }


    void compute_tau_rayleigh(
            const int ncol, const int nlay, const int nbnd, const int ngpt,
            const int ngas, const int nflav, const int neta, const int npres, const int ntemp,
            const int* gpoint_flavor,
            const int* band_lims_gpt,
            const Float* krayl,
            int idx_h2o, const FloatColDry* col_dry, const FloatColGas* col_gas,
            const FloatFMinor* fminor, const int* jeta,
            const Bool* tropo, const int* jtemp,
            FloatTau* tau_rayleigh)
    {
        Tuner_map& tunings = Tuner::get_map();

        dim3 grid(ncol, nlay);
        dim3 block;

        if (tunings.count("compute_tau_rayleigh_kernel") == 0)
        {
            std::tie(grid, block) = tune_kernel(
                "compute_tau_rayleigh_kernel",
                dim3(ncol, nlay),
                {1, 2, 4, 16, 24, 32, 48, 64, 96, 128, 256, 512, 1024}, {1, 2, 4, 8, 16}, {1},
                compute_tau_rayleigh_kernel,
                ncol, nlay, nbnd, ngpt,
                ngas, nflav, neta, npres, ntemp,
                gpoint_flavor,
                band_lims_gpt,
                krayl,
                idx_h2o, col_dry, col_gas,
                fminor, jeta,
                tropo, jtemp,
                tau_rayleigh);

            tunings["compute_tau_rayleigh_kernel"].first = grid;
            tunings["compute_tau_rayleigh_kernel"].second = block;
        }
        else
        {
            block = tunings["compute_tau_rayleigh_kernel"].second;
        }

        grid = calc_grid_size(block, dim3(ncol, nlay));

        compute_tau_rayleigh_kernel<<<grid, block>>>(
                ncol, nlay, nbnd, ngpt,
                ngas, nflav, neta, npres, ntemp,
                gpoint_flavor,
                band_lims_gpt,
                krayl,
                idx_h2o, col_dry, col_gas,
                fminor, jeta,
                tropo, jtemp,
                tau_rayleigh);
    }


    struct Gas_optical_depths_minor_kernel
    {
        template<unsigned int I, unsigned int J, unsigned int K, class... Args>
        static void launch(dim3 grid, dim3 block, Args... args)
        {
            gas_optical_depths_minor_kernel<I, J, K><<<grid, block>>>(args...);
        }
    };


    void compute_tau_absorption(
            const int ncol, const int nlay, const int nband, const int ngpt,
            const int ngas, const int nflav, const int neta, const int npres, const int ntemp,
            const int nminorlower, const int nminorklower,
            const int nminorupper, const int nminorkupper,
            const int idx_h2o,
            const int* gpoint_flavor,
            const int* band_lims_gpt,
            const FloatKMajor* kmajor,
            const FloatKMinor* kminor_lower,
            const FloatKMinor* kminor_upper,
            const int* minor_limits_gpt_lower,
            const int* minor_limits_gpt_upper,
            const Bool* minor_scales_with_density_lower,
            const Bool* minor_scales_with_density_upper,
            const Bool* scale_by_complement_lower,
            const Bool* scale_by_complement_upper,
            const int* idx_minor_lower,
            const int* idx_minor_upper,
            const int* idx_minor_scaling_lower,
            const int* idx_minor_scaling_upper,
            const int* kminor_start_lower,
            const int* kminor_start_upper,
            const Bool* tropo,
            const FloatColMix* col_mix, const FloatFMajor* fmajor,
            const FloatFMinor* fminor, const FloatPressure * play,
            const FloatTemperature * tlay, const FloatColGas* col_gas,
            const int* jeta, const int* jtemp,
            const int* jpress,
            FloatTau* tau)
    {
        using C = constants::gas_optical_depths_major_kernel;
        kernel_launcher::launch(
                Kernel("gas_optical_depths_major_kernel", "src_kernels_cuda/gas_optics_rrtmgp_kernels.cu",
                {
                    C::block_size_x,
                    C::block_size_y,
                    C::block_size_z,
                    C::vector_size,
                    kernel_launcher::TemplateArg::from_type<C::compute_type>(),
                    kernel_launcher::TemplateArg::from_type<C::kmajor_type>(),
                    kernel_launcher::TemplateArg::from_type<C::col_mix_type>(),
                    kernel_launcher::TemplateArg::from_type<C::fmajor_type>(),
                    kernel_launcher::TemplateArg::from_type<C::tau_type>()
               }),
                ncol, nlay, nband, ngpt,
                nflav, neta, npres, ntemp,
                gpoint_flavor, band_lims_gpt,
                kmajor, col_mix, fmajor, jeta,
                tropo, jtemp, jpress,
                tau);

        // Lower
        int idx_tropo = 1;

        using D = constants::gas_optical_depths_minor_kernel;
        kernel_launcher::launch(
                Kernel("gas_optical_depths_minor_kernel", "src_kernels_cuda/gas_optics_rrtmgp_kernels.cu",
               {
                    D::block_size_x,
                    D::block_size_y,
                    D::block_size_z,
                    D::vector_size,
                    D::use_smem,
                    kernel_launcher::TemplateArg::from_type<D::compute_type>(),
                    kernel_launcher::TemplateArg::from_type<D::kminor_type>(),
                    kernel_launcher::TemplateArg::from_type<D::pressure_type>(),
                    kernel_launcher::TemplateArg::from_type<D::temperature_type>(),
                    kernel_launcher::TemplateArg::from_type<D::col_gas_type>(),
                    kernel_launcher::TemplateArg::from_type<D::fminor_type>(),
                    kernel_launcher::TemplateArg::from_type<D::tau_type>(),
                    kernel_launcher::TemplateArg::from_type<D::accuracy_policy>()
                }),
                ncol, nlay, ngpt,
                ngas, nflav, ntemp, neta,
                nminorlower,
                nminorklower,
                idx_h2o, idx_tropo,
                gpoint_flavor,
                kminor_lower,
                minor_limits_gpt_lower,
                minor_scales_with_density_lower,
                scale_by_complement_lower,
                idx_minor_lower,
                idx_minor_scaling_lower,
                kminor_start_lower,
                play, tlay, col_gas,
                fminor, reinterpret_cast<const int2*>(jeta), jtemp,
                tropo, tau);


        // Upper
        idx_tropo = 0;

        kernel_launcher::launch(
            Kernel("gas_optical_depths_minor_kernel", "src_kernels_cuda/gas_optics_rrtmgp_kernels.cu",
                {
                   D::block_size_x,
                   D::block_size_y,
                   D::block_size_z,
                   D::vector_size,
                   D::use_smem,
                   kernel_launcher::TemplateArg::from_type<D::compute_type>(),
                   kernel_launcher::TemplateArg::from_type<D::kminor_type>(),
                   kernel_launcher::TemplateArg::from_type<D::pressure_type>(),
                   kernel_launcher::TemplateArg::from_type<D::temperature_type>(),
                   kernel_launcher::TemplateArg::from_type<D::col_gas_type>(),
                   kernel_launcher::TemplateArg::from_type<D::fminor_type>(),
                   kernel_launcher::TemplateArg::from_type<D::tau_type>(),
                   kernel_launcher::TemplateArg::from_type<D::accuracy_policy>()
                }),
                ncol, nlay, ngpt,
                ngas, nflav, ntemp, neta,
                nminorupper,
                nminorkupper,
                idx_h2o, idx_tropo,
                gpoint_flavor,
                kminor_upper,
                minor_limits_gpt_upper,
                minor_scales_with_density_upper,
                scale_by_complement_upper,
                idx_minor_upper,
                idx_minor_scaling_upper,
                kminor_start_upper,
                play, tlay, col_gas,
                fminor, reinterpret_cast<const int2*>(jeta), jtemp,
                tropo, tau);
    }


    void compute_planck_source(
            const int ncol,
            const int nlay,
            const int nbnd,
            const int ngpt,
            const int nflav,
            const int neta,
            const int npres,
            const int ntemp,
            const int nPlanckTemp,
            const FloatTemperature* tlay,
            const FloatTemperature* tlev,
            const FloatTemperature* tsfc,
            const int sfc_lay,
            const FloatFMajor* fmajor,
            const int* jeta,
            const Bool* tropo,
            const int* jtemp,
            const int* jpress,
            const int* gpoint_bands,
            const int* band_lims_gpt,
            const Float* pfracin,
            const FloatTemperature temp_ref_min,
            const Float totplnk_delta,
            const Float* totplnk,
            const int* gpoint_flavor,
            FloatSurface* sfc_src,
            FloatSource* lay_src,
            FloatSource* lev_src,
            Float* sfc_src_jac)
    {
        Tuner_map& tunings = Tuner::get_map();

        const Float delta_Tsurf = Float(1.);

        dim3 grid_gpu;
        dim3 block_gpu;
        
        if (tunings.count("Planck_source_kernel") == 0)
        {
            std::tie(grid_gpu, block_gpu) = tune_kernel(
                    "Planck_source_kernel",
                    dim3(ncol, nlay, ngpt),
                    {4, 8, 16, 32, 48, 64, 96, 128},
                    {1},
                    {4, 8, 16, 32, 48, 64, 96, 128},
                    Planck_source_kernel,
                    ncol, nlay, nbnd, ngpt,
                    nflav, neta, npres, ntemp, nPlanckTemp,
                    tlay, tlev, tsfc, sfc_lay,
                    fmajor, jeta, tropo, jtemp,
                    jpress, gpoint_bands, band_lims_gpt,
                    pfracin, temp_ref_min, totplnk_delta,
                    totplnk, gpoint_flavor,
                    delta_Tsurf, sfc_src, lay_src,
                    lev_src,
                    sfc_src_jac);
            
            tunings["Planck_source_kernel"].first = grid_gpu;
            tunings["Planck_source_kernel"].second = block_gpu;
        }
        else
        {
            block_gpu = tunings["Planck_source_kernel"].second;
        }

        grid_gpu = calc_grid_size(block_gpu, dim3(ncol, nlay, ngpt));

        Planck_source_kernel<<<grid_gpu, block_gpu>>>(
                ncol, nlay, nbnd, ngpt,
                nflav, neta, npres, ntemp, nPlanckTemp,
                tlay, tlev, tsfc, sfc_lay,
                fmajor, jeta, tropo, jtemp,
                jpress, gpoint_bands, band_lims_gpt,
                pfracin, temp_ref_min, totplnk_delta,
                totplnk, gpoint_flavor,
                delta_Tsurf,
                sfc_src, lay_src,
                lev_src,
                sfc_src_jac);
    }
}
