#include <chrono>
#include <functional>
#include <iostream>
#include <iomanip>

#include "kernel_launcher.h"
#include "kernel_launcher/pragma.h"
#include "rrtmgp_kernel_launcher_cuda.h"
#include "tools_gpu.h"
#include "Array.h"
#include "tuner.h"


namespace
{
#include "gas_optics_kernels.cu"
}

namespace kernel_launcher {
    template <typename T, int N>
    struct IntoKernelArg<Array_gpu<T, N>> {
        static KernelArg convert(Array_gpu<T, N>& arg) {
            return KernelArg::from_array<T>(arg.ptr(), arg.size());
        }

        static KernelArg convert(const Array_gpu<T, N>& arg) {
            return KernelArg::from_array<const T>(arg.ptr(), arg.size());
        }
    };
}

struct Kernel: kernel_launcher::PragmaKernel {
    Kernel(std::string name): kernel_launcher::PragmaKernel("gas_optics_kernels.cu", name) {}

    virtual kernel_launcher::KernelBuilder build() const {
        auto builder = kernel_launcher::PragmaKernel::build();
        builder.define("USECUDA", "1");
        builder.define("RESTRICTKEYWORD", "__restrict__");
#ifdef RTE_RRTMGP_USE_CBOOL
        builder.define("RTE_RRTMGP_USE_CBOOL", "1");
#endif
#ifdef RTE_RRTMGP_SINGLE_PRECISION
        builder.define("RTE_RRTMGP_SINGLE_PRECISION", "1");
#endif

        return builder;
    }

};

namespace rrtmgp_kernel_launcher_cuda
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
            grid = tunings["reorder123x321_kernel"].first;
            block = tunings["reorder123x321_kernel"].second;
        }

        reorder123x321_kernel<<<grid, block>>>(
                ni, nj, nk, arr_in, arr_out);
    }


    void reorder12x21(
            const int ni, const int nj,
            const Float* arr_in, Float* arr_out)
    {
        const int block_i = 32;
        const int block_j = 16;

        const int grid_i = ni/block_i + (ni%block_i > 0);
        const int grid_j = nj/block_j + (nj%block_j > 0);

        dim3 grid_gpu(grid_i, grid_j);
        dim3 block_gpu(block_i, block_j);

        reorder12x21_kernel<<<grid_gpu, block_gpu>>>(
                ni, nj, arr_in, arr_out);
    }


    void zero_array(const int ni, const int nj, const int nk, Float* arr)
    {
        const int block_i = 32;
        const int block_j = 16;
        const int block_k = 1;

        const int grid_i = ni/block_i + (ni%block_i > 0);
        const int grid_j = nj/block_j + (nj%block_j > 0);
        const int grid_k = nk/block_k + (nk%block_k > 0);

        dim3 grid_gpu(grid_i, grid_j, grid_k);
        dim3 block_gpu(block_i, block_j, block_k);

        zero_array_kernel<<<grid_gpu, block_gpu>>>(
                ni, nj, nk, arr);

    }


    void interpolation(
            const int ncol, const int nlay,
            const int ngas, const int nflav, const int neta, const int npres, const int ntemp,
            const int* flavor,
            const Float* press_ref_log,
            const Float* temp_ref,
            Float press_ref_log_delta,
            Float temp_ref_min,
            Float temp_ref_delta,
            Float press_ref_trop_log,
            const Float* vmr_ref,
            const Float* play,
            const Float* tlay,
            Float* col_gas,
            int* jtemp,
            Float* fmajor, Float* fminor,
            Float* col_mix,
            Bool* tropo,
            int* jeta,
            int* jpress)
    {
        const int block_col  = 4;
        const int block_lay  = 2;
        const int block_flav = 16;

        const int grid_col  = ncol /block_col  + (ncol%block_col   > 0);
        const int grid_lay  = nlay /block_lay  + (nlay%block_lay   > 0);
        const int grid_flav = nflav/block_flav + (nflav%block_flav > 0);

        dim3 grid_gpu(grid_col, grid_lay, grid_flav);
        dim3 block_gpu(block_col, block_lay, block_flav);

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
            const Float* tau_abs, const Float* tau_rayleigh,
            Float* tau, Float* ssa, Float* g)
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
            grid = tunings["combine_abs_and_rayleigh_kernel"].first;
            block = tunings["combine_abs_and_rayleigh_kernel"].second;
        }

        combine_abs_and_rayleigh_kernel<<<grid, block>>>(
                ncol, nlay, ngpt, tmin,
                tau_abs, tau_rayleigh,
                tau, ssa, g);
    }


    void compute_tau_rayleigh(
            const int ncol, const int nlay, const int nbnd, const int ngpt,
            const int ngas, const int nflav, const int neta, const int npres, const int ntemp,
            const int* gpoint_flavor,
            const int* gpoint_bands,
            const int* band_lims_gpt,
            const Float* krayl,
            int idx_h2o, const Float* col_dry, const Float* col_gas,
            const Float* fminor, const int* jeta,
            const Bool* tropo, const int* jtemp,
            Float* tau_rayleigh)
    {
        Tuner_map& tunings = Tuner::get_map();

        dim3 grid(ncol, nlay, ngpt);
        dim3 block;

        if (tunings.count("compute_tau_rayleigh_kernel") == 0)
        {
            std::tie(grid, block) = tune_kernel(
                    "compute_tau_rayleigh_kernel",
                    dim3(ncol, nlay, ngpt),
                    {1, 2, 4, 16, 24, 32}, {1, 2, 4}, {1, 2, 4, 8, 16},
                    compute_tau_rayleigh_kernel,
                    ncol, nlay, nbnd, ngpt,
                    ngas, nflav, neta, npres, ntemp,
                    gpoint_flavor,
                    gpoint_bands,
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
            grid = tunings["compute_tau_rayleigh_kernel"].first;
            block = tunings["compute_tau_rayleigh_kernel"].second;
        }

        compute_tau_rayleigh_kernel<<<grid, block>>>(
                ncol, nlay, nbnd, ngpt,
                ngas, nflav, neta, npres, ntemp,
                gpoint_flavor,
                gpoint_bands,
                band_lims_gpt,
                krayl,
                idx_h2o, col_dry, col_gas,
                fminor, jeta,
                tropo, jtemp,
                tau_rayleigh);
    }


    struct Gas_optical_depths_major_kernel
    {
        template<unsigned int I, unsigned int J, unsigned int K, class... Args>
        static void launch(dim3 grid, dim3 block, Args... args)
        {
            gas_optical_depths_major_kernel<I, J, K><<<grid, block>>>(args...);
        }
    };


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
            const Array_gpu<int, 2>& gpoint_flavor,
            const Array_gpu<int, 2>& band_lims_gpt,
            const Array_gpu<Float, 4>& kmajor,
            const Array_gpu<Float, 3>& kminor_lower,
            const Array_gpu<Float, 3>& kminor_upper,
            const Array_gpu<int, 2>& minor_limits_gpt_lower,
            const Array_gpu<int, 2>& minor_limits_gpt_upper,
            const Array_gpu<Bool, 1>& minor_scales_with_density_lower,
            const Array_gpu<Bool, 1>& minor_scales_with_density_upper,
            const Array_gpu<Bool, 1>& scale_by_complement_lower,
            const Array_gpu<Bool, 1>& scale_by_complement_upper,
            const Array_gpu<int, 1>& idx_minor_lower,
            const Array_gpu<int, 1>& idx_minor_upper,
            const Array_gpu<int, 1>& idx_minor_scaling_lower,
            const Array_gpu<int, 1>& idx_minor_scaling_upper,
            const Array_gpu<int, 1>& kminor_start_lower,
            const Array_gpu<int, 1>& kminor_start_upper,
            const Array_gpu<Bool, 2>& tropo,
            const Array_gpu<Float, 4>& col_mix, const Array_gpu<Float, 6>& fmajor,
            const Array_gpu<Float, 5>& fminor, const Array_gpu<Float, 2>& play,
            const Array_gpu<Float, 2>& tlay, const Array_gpu<Float, 3>& col_gas,
            const Array_gpu<int, 4>& jeta, const Array_gpu<int, 2>& jtemp,
            const Array_gpu<int, 2>& jpress,
            Array_gpu<Float, 3>& tau)
    {
        kernel_launcher::launch(
                Kernel("gas_optical_depths_major_kernel"),
                ncol, nlay, nband, ngpt,
                nflav, neta, npres, ntemp,
                gpoint_flavor, band_lims_gpt,
                kmajor, col_mix, fmajor, jeta,
                tropo, jtemp, jpress,
                tau);

        // Lower
        int idx_tropo = 1;

        kernel_launcher::launch(
                Kernel("gas_optical_depths_minor_kernel"),
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
                fminor, jeta, jtemp,
                tropo, tau,
                kernel_launcher::cuda_span<Float>(nullptr, 0));


        // Upper
        idx_tropo = 0;

        dim3 grid_gpu_min_2(ngpt, nlay, ncol);
        dim3 block_gpu_min_2;

        Tuner_map& tunings = Tuner::get_map();

        if (tunings.count("gas_optical_depths_minor_kernel_upper") == 0)
        {
            Float* tau_tmp = Tools_gpu::allocate_gpu<Float>(ngpt*nlay*ncol);
            std::tie(grid_gpu_min_2, block_gpu_min_2) =
                    tune_kernel_compile_time<Gas_optical_depths_minor_kernel>(
                            "gas_optical_depths_minor_kernel_upper",
                            dim3(ncol, nlay),
                            std::integer_sequence<unsigned int, 1, 2, 4, 8, 16>{},
                            std::integer_sequence<unsigned int, 1, 2, 4>{},
                            std::integer_sequence<unsigned int, 1, 2, 4, 8, 16, 32, 48, 64, 96, 128>{},
                            ncol, nlay, ngpt,
                            ngas, nflav, ntemp, neta,
                            nminorupper,
                            nminorkupper,
                            idx_h2o, idx_tropo,
                            gpoint_flavor.ptr(),
                            kminor_upper.ptr(),
                            minor_limits_gpt_upper.ptr(),
                            minor_scales_with_density_upper.ptr(),
                            scale_by_complement_upper.ptr(),
                            idx_minor_upper.ptr(),
                            idx_minor_scaling_upper.ptr(),
                            kminor_start_upper.ptr(),
                            play.ptr(), tlay.ptr(), col_gas.ptr(),
                            fminor.ptr(), jeta.ptr(), jtemp.ptr(),
                            tropo.ptr(), tau_tmp, nullptr);
            Tools_gpu::free_gpu<Float>(tau_tmp);

            tunings["gas_optical_depths_minor_kernel_upper"].first = grid_gpu_min_2;
            tunings["gas_optical_depths_minor_kernel_upper"].second = block_gpu_min_2;
        }
        else
        {
            grid_gpu_min_2 = tunings["gas_optical_depths_minor_kernel_upper"].first;
            block_gpu_min_2 = tunings["gas_optical_depths_minor_kernel_upper"].second;
        }

        run_kernel_compile_time<Gas_optical_depths_minor_kernel>(
                std::integer_sequence<unsigned int, 1, 2, 4, 8, 16>{},
                std::integer_sequence<unsigned int, 1, 2, 4>{},
                std::integer_sequence<unsigned int, 1, 2, 4, 8, 16, 32, 48, 64, 96, 128>{},
                grid_gpu_min_2, block_gpu_min_2,
                ncol, nlay, ngpt,
                ngas, nflav, ntemp, neta,
                nminorupper,
                nminorkupper,
                idx_h2o, idx_tropo,
                gpoint_flavor.ptr(),
                kminor_upper.ptr(),
                minor_limits_gpt_upper.ptr(),
                minor_scales_with_density_upper.ptr(),
                scale_by_complement_upper.ptr(),
                idx_minor_upper.ptr(),
                idx_minor_scaling_upper.ptr(),
                kminor_start_upper.ptr(),
                play.ptr(), tlay.ptr(), col_gas.ptr(),
                fminor.ptr(), jeta.ptr(), jtemp.ptr(),
                tropo.ptr(), tau.ptr(), nullptr);
    }


    void Planck_source(
            const int ncol, const int nlay, const int nbnd, const int ngpt,
            const int nflav, const int neta, const int npres, const int ntemp,
            const int nPlanckTemp,
            const Float* tlay,
            const Float* tlev,
            const Float* tsfc,
            const int sfc_lay,
            const Float* fmajor,
            const int* jeta,
            const Bool* tropo,
            const int* jtemp,
            const int* jpress,
            const int* gpoint_bands,
            const int* band_lims_gpt,
            const Float* pfracin,
            const Float temp_ref_min, const Float totplnk_delta,
            const Float* totplnk,
            const int* gpoint_flavor,
            Float* sfc_src,
            Float* lay_src,
            Float* lev_src_inc,
            Float* lev_src_dec,
            Float* sfc_src_jac)
    {
        Tuner_map& tunings = Tuner::get_map();

        const Float delta_Tsurf = Float(1.);

        const int block_gpt = 16;
        const int block_lay = 4;
        const int block_col = 2;

        const int grid_gpt = ngpt/block_gpt + (ngpt%block_gpt > 0);
        const int grid_lay = nlay/block_lay + (nlay%block_lay > 0);
        const int grid_col = ncol/block_col + (ncol%block_col > 0);

        dim3 grid_gpu(grid_gpt, grid_lay, grid_col);
        dim3 block_gpu(block_gpt, block_lay, block_col);

        if (tunings.count("Planck_source_kernel") == 0)
        {
            std::tie(grid_gpu, block_gpu) = tune_kernel(
                    "Planck_source_kernel",
                    dim3(ngpt, nlay, ncol),
                    {1, 2, 4},
                    {1, 2},
                    {1, 2, 4, 8, 16, 32, 48, 64, 96, 128, 256},
                    Planck_source_kernel,
                    ncol, nlay, nbnd, ngpt,
                    nflav, neta, npres, ntemp, nPlanckTemp,
                    tlay, tlev, tsfc, sfc_lay,
                    fmajor, jeta, tropo, jtemp,
                    jpress, gpoint_bands, band_lims_gpt,
                    pfracin, temp_ref_min, totplnk_delta,
                    totplnk, gpoint_flavor,
                    delta_Tsurf, sfc_src, lay_src,
                    lev_src_inc, lev_src_dec,
                    sfc_src_jac);

            tunings["Planck_source_kernel"].first = grid_gpu;
            tunings["Planck_source_kernel"].second = block_gpu;
        }
        else
        {
            grid_gpu = tunings["Planck_source_kernel"].first;
            block_gpu = tunings["Planck_source_kernel"].second;
        }

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
                lev_src_inc, lev_src_dec,
                sfc_src_jac);
    }
}
