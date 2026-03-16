#include <chrono>
#include "rte_solver_kernels_cuda.h"
#include "tools_gpu.h"
#include "tuner.h"
#include <float.h>
#include "types.h"
#include "kernel_float.h"
#include <iomanip>

#include "../include/tools_gpu.h"


namespace
{
    #include "rte_solver_kernels.cu"

    using Tools_gpu::calc_grid_size;
}


namespace Rte_solver_kernels_cuda
{
    void apply_BC(const int ncol, const int nlay, const int ngpt, const Bool top_at_1,
                  const FloatFlux* inc_flux_dir, const Float* mu0, FloatFlux* gpt_flux_dir)
    {
        dim3 block_gpu(32, 32);
        dim3 grid_gpu = calc_grid_size(block_gpu, dim3(ncol, ngpt));

        apply_BC_kernel<<<grid_gpu, block_gpu>>>(ncol, nlay, ngpt, top_at_1, inc_flux_dir, mu0, gpt_flux_dir);
    }


    void apply_BC(const int ncol, const int nlay, const int ngpt, const Bool top_at_1, FloatFlux* gpt_flux_dn)
    {
        dim3 block_gpu(32, 32);
        dim3 grid_gpu = calc_grid_size(block_gpu, dim3(ncol, ngpt));

        apply_BC_kernel<<<grid_gpu, block_gpu>>>(ncol, nlay, ngpt, top_at_1, gpt_flux_dn);
    }


    void apply_BC(const int ncol, const int nlay, const int ngpt, const Bool top_at_1, const FloatFlux* inc_flux_dif, FloatFlux* gpt_flux_dn)
    {
        dim3 block_gpu(32, 32);
        dim3 grid_gpu = calc_grid_size(block_gpu, dim3(ncol, ngpt));

        apply_BC_kernel<<<grid_gpu, block_gpu>>>(ncol, nlay, ngpt, top_at_1, inc_flux_dif, gpt_flux_dn);
    }


    void lw_secants_array(
            const int ncol, const int ngpt, const int n_gauss_quad, const int max_gauss_pts,
            const Float* gauss_Ds, Float* secants)
    {
        dim3 block_gpu(32, 32);
        dim3 grid_gpu = calc_grid_size(block_gpu, dim3(ncol, ngpt, n_gauss_quad));

        lw_secants_array_kernel<<<grid_gpu, block_gpu>>>(
                ncol, ngpt, n_gauss_quad, max_gauss_pts,
                gauss_Ds, secants);
    }

    template <Bool top_at_1>
    void lw_solver_noscat_impl(
            const int ncol, const int nlay, const int ngpt, const int nmus,
            const Float* secants, const Float* weights,
            const FloatTau* tau, const FloatSource* lay_source,
            const FloatSource* lev_source,
            const FloatSurface* sfc_emis, const FloatSurface* sfc_src,
            const FloatFlux* inc_flux,
            FloatFlux* flux_up, FloatFlux* flux_dn,
            const Bool do_broadband, FloatFlux* flux_up_loc, FloatFlux* flux_dn_loc,
            const Bool do_jacobians, const FloatSurface* sfc_src_jac, FloatFlux* flux_up_jac)
    {
        using C = constants::lw_solver_noscat_kernel;
        using FloatIntermediate = C::intermediate_type;
        Float eps = std::numeric_limits<Float>::epsilon();

        const int flx_size = ncol*(nlay+1)*ngpt;
        const int opt_size = ncol*nlay*ngpt;
        const int sfc_size = ncol*ngpt;;

        FloatIntermediate* trans = Tools_gpu::allocate_gpu<FloatIntermediate>(opt_size);
        FloatIntermediate* source_dn = Tools_gpu::allocate_gpu<FloatIntermediate>(opt_size);
        FloatIntermediate* source_up = Tools_gpu::allocate_gpu<FloatIntermediate>(opt_size);
        FloatFlux* radn_dn = Tools_gpu::allocate_gpu<FloatFlux>(flx_size);
        FloatFlux* radn_up = Tools_gpu::allocate_gpu<FloatFlux>(flx_size);
        FloatFlux* radn_up_jac = Tools_gpu::allocate_gpu<FloatFlux>(flx_size);

        const int top_level = top_at_1 ? 0 : nlay;
        const Float tau_thres = sqrt(sqrt(eps));

        // Upper boundary condition.
        if (inc_flux == nullptr)
            Rte_solver_kernels_cuda::apply_BC(ncol, nlay, ngpt, top_at_1, flux_dn);
        else
            Rte_solver_kernels_cuda::apply_BC(ncol, nlay, ngpt, top_at_1, inc_flux, flux_dn);

        dim3 block_gpu(C::block_size_x, C::block_size_y);
        dim3 grid_gpu = calc_grid_size(block_gpu, dim3(ncol / C::vector_size, ngpt));

        lw_solver_noscat_kernel<top_at_1,
                    C::block_size_x,
                    C::block_size_y,
                    C::loop_unroll_factor_init,
                    C::loop_unroll_factor_nlay,
                    C::vector_size,
                    Float,
                    C::tau_type,
                    C::source_type,
                    C::surface_type,
                    C::flux_type,
                    C::compute_type,
                    C::intermediate_type><<<grid_gpu, block_gpu>>>(ncol, nlay, ngpt, tau_thres,
                secants, weights, tau, lay_source,
                lev_source,
                sfc_emis, sfc_src, flux_up, flux_dn, sfc_src_jac,
                flux_up_jac, trans,
                source_dn, source_up);

        dim3 block_gpu2d(64, 2);
        dim3 grid_gpu2d = calc_grid_size(block_gpu2d, dim3(ncol, ngpt));

        apply_BC_kernel_lw<<<grid_gpu2d, block_gpu2d>>>(top_level, ncol, nlay, ngpt, top_at_1, flux_dn, radn_dn);

        if (nmus > 1)
        {
            for (int imu=1; imu<nmus; ++imu)
            {
                throw std::runtime_error("Not implemented due to lacking test case");
                /*
                lw_solver_noscat_step_1_kernel<<<grid_1, block_1>>>(
                        ncol, nlay, ngpt, eps, top_at_1,
                        secants+imu, weights+imu, tau, lay_source,
                        lev_source_inc, lev_source_dec,
                        sfc_emis, sfc_src, radn_up, radn_dn, sfc_src_jac,
                        radn_up_jac, tau_loc, trans,
                        source_dn, source_up, source_sfc, sfc_albedo, source_sfc_jac);

                lw_solver_noscat_step_2_kernel<<<grid_2, block_2>>>(
                        ncol, nlay, ngpt, eps, top_at_1,
                        secants+imu, weights+imu, tau, lay_source,
                        lev_source_inc, lev_source_dec,
                        sfc_emis, sfc_src,
                        radn_up, radn_dn, sfc_src_jac,
                        radn_up_jac, tau_loc, trans,
                        source_dn, source_up, source_sfc, sfc_albedo, source_sfc_jac);

                lw_solver_noscat_step_3_kernel<<<grid_3, block_3>>>(
                        ncol, nlay, ngpt, eps, top_at_1,
                        secants+imu, weights+imu, tau, lay_source,
                        lev_source_inc, lev_source_dec,
                        sfc_emis, sfc_src, radn_up, radn_dn, sfc_src_jac,
                        radn_up_jac, tau_loc, trans,
                        source_dn, source_up, source_sfc, sfc_albedo, source_sfc_jac);

                add_fluxes_kernel<<<grid_gpu3d, block_gpu3d>>>(
                        ncol, nlay+1, ngpt,
                        radn_up, radn_dn, radn_up_jac,
                        flux_up, flux_dn, flux_up_jac);
                        */
            }
        }

        Tools_gpu::free_gpu(trans);
        Tools_gpu::free_gpu(source_dn);
        Tools_gpu::free_gpu(source_up);
        Tools_gpu::free_gpu(radn_dn);
        Tools_gpu::free_gpu(radn_up);
        Tools_gpu::free_gpu(radn_up_jac);
    }

    void lw_solver_noscat(
            const int ncol, const int nlay, const int ngpt, const Bool top_at_1, const int nmus,
            const Float* secants, const Float* weights,
            const FloatTau* tau, const FloatSource* lay_source,
            const FloatSource* lev_source,
            const FloatSurface* sfc_emis, const FloatSurface* sfc_src,
            const FloatFlux* inc_flux,
            FloatFlux* flux_up, FloatFlux* flux_dn,
            const Bool do_broadband, FloatFlux* flux_up_loc, FloatFlux* flux_dn_loc,
            const Bool do_jacobians, const FloatSurface* sfc_src_jac, FloatFlux* flux_up_jac)
    {
        if (top_at_1) {
            lw_solver_noscat_impl<true>(
                    ncol, nlay, ngpt, nmus,
                    secants, weights,
                    tau, lay_source,
                    lev_source,
                    sfc_emis, sfc_src,
                    inc_flux,
                    flux_up, flux_dn,
                    do_broadband, flux_up_loc, flux_dn_loc,
                    do_jacobians, sfc_src_jac, flux_up_jac);
        } else {
            lw_solver_noscat_impl<false>(
                    ncol, nlay, ngpt, nmus,
                    secants, weights,
                    tau, lay_source,
                    lev_source,
                    sfc_emis, sfc_src,
                    inc_flux,
                    flux_up, flux_dn,
                    do_broadband, flux_up_loc, flux_dn_loc,
                    do_jacobians, sfc_src_jac, flux_up_jac);
        }
    }

    void sw_solver_2stream(
            const int ncol, const int nlay, const int ngpt, const Bool top_at_1,
            const FloatTau* tau, const FloatOptical* ssa, const FloatOptical* g,
            const Float* mu0,
            const FloatSurface* sfc_alb_dir, const FloatSurface* sfc_alb_dif,
            const FloatFlux* inc_flux_dir,
            FloatFlux* flux_up, FloatFlux* flux_dn, FloatFlux* flux_dir,
            const Bool has_dif_bc, const FloatFlux* inc_flux_dif,
            const Bool do_broadband, FloatFlux* flux_up_loc, FloatFlux* flux_dn_loc, FloatFlux* flux_dir_loc)
    {
        using C = constants::sw_solver_kernel;
        using FloatIntermediate = C::intermediate_type;
        const int opt_size = ncol*nlay*ngpt;
        const int alb_size = ncol*ngpt;
        const int flx_size = ncol*(nlay+1)*ngpt;

        FloatIntermediate* r_dif = Tools_gpu::allocate_gpu<FloatIntermediate>(opt_size);
        FloatIntermediate* t_dif = Tools_gpu::allocate_gpu<FloatIntermediate>(opt_size);
        FloatSource* source_up = Tools_gpu::allocate_gpu<FloatSource>(opt_size);
        FloatSource* source_dn = Tools_gpu::allocate_gpu<FloatSource>(opt_size);
        FloatSurface* source_sfc = Tools_gpu::allocate_gpu<FloatSurface>(alb_size);
        FloatIntermediate* albedo = Tools_gpu::allocate_gpu<FloatIntermediate>(flx_size);
        FloatIntermediate* src = Tools_gpu::allocate_gpu<FloatIntermediate>(flx_size);
        FloatIntermediate* denom = Tools_gpu::allocate_gpu<FloatIntermediate>(opt_size);

        // Step0. Upper boundary condition. At this stage, flux_dn contains the diffuse radiation only.
        Rte_solver_kernels_cuda::apply_BC(ncol, nlay, ngpt, top_at_1, inc_flux_dir, mu0, flux_dir);
        if (inc_flux_dif == nullptr)
            Rte_solver_kernels_cuda::apply_BC(ncol, nlay, ngpt, top_at_1, flux_dn);
        else
            Rte_solver_kernels_cuda::apply_BC(ncol, nlay, ngpt, top_at_1, inc_flux_dif, flux_dn);

        // Step 1.
        dim3 block_gpu(C::block_size_x, C::block_size_y);
        dim3 grid_gpu = calc_grid_size(block_gpu, dim3(ncol / C::vector_size, ngpt));
        if (top_at_1) {
            sw_solver_kernel<true,
                    C::block_size_x,
                    C::block_size_y,
                    C::vector_size,
                    C::loop_unroll_factor_nlay,
                    Float,
                    C::compute_type,
                    C::stream_type,
                    C::tau_type,
                    C::source_type,
                    C::surface_type,
                    C::flux_type,
                    C::intermediate_type><<<grid_gpu, block_gpu>>>(ncol, nlay, ngpt, tau, ssa, g, mu0, r_dif, t_dif,
                sfc_alb_dir, sfc_alb_dif,
                source_up, source_dn, source_sfc,
                flux_up, flux_dn, flux_dir,
                albedo, src, denom);
        }
        else {
            sw_solver_kernel<false,
                    C::block_size_x,
                    C::block_size_y,
                    C::vector_size,
                    C::loop_unroll_factor_nlay,
                    Float,
                    C::compute_type,
                    C::stream_type,
                    C::tau_type,
                    C::source_type,
                    C::surface_type,
                    C::flux_type,
                    C::intermediate_type><<<grid_gpu, block_gpu>>>(ncol, nlay, ngpt, tau, ssa, g, mu0, r_dif, t_dif,
                sfc_alb_dir, sfc_alb_dif,
                source_up, source_dn, source_sfc,
                flux_up, flux_dn, flux_dir,
                albedo, src, denom);
        }

        Tools_gpu::free_gpu(r_dif);
        Tools_gpu::free_gpu(t_dif);
        Tools_gpu::free_gpu(source_up);
        Tools_gpu::free_gpu(source_dn);
        Tools_gpu::free_gpu(source_sfc);
        Tools_gpu::free_gpu(albedo);
        Tools_gpu::free_gpu(src);
        Tools_gpu::free_gpu(denom);
    }
}
