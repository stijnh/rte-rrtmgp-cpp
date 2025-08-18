#include <chrono>

#include "kernel.h"
#include "rte_solver_kernels_cuda.h"
#include "tools_gpu.h"
#include "tuner.h"

#include <iomanip>


namespace
{
    #include "rte_solver_kernels.cu"

    using Tools_gpu::calc_grid_size;
}


namespace Rte_solver_kernels_cuda
{
    void apply_BC(const int ncol, const int nlay, const int ngpt, const Bool top_at_1,
                  const Float* inc_flux_dir, const Float* mu0, Float* gpt_flux_dir)
    {
        dim3 block_gpu(32, 32);
        dim3 grid_gpu = calc_grid_size(block_gpu, dim3(ncol, ngpt));

        apply_BC_kernel<<<grid_gpu, block_gpu>>>(ncol, nlay, ngpt, top_at_1, inc_flux_dir, mu0, gpt_flux_dir);
    }


    void apply_BC(const int ncol, const int nlay, const int ngpt, const Bool top_at_1, Float* gpt_flux_dn)
    {
        dim3 block_gpu(32, 32);
        dim3 grid_gpu = calc_grid_size(block_gpu, dim3(ncol, ngpt));

        apply_BC_kernel<<<grid_gpu, block_gpu>>>(ncol, nlay, ngpt, top_at_1, gpt_flux_dn);
    }


    void apply_BC(const int ncol, const int nlay, const int ngpt, const Bool top_at_1, const Float* inc_flux_dif, Float* gpt_flux_dn)
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
            const Float* tau, const Float* lay_source,
            const Float* lev_source,
            const Float* sfc_emis, const Float* sfc_src,
            const Float* inc_flux,
            Float* flux_up, Float* flux_dn,
            const Bool do_broadband, Float* flux_up_loc, Float* flux_dn_loc,
            const Bool do_jacobians, const Float* sfc_src_jac, Float* flux_up_jac)
    {
        Float eps = std::numeric_limits<Float>::epsilon();

        const int flx_size = ncol*(nlay+1)*ngpt;
        const int opt_size = ncol*nlay*ngpt;
        const int sfc_size = ncol*ngpt;;

        Float* trans = Tools_gpu::allocate_gpu<Float>(opt_size);
        Float* source_dn = Tools_gpu::allocate_gpu<Float>(opt_size);
        Float* source_up = Tools_gpu::allocate_gpu<Float>(opt_size);
        Float* radn_dn = Tools_gpu::allocate_gpu<Float>(flx_size);
        Float* radn_up = Tools_gpu::allocate_gpu<Float>(flx_size);
        Float* radn_up_jac = Tools_gpu::allocate_gpu<Float>(flx_size);

        dim3 block_gpu2d(64, 2);
        dim3 grid_gpu2d = calc_grid_size(block_gpu2d, dim3(ncol, ngpt));

        const int top_level = top_at_1 ? 0 : nlay;
        const Float tau_thres = sqrt(sqrt(eps));


        // Upper boundary condition.
        if (inc_flux == nullptr)
            Rte_solver_kernels_cuda::apply_BC(ncol, nlay, ngpt, top_at_1, flux_dn);
        else
            Rte_solver_kernels_cuda::apply_BC(ncol, nlay, ngpt, top_at_1, inc_flux, flux_dn);

        kernel_launcher::launch(
                Kernel("lw_solver_noscat_kernel", "src_kernels_cuda/rte_solver_kernels.cu", {top_at_1}),
                ncol, nlay, ngpt, tau_thres,
                secants, weights, tau, lay_source,
                lev_source,
                sfc_emis, sfc_src, flux_up, flux_dn, sfc_src_jac,
                flux_up_jac, trans,
                source_dn, source_up);

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
            const Float* tau, const Float* lay_source,
            const Float* lev_source,
            const Float* sfc_emis, const Float* sfc_src,
            const Float* inc_flux,
            Float* flux_up, Float* flux_dn,
            const Bool do_broadband, Float* flux_up_loc, Float* flux_dn_loc,
            const Bool do_jacobians, const Float* sfc_src_jac, Float* flux_up_jac)
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
            const Float* tau, const Float* ssa, const Float* g,
            const Float* mu0,
            const Float* sfc_alb_dir, const Float* sfc_alb_dif,
            const Float* inc_flux_dir,
            Float* flux_up, Float* flux_dn, Float* flux_dir,
            const Bool has_dif_bc, const Float* inc_flux_dif,
            const Bool do_broadband, Float* flux_up_loc, Float* flux_dn_loc, Float* flux_dir_loc)
    {
        const int opt_size = ncol*nlay*ngpt;
        const int alb_size = ncol*ngpt;
        const int flx_size = ncol*(nlay+1)*ngpt;

        Float* r_dif = Tools_gpu::allocate_gpu<Float>(opt_size);
        Float* t_dif = Tools_gpu::allocate_gpu<Float>(opt_size);
        Float* source_up = Tools_gpu::allocate_gpu<Float>(opt_size);
        Float* source_dn = Tools_gpu::allocate_gpu<Float>(opt_size);
        Float* source_sfc = Tools_gpu::allocate_gpu<Float>(alb_size);
        Float* albedo = Tools_gpu::allocate_gpu<Float>(flx_size);
        Float* src = Tools_gpu::allocate_gpu<Float>(flx_size);
        Float* denom = Tools_gpu::allocate_gpu<Float>(opt_size);

        // Step0. Upper boundary condition. At this stage, flux_dn contains the diffuse radiation only.
        Rte_solver_kernels_cuda::apply_BC(ncol, nlay, ngpt, top_at_1, inc_flux_dir, mu0, flux_dir);
        if (inc_flux_dif == nullptr)
            Rte_solver_kernels_cuda::apply_BC(ncol, nlay, ngpt, top_at_1, flux_dn);
        else
            Rte_solver_kernels_cuda::apply_BC(ncol, nlay, ngpt, top_at_1, inc_flux_dif, flux_dn);

        // Step 1.
        kernel_launcher::launch(
                Kernel("sw_solver_kernel", "src_kernels_cuda/rte_solver_kernels.cu", {top_at_1}),
                ncol, nlay, ngpt, tau, ssa, g, mu0, r_dif, t_dif,
                sfc_alb_dir, sfc_alb_dif,
                source_up, source_dn, source_sfc,
                flux_up, flux_dn, flux_dir,
                albedo, src, denom);

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
