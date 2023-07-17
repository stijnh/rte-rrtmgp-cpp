#include <chrono>

#include "rte_kernel_launcher_cuda.h"
#include "tools_gpu.h"
#include "Array.h"
#include "tuner.h"
#include "Kernel.h"

#include <iomanip>


namespace
{
    #include "rte_solver_kernels.cu"
}


namespace rte_kernel_launcher_cuda
{
    void apply_BC(const int ncol, const int nlay, const int ngpt, const Bool top_at_1,
                  const Float* inc_flux_dir, const Float* mu0, Float* gpt_flux_dir)
    {
        const int block_col = 32;
        const int block_gpt = 32;

        const int grid_col = ncol/block_col + (ncol%block_col > 0);
        const int grid_gpt = ngpt/block_gpt + (ngpt%block_gpt > 0);

        dim3 grid_gpu(grid_col, grid_gpt);
        dim3 block_gpu(block_col, block_gpt);

        apply_BC_kernel<<<grid_gpu, block_gpu>>>(ncol, nlay, ngpt, top_at_1, inc_flux_dir, mu0, gpt_flux_dir);
    }


    void apply_BC(const int ncol, const int nlay, const int ngpt, const Bool top_at_1, Float* gpt_flux_dn)
    {
        const int block_col = 32;
        const int block_gpt = 32;

        const int grid_col = ncol/block_col + (ncol%block_col > 0);
        const int grid_gpt = ngpt/block_gpt + (ngpt%block_gpt > 0);

        dim3 grid_gpu(grid_col, grid_gpt);
        dim3 block_gpu(block_col, block_gpt);
        apply_BC_kernel<<<grid_gpu, block_gpu>>>(ncol, nlay, ngpt, top_at_1, gpt_flux_dn);
    }


    void apply_BC(const int ncol, const int nlay, const int ngpt, const Bool top_at_1, const Float* inc_flux_dif, Float* gpt_flux_dn)
    {
        const int block_col = 32;
        const int block_gpt = 32;

        const int grid_col = ncol/block_col + (ncol%block_col > 0);
        const int grid_gpt = ngpt/block_gpt + (ngpt%block_gpt > 0);

        dim3 grid_gpu(grid_col, grid_gpt);
        dim3 block_gpu(block_col, block_gpt);

        apply_BC_kernel<<<grid_gpu, block_gpu>>>(ncol, nlay, ngpt, top_at_1, inc_flux_dif, gpt_flux_dn);
    }


    void lw_secants_array(
            const int ncol, const int ngpt, const int n_gauss_quad, const int max_gauss_pts,
            const Float* gauss_Ds, Float* secants)
    {
        const int block_col = 32;
        const int block_gpt = 32;

        const int grid_col = ncol/block_col + (ncol%block_col > 0);
        const int grid_gpt = ngpt/block_gpt + (ngpt%block_gpt > 0);

        dim3 grid_gpu(grid_col, grid_gpt, n_gauss_quad);
        dim3 block_gpu(block_col, block_gpt, 1);

        lw_secants_array_kernel<<<grid_gpu, block_gpu>>>(
                ncol, ngpt, n_gauss_quad, max_gauss_pts,
                gauss_Ds, secants);
    }


    void lw_solver_noscat_gaussquad(
            const int ncol, const int nlay, const int ngpt, const Bool top_at_1, const int nmus,
            const Float* secants, const Float* weights,
            const Float* tau, const Float* lay_source,
            const Float* lev_source_inc, const Float* lev_source_dec,
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

        Float* source_sfc = Tools_gpu::allocate_gpu<Float>(sfc_size);
        Float* source_sfc_jac = Tools_gpu::allocate_gpu<Float>(sfc_size);
        Float* sfc_albedo = Tools_gpu::allocate_gpu<Float>(sfc_size);
        Float* tau_loc = Tools_gpu::allocate_gpu<Float>(opt_size);
        Float* trans = Tools_gpu::allocate_gpu<Float>(opt_size);
        Float* source_dn = Tools_gpu::allocate_gpu<Float>(opt_size);
        Float* source_up = Tools_gpu::allocate_gpu<Float>(opt_size);
        Float* radn_dn = Tools_gpu::allocate_gpu<Float>(flx_size);
        Float* radn_up = Tools_gpu::allocate_gpu<Float>(flx_size);
        Float* radn_up_jac = Tools_gpu::allocate_gpu<Float>(flx_size);

        const int block_col2d = 64;
        const int block_gpt2d = 2;

        const int grid_col2d = ncol/block_col2d + (ncol%block_col2d > 0);
        const int grid_gpt2d = ngpt/block_gpt2d + (ngpt%block_gpt2d > 0);

        dim3 grid_gpu2d(grid_col2d, grid_gpt2d);
        dim3 block_gpu2d(block_col2d, block_gpt2d);

        const int block_col3d = 96;
        const int block_lay3d = 1;
        const int block_gpt3d = 1;

        const int grid_col3d = ncol/block_col3d + (ncol%block_col3d > 0);
        const int grid_lay3d = (nlay+1)/block_lay3d + ((nlay+1)%block_lay3d > 0);
        const int grid_gpt3d = ngpt/block_gpt3d + (ngpt%block_gpt3d > 0);

        dim3 grid_gpu3d(grid_col3d, grid_lay3d, grid_gpt3d);
        dim3 block_gpu3d(block_col3d, block_lay3d, block_gpt3d);

        const int top_level = top_at_1 ? 0 : nlay;


        // Upper boundary condition.
        if (inc_flux == nullptr)
            rte_kernel_launcher_cuda::apply_BC(ncol, nlay, ngpt, top_at_1, flux_dn);
        else
            rte_kernel_launcher_cuda::apply_BC(ncol, nlay, ngpt, top_at_1, inc_flux, flux_dn);


        // Step 1.
        Tuner_map& tunings = Tuner::get_map();

        dim3 grid_1, block_1;

        if (tunings.count("lw_step_1") == 0)
        {
                   Float* flux_up_tmp = Tools_gpu::allocate_gpu<Float>(ngpt*(nlay+1)*ncol);
                   Float* flux_dn_tmp = Tools_gpu::allocate_gpu<Float>(ngpt*(nlay+1)*ncol);
                   Float* flux_up_jac_tmp = Tools_gpu::allocate_gpu<Float>(ngpt*(nlay+1)*ncol);

            std::tie(grid_1, block_1) = tune_kernel(
                    "lw_step_1",
                    dim3(ncol, nlay, ngpt),
                    {8, 16, 24, 32, 48, 64, 96, 128, 256, 512, 1024}, {1, 2, 4, 8}, {1},
                    lw_solver_noscat_step_1_kernel,
                    ncol, nlay, ngpt, eps, top_at_1,
                    secants, weights, tau, lay_source,
                    lev_source_inc, lev_source_dec,
                    sfc_emis, sfc_src, flux_up_tmp, flux_dn_tmp, sfc_src_jac,
                    flux_up_jac_tmp, tau_loc, trans,
                    source_dn, source_up, source_sfc, sfc_albedo, source_sfc_jac);
                   
                    Tools_gpu::free_gpu<Float>(flux_up_tmp);
                    Tools_gpu::free_gpu<Float>(flux_dn_tmp);
                    Tools_gpu::free_gpu<Float>(flux_up_jac_tmp);

            tunings["lw_step_1"].first = grid_1;
            tunings["lw_step_1"].second = block_1;
        }
        else
        {
            grid_1 = tunings["lw_step_1"].first;
            block_1 = tunings["lw_step_1"].second;
        }

        lw_solver_noscat_step_1_kernel<<<grid_1, block_1>>>(
                ncol, nlay, ngpt, eps, top_at_1,
                secants, weights, tau, lay_source,
                lev_source_inc, lev_source_dec,
                sfc_emis, sfc_src, flux_up, flux_dn, sfc_src_jac,
                flux_up_jac, tau_loc, trans,
                source_dn, source_up, source_sfc, sfc_albedo, source_sfc_jac);


        // Step 2.
        dim3 grid_2, block_2;

        if (tunings.count("lw_step_2") == 0)
        {
            Float* flux_up_tmp = Tools_gpu::allocate_gpu<Float>(ngpt*(nlay+1)*ncol);
            Float* flux_dn_tmp = Tools_gpu::allocate_gpu<Float>(ngpt*(nlay+1)*ncol);
            Float* flux_up_jac_tmp = Tools_gpu::allocate_gpu<Float>(ngpt*(nlay+1)*ncol);
            
            std::tie(grid_2, block_2) = tune_kernel(
                    "lw_step_2",
                    dim3(ncol, ngpt),
                    {64, 128, 256, 384, 512, 768, 1024}, {1, 2, 4}, {1},
                    lw_solver_noscat_step_2_kernel,
                    ncol, nlay, ngpt, eps, top_at_1,
                    secants, weights, tau, lay_source,
                    lev_source_inc, lev_source_dec,
                    sfc_emis, sfc_src, flux_up_tmp, flux_dn_tmp, sfc_src_jac,
                    flux_up_jac_tmp, tau_loc, trans,
                    source_dn, source_up, source_sfc, sfc_albedo, source_sfc_jac);

            Tools_gpu::free_gpu<Float>(flux_up_tmp);
            Tools_gpu::free_gpu<Float>(flux_dn_tmp);
            Tools_gpu::free_gpu<Float>(flux_up_jac_tmp);
            
            tunings["lw_step_2"].first = grid_2;
            tunings["lw_step_2"].second = block_2;
        }
        else
        {
            grid_2 = tunings["lw_step_2"].first;
            block_2 = tunings["lw_step_2"].second;
        }

        lw_solver_noscat_step_2_kernel<<<grid_2, block_2>>>(
                ncol, nlay, ngpt, eps, top_at_1,
                secants, weights, tau, lay_source,
                lev_source_inc, lev_source_dec,
                sfc_emis, sfc_src, flux_up, flux_dn, sfc_src_jac,
                flux_up_jac, tau_loc, trans,
                source_dn, source_up, source_sfc, sfc_albedo, source_sfc_jac);


        // Step 3.
        dim3 grid_3, block_3;

        if (tunings.count("lw_step_3") == 0)
        {
            Float* flux_up_tmp = Tools_gpu::allocate_gpu<Float>(ngpt*(nlay+1)*ncol);
            Float* flux_dn_tmp = Tools_gpu::allocate_gpu<Float>(ngpt*(nlay+1)*ncol);
            Float* flux_up_jac_tmp = Tools_gpu::allocate_gpu<Float>(ngpt*(nlay+1)*ncol);
            
            std::tie(grid_3, block_3) = tune_kernel(
                    "lw_step_3",
                    dim3(ncol, nlay+1, ngpt),
                    {8, 16, 24, 32, 48, 64, 96, 128, 256}, {1, 2, 4, 8}, {1},
                    lw_solver_noscat_step_3_kernel,
                    ncol, nlay, ngpt, eps, top_at_1,
                    secants, weights, tau, lay_source,
                    lev_source_inc, lev_source_dec,
                    sfc_emis, sfc_src, flux_up_tmp, flux_dn_tmp, sfc_src_jac,
                    flux_up_jac_tmp, tau_loc, trans,
                    source_dn, source_up, source_sfc, sfc_albedo, source_sfc_jac);

            Tools_gpu::free_gpu<Float>(flux_up_tmp);
            Tools_gpu::free_gpu<Float>(flux_dn_tmp);
            Tools_gpu::free_gpu<Float>(flux_up_jac_tmp);

            tunings["lw_step_3"].first = grid_3;
            tunings["lw_step_3"].second = block_3;
        }
        else
        {
            grid_3 = tunings["lw_step_3"].first;
            block_3 = tunings["lw_step_3"].second;
        }

        lw_solver_noscat_step_3_kernel<<<grid_3, block_3>>>(
                ncol, nlay, ngpt, eps, top_at_1,
                secants, weights, tau, lay_source,
                lev_source_inc, lev_source_dec,
                sfc_emis, sfc_src, flux_up, flux_dn, sfc_src_jac,
                flux_up_jac, tau_loc, trans,
                source_dn, source_up, source_sfc, sfc_albedo, source_sfc_jac);

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

        Tools_gpu::free_gpu(source_sfc);
        Tools_gpu::free_gpu(source_sfc_jac);
        Tools_gpu::free_gpu(sfc_albedo);
        Tools_gpu::free_gpu(tau_loc);
        Tools_gpu::free_gpu(trans);
        Tools_gpu::free_gpu(source_dn);
        Tools_gpu::free_gpu(source_up);
        Tools_gpu::free_gpu(radn_dn);
        Tools_gpu::free_gpu(radn_up);
        Tools_gpu::free_gpu(radn_up_jac);
    }


    void sw_solver_2stream(
            const int ncol, const int nlay, const int ngpt, const Bool top_at_1,
            const Array_gpu<Float, 3>& tau, const Array_gpu<Float, 3>& ssa, const Array_gpu<Float, 3>& g,
            const Array_gpu<Float, 1>& mu0,
            const Array_gpu<Float, 2>& sfc_alb_dir, const Array_gpu<Float, 2>& sfc_alb_dif,
            const Array_gpu<Float, 2>& inc_flux_dir,
            Array_gpu<Float, 3>& flux_up, Array_gpu<Float, 3>& flux_dn, Array_gpu<Float, 3>& flux_dir,
            const Bool has_dif_bc, const Array_gpu<Float, 2>& inc_flux_dif,
            const Bool do_broadband, Array_gpu<Float, 3>& flux_up_loc, Array_gpu<Float, 3>& flux_dn_loc, Array_gpu<Float, 3>& flux_dir_loc)
    {
        const int opt_size = ncol*nlay*ngpt;
        const int alb_size = ncol*ngpt;
        const int flx_size = ncol*(nlay+1)*ngpt;

        Array_gpu<Float, 3> r_dif({ncol, nlay, ngpt});
        Array_gpu<Float, 3> t_dif({ncol, nlay, ngpt});
        Array_gpu<Float, 3> source_up({ncol, nlay, ngpt});
        Array_gpu<Float, 3> source_dn({ncol, nlay, ngpt});
        Array_gpu<Float, 2> source_sfc({ncol, ngpt});
        Array_gpu<Float, 3> albedo({ncol, nlay + 1, ngpt});
        Array_gpu<Float, 3> src({ncol, nlay + 1, ngpt});
        Array_gpu<Float, 3> denom({ncol, nlay, ngpt});

        // Step0. Upper boundary condition. At this stage, flux_dn contains the diffuse radiation only.
        rte_kernel_launcher_cuda::apply_BC(ncol, nlay, ngpt, top_at_1, inc_flux_dir.ptr(), mu0.ptr(), flux_dir.ptr());
        if (inc_flux_dif.ptr() == nullptr)
            rte_kernel_launcher_cuda::apply_BC(ncol, nlay, ngpt, top_at_1, flux_dn.ptr());
        else
            rte_kernel_launcher_cuda::apply_BC(ncol, nlay, ngpt, top_at_1, inc_flux_dif.ptr(), flux_dn.ptr());


        // Step 1.
        kernel_launcher::launch(
                Kernel("sw_source_2stream_kernel", "rte_solver_kernels.cu", {int(top_at_1)}),
                ncol, nlay, ngpt, tau, ssa, g, mu0, r_dif, t_dif,
                sfc_alb_dir, source_up, source_dn, source_sfc, flux_dir);


        // Step 2.
        kernel_launcher::launch(
            Kernel("sw_adding_kernel", "rte_solver_kernels.cu", {int(top_at_1)}),
            ncol, nlay, ngpt, top_at_1,
            sfc_alb_dif, (const Array_gpu<Float, 3>&) r_dif, (const Array_gpu<Float, 3>&) t_dif,
            (const Array_gpu<Float, 3>&) source_dn, (const Array_gpu<Float, 3>&) source_up, (const Array_gpu<Float, 3>&) source_sfc,
            flux_up, flux_dn, flux_dir,
            albedo, src, denom);
    }
}
