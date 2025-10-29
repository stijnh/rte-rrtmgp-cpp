#include <float.h>

#include "types.h"
#include "kernel_float.h"


template<typename TF> __device__ constexpr TF k_min();
template<> __device__ constexpr double k_min() { return 1.e-12; }
template<> __device__ constexpr float k_min() { return 0.010; }

__global__
void lw_secants_array_kernel(
        const int ncol, const int ngpt, const int n_gauss_quad, const int max_gauss_pts,
        const Float* __restrict__ gauss_Ds, Float* __restrict__ secants)
{
    const int icol = blockIdx.x*blockDim.x + threadIdx.x;
    const int igpt = blockIdx.y*blockDim.y + threadIdx.y;
    const int imu = blockIdx.z;

    if ( (icol < ncol) && (igpt < ngpt) && (imu < n_gauss_quad) )
    {
        const int idx_s = icol + igpt*ncol + imu*ncol*ngpt;
        const int idx_g = imu + (n_gauss_quad-1)*max_gauss_pts;

        secants[idx_s] = gauss_Ds[idx_g];
    }
}

template <
        int loop_unroll_factor_nlay,
        int vector_size,
        typename tau_type,
        typename source_type,
        typename surface_type,
        typename flux_type,
        typename compute_type,
        typename intermediate_type
>
__device__
void lw_transport_noscat_kernel(
        const int ivcol,
        const int igpt,
        const int nvcol,
        const int nlay,
        const int ngpt,
        const Bool top_at_1,
        kernel_float::vec_ptr<compute_type, vector_size, const tau_type> tau,
        kernel_float::vec_ptr<compute_type, vector_size, const intermediate_type> trans,
        const kernel_float::vec<compute_type, vector_size> sfc_albedo,
        kernel_float::vec_ptr<compute_type, vector_size, const intermediate_type> source_dn,
        kernel_float::vec_ptr<compute_type, vector_size, const intermediate_type> source_up,
        kernel_float::vec<compute_type, vector_size> source_sfc,
        kernel_float::vec_ptr<compute_type, vector_size, flux_type> radn_up,
        kernel_float::vec_ptr<compute_type, vector_size, flux_type> radn_dn,
        const kernel_float::vec<compute_type, vector_size> source_sfc_jac,
        kernel_float::vec_ptr<compute_type, vector_size, flux_type> radn_up_jac,
        const kernel_float::vec<compute_type, vector_size> radn_dn_top,
        compute_type scaling)
{
    using Float = compute_type;

    if (top_at_1)
    {
        const int idx_top = ivcol + igpt*nvcol*(nlay+1);
        auto radn_dn_loc = radn_dn_top;
        radn_dn[idx_top] = radn_dn_loc * scaling;

#pragma unroll loop_unroll_factor_nlay
        for (int ilev=0; ilev<(nlay); ++ilev)
        {
            const int idx1 = ivcol + (ilev+1)*nvcol + igpt*nvcol*(nlay+1);
            const int idx3 = ivcol + ilev*nvcol + igpt*nvcol*nlay;
            radn_dn_loc = fma(trans[idx3], radn_dn_loc, source_dn[idx3]);
            radn_dn[idx1] = radn_dn_loc * scaling;
        }

        auto radn_up_loc = fma(radn_dn_loc, sfc_albedo, source_sfc);
        auto radn_jac_loc = source_sfc_jac;

        const int idx_bot = ivcol + nlay*nvcol + igpt*nvcol*(nlay+1);
        radn_up[idx_bot] = radn_up_loc * scaling;
        radn_up_jac[idx_bot] = radn_jac_loc * scaling;

#pragma unroll loop_unroll_factor_nlay
        for (int ilev=nlay-1; ilev>=0; --ilev)
        {
            const int idx3 = ivcol + ilev*nvcol + igpt*nvcol*nlay;
            radn_up_loc = fma(trans[idx3], radn_up_loc, source_up[idx3]);
            radn_jac_loc = trans[idx3] * radn_jac_loc;

            const int idx1 = ivcol + ilev*nvcol + igpt*nvcol*(nlay+1);
            radn_up[idx1] = radn_up_loc * scaling;
            radn_up_jac[idx1] = radn_jac_loc * scaling;
        }
    }
    else
    {
        const int idx_top = ivcol + nlay*nvcol + igpt*nvcol*(nlay+1);
        auto radn_dn_loc = radn_dn_top;
        radn_dn[idx_top] = radn_dn_loc * scaling;

#pragma unroll loop_unroll_factor_nlay
        for (int ilev=(nlay-1); ilev>=0; --ilev)
        {
            const int idx1 = ivcol + ilev*nvcol + igpt*nvcol*(nlay+1);
            const int idx3 = ivcol + ilev*nvcol + igpt*nvcol*nlay;
            radn_dn_loc = fma(trans[idx3], radn_dn_loc, source_dn[idx3]);
            radn_dn[idx1] = radn_dn_loc * scaling;
        }

        auto radn_up_loc = fma(radn_dn_loc, sfc_albedo, source_sfc);
        auto radn_jac_loc = source_sfc_jac;

        const int idx_bot = ivcol + igpt*nvcol*(nlay+1);
        radn_up[idx_bot] = radn_up_loc * scaling;
        radn_up_jac[idx_bot] = radn_jac_loc * scaling;

#pragma unroll loop_unroll_factor_nlay
        for (int ilev=0; ilev<nlay; ++ilev)
        {
            const int idx3 = ivcol + ilev*nvcol + igpt*nvcol*nlay;
            radn_up_loc = fma(trans[idx3], radn_up_loc, source_up[idx3]);
            radn_jac_loc = trans[idx3] * radn_jac_loc;

            const int idx1 = ivcol + (ilev+1)*nvcol + igpt*nvcol*(nlay+1);
            radn_up[idx1] = radn_up_loc * scaling;
            radn_up_jac[idx1] = radn_jac_loc * scaling;
        }
    }
}


#pragma kernel problem_size(ncol, ngpt)
#pragma kernel block_size(block_size_x, block_size_y)
#pragma kernel grid_divisor(block_size_x * vector_size, block_size_y)
#pragma kernel buffer(D_ptr[ncol*ngpt])
#pragma kernel buffer(weight_ptr[1])
#pragma kernel buffer(tau_ptr[ncol*nlay*ngpt])
#pragma kernel buffer(lay_source_ptr[ncol*nlay*ngpt])
#pragma kernel buffer(lev_source_ptr[ngpt * ncol * (nlay + 1)])
#pragma kernel buffer(sfc_emis_ptr[ngpt*ncol])
#pragma kernel buffer(sfc_src_ptr[ngpt*ncol])
#pragma kernel buffer(radn_up_ptr[ngpt*ncol*(nlay+1)])
#pragma kernel buffer(radn_dn_ptr[ngpt*ncol*(nlay+1)])
#pragma kernel buffer(sfc_src_jac_ptr[ngpt*ncol])
#pragma kernel buffer(radn_up_jac_ptr[ngpt*ncol*(nlay+1)])
#pragma kernel buffer(trans_ptr[ncol*nlay*ngpt])
#pragma kernel buffer(source_dn_ptr[ncol*nlay*ngpt])
#pragma kernel buffer(source_up_ptr[ncol*nlay*ngpt])
template <
        Bool top_at_1,
        int block_size_x,
        int block_size_y,
        int loop_unroll_factor_init,
        int loop_unroll_factor_nlay,
        int vector_size,
        typename float_type,
        typename tau_type,
        typename source_type,
        typename surface_type,
        typename flux_type,
        typename compute_type,
        typename intermediate_type
>
__global__
void lw_solver_noscat_kernel(
        const int ncol,
        const int nlay,
        const int ngpt,
        const float_type tau_thres,
        const float_type* __restrict__ D_ptr,
        const float_type* __restrict__ weight_ptr,
        const tau_type* __restrict__ tau_ptr,
        const source_type* __restrict__ lay_source_ptr,
        const source_type* __restrict__ lev_source_ptr,
        const surface_type* __restrict__ sfc_emis_ptr,
        const surface_type* __restrict__ sfc_src_ptr,
        flux_type* __restrict__ radn_up_ptr,
        flux_type* __restrict__ radn_dn_ptr,
        const surface_type* __restrict__ sfc_src_jac_ptr,
        flux_type* __restrict__ radn_up_jac_ptr,
        intermediate_type* __restrict__ trans_ptr,
        intermediate_type* __restrict__ source_dn_ptr,
        intermediate_type* __restrict__ source_up_ptr
)
__launch_bounds__(block_size_x*block_size_y)
{
    using Float = compute_type;
    auto D = kernel_float::wrap_ptr<compute_type, vector_size>(D_ptr);
    auto weight = kernel_float::wrap_ptr<compute_type>(weight_ptr);
    auto tau = kernel_float::wrap_ptr<compute_type, vector_size>(tau_ptr);
    auto lay_source = kernel_float::wrap_ptr<compute_type, vector_size>(lay_source_ptr);
    auto lev_source = kernel_float::wrap_ptr<compute_type, vector_size>(lev_source_ptr);
    auto sfc_emis = kernel_float::wrap_ptr<compute_type, vector_size>(sfc_emis_ptr);
    auto sfc_src = kernel_float::wrap_ptr<compute_type, vector_size>(sfc_src_ptr);
    auto radn_up = kernel_float::wrap_ptr<compute_type, vector_size>(radn_up_ptr);
    auto radn_dn = kernel_float::wrap_ptr<compute_type, vector_size>(radn_dn_ptr);
    auto sfc_src_jac = kernel_float::wrap_ptr<compute_type, vector_size>(sfc_src_jac_ptr);
    auto radn_up_jac = kernel_float::wrap_ptr<compute_type, vector_size>(radn_up_jac_ptr);
    auto trans = kernel_float::wrap_ptr<compute_type, vector_size>(trans_ptr);
    auto source_dn = kernel_float::wrap_ptr<compute_type, vector_size>(source_dn_ptr);
    auto source_up = kernel_float::wrap_ptr<compute_type, vector_size>(source_up_ptr);

    const int ivcol = blockIdx.x*blockDim.x + threadIdx.x;
    const int igpt = blockIdx.y*blockDim.y + threadIdx.y;
    const int nvcol = ncol / vector_size;

    if ( (ivcol < nvcol) && (igpt < ngpt) )
    {

#pragma unroll loop_unroll_factor_init
        for (int ilay=0; ilay<nlay; ++ilay) {
            const int idx_lay = ivcol + ilay * nvcol + igpt * nvcol * nlay;
            const int idx_lev = ivcol + ilay * nvcol + igpt * nvcol * (nlay + 1);
            const int idx_lev_p = ivcol + (ilay + 1) * nvcol + igpt * nvcol * (nlay + 1);

            const int idx_D = ivcol + igpt * nvcol;

            auto tau_loc = tau[idx_lay] * D[idx_D];
            auto trans_loc = exp(-tau_loc);
            auto trans_loc_inv = -expm1(-tau_loc); // `1 - trans_loc`

            const auto fact = where(
                    tau_loc > (tau_thres),
                    trans_loc_inv / tau_loc - trans_loc,
                    tau_loc *
                    (Float(.5) + tau_loc * (Float(-1. / 3.) + tau_loc * Float(1. / 8.))));

            auto src_inc = trans_loc_inv * lev_source[idx_lev_p] +
                           Float(2.) * fact * (lay_source[idx_lay] - lev_source[idx_lev_p]);
            auto src_dec = trans_loc_inv * lev_source[idx_lev] +
                           Float(2.) * fact * (lay_source[idx_lay] - lev_source[idx_lev]);

            trans[idx_lay] = trans_loc;
            source_dn[idx_lay] = top_at_1 ? src_inc : src_dec;
            source_up[idx_lay] = top_at_1 ? src_dec : src_inc;
        }

        const int idx2d = ivcol + igpt*nvcol;
        auto sfc_albedo = Float(1.) - sfc_emis[idx2d];
        auto source_sfc = sfc_emis[idx2d] * sfc_src[idx2d];
        auto source_sfc_jac = sfc_emis[idx2d] * sfc_src_jac[idx2d];

        const compute_type pi = 3.1415926535897932384626433832795028841971693993751058209749445923078164062;
        auto scaling = pi * weight[0];
        const int idx_top = ivcol + (top_at_1 ? 0 : nlay)*nvcol + igpt*nvcol*(nlay+1);
        const auto radn_dn_top = radn_dn.read(idx_top) / (Float(2.) * scaling);

        lw_transport_noscat_kernel<
            loop_unroll_factor_nlay,
            vector_size,
            tau_type,
            source_type,
            surface_type,
            flux_type,
            compute_type,
            intermediate_type
        >(
            ivcol, igpt, nvcol, nlay, ngpt, top_at_1, tau, trans, sfc_albedo, source_dn,
            source_up, source_sfc, radn_up, radn_dn, source_sfc_jac, radn_up_jac, radn_dn_top, scaling
        );
    }
}


template<Bool top_at_1> __global__
void sw_source_kernel(
        const int ncol, const int nlay, const int ngpt, const Bool _top_at_1,
        Float* __restrict__ r_dir, Float* __restrict__ t_dir, Float* __restrict__ t_noscat,
        const Float* __restrict__ sfc_alb_dir, Float* __restrict__ source_up, Float* __restrict__ source_dn,
        Float* __restrict__ source_sfc, Float* __restrict__ flux_dir)
{
    const int icol = blockIdx.x*blockDim.x + threadIdx.x;
    const int igpt = blockIdx.y*blockDim.y + threadIdx.y;

    if ( (icol < ncol) && (igpt < ngpt) )
    {
        if (top_at_1)
        {
            for (int ilay=0; ilay<nlay; ++ilay)
            {
                const int idx_lay  = icol + ilay*ncol + igpt*nlay*ncol;
                const int idx_lev1 = icol + ilay*ncol + igpt*(nlay+1)*ncol;
                const int idx_lev2 = icol + (ilay+1)*ncol + igpt*(nlay+1)*ncol;
                source_up[idx_lay] = r_dir[idx_lay] * flux_dir[idx_lev1];
                source_dn[idx_lay] = t_dir[idx_lay] * flux_dir[idx_lev1];
                flux_dir[idx_lev2] = t_noscat[idx_lay] * flux_dir[idx_lev1];

            }
            const int sfc_idx = icol + igpt*ncol;
            const int flx_idx = icol + nlay*ncol + igpt*(nlay+1)*ncol;
            source_sfc[sfc_idx] = flux_dir[flx_idx] * sfc_alb_dir[icol];
        }
        else
        {
            for (int ilay=nlay-1; ilay>=0; --ilay)
            {
                const int idx_lay  = icol + ilay*ncol + igpt*nlay*ncol;
                const int idx_lev1 = icol + (ilay)*ncol + igpt*(nlay+1)*ncol;
                const int idx_lev2 = icol + (ilay+1)*ncol + igpt*(nlay+1)*ncol;
                source_up[idx_lay] = r_dir[idx_lay] * flux_dir[idx_lev2];   //uses updated flux_dir from previous iteration
                source_dn[idx_lay] = t_dir[idx_lay] * flux_dir[idx_lev2];   //uses updated flux_dir from previous
                flux_dir[idx_lev1] = t_noscat[idx_lay] * flux_dir[idx_lev2];//updates flux_dir for 0 to nlay-1

            }
            const int sfc_idx = icol + igpt*ncol;
            const int flx_idx = icol + igpt*(nlay+1)*ncol;
            source_sfc[sfc_idx] = flux_dir[flx_idx] * sfc_alb_dir[icol];
        }
    }
}

__global__
void apply_BC_kernel_lw(const int isfc, int ncol, const int nlay, const int ngpt, const Bool top_at_1, const FloatFlux* __restrict__ inc_flux, FloatFlux* __restrict__ flux_dn)
{
    const int icol = blockIdx.x*blockDim.x + threadIdx.x;
    const int igpt = blockIdx.y*blockDim.y + threadIdx.y;

    if ( (icol < ncol) && (igpt < ngpt) )
    {
        const int idx_in  = icol + isfc*ncol + igpt*ncol*(nlay+1);
        const int idx_out = (top_at_1) ? icol + igpt*ncol*(nlay+1) : icol + nlay*ncol + igpt*ncol*(nlay+1);
        flux_dn[idx_out] = FloatFlux(inc_flux[idx_in]);
    }
}

__global__
void apply_BC_kernel(const int ncol, const int nlay, const int ngpt, const Bool top_at_1, const FloatFlux* __restrict__ inc_flux, FloatFlux* __restrict__ flux_dn)
{
    const int icol = blockIdx.x*blockDim.x + threadIdx.x;
    const int igpt = blockIdx.y*blockDim.y + threadIdx.y;
    if ( (icol < ncol) && (igpt < ngpt) )
    {
        const int idx_out = icol + ((top_at_1 ? 0 : (nlay * ncol))) + (igpt * ncol * (nlay + 1));
        const int idx_in = icol + (igpt * ncol);
        flux_dn[idx_out] = FloatFlux(inc_flux[idx_in]);
    }
}

__global__
void apply_BC_kernel(const int ncol, const int nlay, const int ngpt, const Bool top_at_1, const FloatFlux* __restrict__ inc_flux, const Float* __restrict__ factor, FloatFlux* __restrict__ flux_dn)
{
    const int icol = blockIdx.x*blockDim.x + threadIdx.x;
    const int igpt = blockIdx.y*blockDim.y + threadIdx.y;
    if ( (icol < ncol) && (igpt < ngpt) )
    {
        const int idx_out = icol + ((top_at_1 ? 0 : (nlay * ncol))) + (igpt * ncol * (nlay + 1));
        const int idx_in = icol + (igpt * ncol);

        flux_dn[idx_out] = FloatFlux(Float(inc_flux[idx_in]) * factor[icol]);
    }
}

__global__
void apply_BC_kernel(const int ncol, const int nlay, const int ngpt, const Bool top_at_1, FloatFlux* __restrict__ flux_dn)
{
    const int icol = blockIdx.x*blockDim.x + threadIdx.x;
    const int igpt = blockIdx.y*blockDim.y + threadIdx.y;
    if ( (icol < ncol) && (igpt < ngpt) )
    {
        const int idx_out = icol + ((top_at_1 ? 0 : (nlay * ncol))) + (igpt * ncol * (nlay + 1));
        flux_dn[idx_out] = FloatFlux(0);
    }
}

__global__
void sw_2stream_kernel(
        const int ncol, const int nlay, const int ngpt, const Float tmin,
        const Float* __restrict__ tau, const Float* __restrict__ ssa,
        const Float* __restrict__ g, const Float* __restrict__ mu0,
        Float* __restrict__ r_dif, Float* __restrict__ t_dif,
        Float* __restrict__ r_dir, Float* __restrict__ t_dir,
        Float* __restrict__ t_noscat)
{
    const int icol = blockIdx.x*blockDim.x + threadIdx.x;
    const int ilay = blockIdx.y*blockDim.y + threadIdx.y;
    const int igpt = blockIdx.z*blockDim.z + threadIdx.z;

    if ( (icol < ncol) && (ilay < nlay) && (igpt < ngpt) )
    {
        const int idx = icol + ilay*ncol + igpt*nlay*ncol;
        const Float mu0_inv = Float(1.)/mu0[icol];
        const Float gamma1 = (Float(8.) - ssa[idx] * (Float(5.) + Float(3.) * g[idx])) * Float(.25);
        const Float gamma2 = Float(3.) * (ssa[idx] * (Float(1.) -          g[idx])) * Float(.25);
        const Float gamma3 = (Float(2.) - Float(3.) * mu0[icol] *          g[idx])  * Float(.25);
        const Float gamma4 = Float(1.) - gamma3;

        const Float alpha1 = gamma1 * gamma4 + gamma2 * gamma3;
        const Float alpha2 = gamma1 * gamma3 + gamma2 * gamma4;

        const Float k = sqrt(max((gamma1 - gamma2) * (gamma1 + gamma2), k_min<Float>()));
        const Float exp_minusktau = exp(-tau[idx] * k);
        const Float exp_minus2ktau = exp_minusktau * exp_minusktau;

        const Float rt_term = Float(1.) / (k      * (Float(1.) + exp_minus2ktau) +
                                     gamma1 * (Float(1.) - exp_minus2ktau));
        r_dif[idx] = rt_term * gamma2 * (Float(1.) - exp_minus2ktau);
        t_dif[idx] = rt_term * Float(2.) * k * exp_minusktau;
        t_noscat[idx] = exp(-tau[idx] * mu0_inv);

        const Float k_mu     = k * mu0[icol];
        const Float k_gamma3 = k * gamma3;
        const Float k_gamma4 = k * gamma4;

        const Float fact = (abs(Float(1.) - k_mu*k_mu) > tmin) ? Float(1.) - k_mu*k_mu : tmin;
        const Float rt_term2 = ssa[idx] * rt_term / fact;

        r_dir[idx] = rt_term2  * ((Float(1.) - k_mu) * (alpha2 + k_gamma3)   -
                                  (Float(1.) + k_mu) * (alpha2 - k_gamma3) * exp_minus2ktau -
                                   Float(2.) * (k_gamma3 - alpha2 * k_mu)  * exp_minusktau * t_noscat[idx]);

        t_dir[idx] = -rt_term2 * ((Float(1.) + k_mu) * (alpha1 + k_gamma4) * t_noscat[idx]   -
                                  (Float(1.) - k_mu) * (alpha1 - k_gamma4) * exp_minus2ktau * t_noscat[idx] -
                                   Float(2.) * (k_gamma4 + alpha1 * k_mu)  * exp_minusktau);
    }
}

/*
template<typename Float>__global__
void sw_source_adding_kernel(const int ncol, const int nlay, const int ngpt, const Bool top_at_1,
                             const Float* __restrict__ sfc_alb_dir, const Float* __restrict__ sfc_alb_dif,
                             Float* __restrict__ r_dif, Float* __restrict__ t_dif,
                             Float* __restrict__ r_dir, Float* __restrict__ t_dir, Float* __restrict__ t_noscat,
                             Float* __restrict__ flux_up, Float* __restrict__ flux_dn, Float* __restrict__ flux_dir,
                             Float* __restrict__ source_up, Float* __restrict__ source_dn, Float* __restrict__ source_sfc,
                             Float* __restrict__ albedo, Float* __restrict__ src, Float* __restrict__ denom)
{
    const int icol = blockIdx.x*blockDim.x + threadIdx.x;
    const int igpt = blockIdx.y*blockDim.y + threadIdx.y;

    if ( (icol < ncol) && (igpt < ngpt) )
    {
        sw_source_kernel(icol, igpt, ncol, nlay, top_at_1, r_dir, t_dir,
                         t_noscat, sfc_alb_dir, source_up, source_dn, source_sfc, flux_dir);

        sw_adding_kernel(icol, igpt, ncol, nlay, top_at_1, sfc_alb_dif,
                         r_dif, t_dif, source_dn, source_up, source_sfc,
                         flux_up, flux_dn, flux_dir, albedo, src, denom);
    }
}


__global__
void lw_solver_noscat_gaussquad_kernel(
        const int ncol, const int nlay, const int ngpt, const Float eps, const Bool top_at_1, const int nmus,
        const Float* __restrict__ secants, const Float* __restrict__ weights,
        const Float* __restrict__ tau, const Float* __restrict__ lay_source,
        const Float* __restrict__ lev_source_inc, const Float* __restrict__ lev_source_dec, const Float* __restrict__ sfc_emis,
        const Float* __restrict__ sfc_src, Float* __restrict__ radn_up, Float* __restrict__ radn_dn,
        const Float* __restrict__ sfc_src_jac, Float* __restrict__ radn_up_jac, Float* __restrict__ tau_loc,
        Float* __restrict__ trans, Float* __restrict__ source_dn, Float* __restrict__ source_up,
        Float* __restrict__ source_sfc, Float* __restrict__ sfc_albedo, Float* __restrict__ source_sfc_jac,
        Float* __restrict__ flux_up, Float* __restrict__ flux_dn, Float* __restrict__ flux_up_jac)
{
    const int icol = blockIdx.x*blockDim.x + threadIdx.x;
    const int igpt = blockIdx.y*blockDim.y + threadIdx.y;

    // CvH ONLY TO MAKE IT COMPILE. REMOVE !!!!
    Float* ds = secants;

    if ( (icol < ncol) && (igpt < ngpt) )
    {
        lw_solver_noscat_kernel(
                icol, igpt, ncol, nlay, ngpt, eps, top_at_1, ds[0], weights[0], tau, lay_source,
                lev_source_inc, lev_source_dec, sfc_emis, sfc_src, flux_up, flux_dn, sfc_src_jac,
                flux_up_jac, tau_loc, trans, source_dn, source_up, source_sfc, sfc_albedo, source_sfc_jac);

        const int top_level = top_at_1 ? 0 : nlay;
        apply_BC_kernel_lw(icol, igpt, top_level, ncol, nlay, ngpt, top_at_1, flux_dn, radn_dn);

        if (nmus > 1)
        {
            for (int imu=1; imu<nmus; ++imu)
            {
                lw_solver_noscat_kernel(
                        icol, igpt, ncol, nlay, ngpt, eps, top_at_1, ds[imu], weights[imu], tau, lay_source,
                        lev_source_inc, lev_source_dec, sfc_emis, sfc_src, radn_up, radn_dn, sfc_src_jac,
                        radn_up_jac, tau_loc, trans, source_dn, source_up, source_sfc, sfc_albedo, source_sfc_jac);

                for (int ilev=0; ilev<(nlay+1); ++ilev)
                {
                    const int idx = icol + ilev*ncol + igpt*ncol*(nlay+1);
                    flux_up[idx] += radn_up[idx];
                    flux_dn[idx] += radn_dn[idx];
                    flux_up_jac[idx] += radn_up_jac[idx];
                }
            }
        }
    }
}
*/


__global__
void add_fluxes_kernel(
        const int ncol, const int nlev, const int ngpt,
        const Float* __restrict__ radn_up, const Float* __restrict__ radn_dn, const Float* __restrict__ radn_up_jac,
        Float* __restrict__ flux_up, Float* __restrict__ flux_dn, Float* __restrict__ flux_up_jac)
{
    const int icol = blockIdx.x*blockDim.x + threadIdx.x;
    const int ilev = blockIdx.y*blockDim.y + threadIdx.y;
    const int igpt = blockIdx.z*blockDim.z + threadIdx.z;

    if ( (icol < ncol) && (ilev < nlev) && (igpt < ngpt) )
    {
        const int idx = icol + ilev*ncol + igpt*ncol*nlev;

        flux_up[idx] += radn_up[idx];
        flux_dn[idx] += radn_dn[idx];
        flux_up_jac[idx] += radn_up_jac[idx];
    }
}


template<typename TF> __device__ constexpr TF tmin();
template<> __forceinline__ __device__ constexpr double tmin() { return DBL_EPSILON; }
template<> __forceinline__ __device__ constexpr float tmin() { return FLT_EPSILON; }

template<
        typename TF,
        int vector_size,
        typename compute_type,
        typename tau_type,
        typename intermediate_type
>
__device__
void sw_2stream_function(
        const int ivcol, const int ilay, const int igpt,
        const int nvcol, const int nlay, const int ngpt,
        kernel_float::vec_ptr<TF, vector_size, const tau_type> tau,
        kernel_float::vec_ptr<TF, vector_size, const tau_type> ssa,
        kernel_float::vec_ptr<TF, vector_size, const tau_type> g,
        kernel_float::vec_ptr<TF, vector_size, const compute_type> mu0,
        kernel_float::vec_ptr<TF, vector_size, intermediate_type> r_dif,
        kernel_float::vec_ptr<TF, vector_size, intermediate_type> t_dif,
        kernel_float::vec_ptr<TF, vector_size, compute_type> r_dir,
        kernel_float::vec_ptr<TF, vector_size, compute_type> t_dir,
        kernel_float::vec_ptr<TF, vector_size, compute_type> t_noscat_out)
{
    using vec = kernel_float::vec<TF, vector_size>;
    const int idx = ivcol + ilay*nvcol + igpt*nlay*nvcol;

    const vec mu0_inv = TF(1.)/mu0[ivcol];
    const vec gamma1 = (TF(8.) - ssa[idx] * (TF(5.) + TF(3.) * g[idx])) * TF(.25);
    const vec gamma2 = TF(3.) * (ssa[idx] * (TF(1.) - g[idx])) * TF(.25);
    const vec gamma3 = (TF(2.) - TF(3.) * mu0[ivcol] * g[idx])  * TF(.25);
    const vec gamma4 = TF(1.) - gamma3;

    const vec alpha1 = gamma1 * gamma4 + gamma2 * gamma3;
    const vec alpha2 = gamma1 * gamma3 + gamma2 * gamma4;

    const auto k = sqrt(max((gamma1 - gamma2) * (gamma1 + gamma2), k_min<TF>()));
    const auto exp_minusktau = exp(-tau[idx] * k);
    const auto exp_minus2ktau = exp_minusktau * exp_minusktau;
    const auto one_minus_exp_minus2ktau = -expm1(TF(-2) * tau[idx] * k);

    //const auto rt_term = TF(1.) / (k      * (TF(1.) + exp_minus2ktau) +
    //                             gamma1 * one_minus_exp_minus2ktau);
    const auto rt_term = TF(1.) / fma(gamma1 - k, one_minus_exp_minus2ktau, 2 * k);
    r_dif[idx] = rt_term * gamma2 * one_minus_exp_minus2ktau;
    t_dif[idx] = rt_term * TF(2.) * k * exp_minusktau;

    auto t_noscat = exp(-tau[idx] * mu0_inv);
    auto one_minus_t_noscat = -expm1(-tau[idx] * mu0_inv);
    *t_noscat_out = t_noscat;

    const auto k_mu     = k * mu0[ivcol];
    const auto k_gamma3 = k * gamma3;
    const auto k_gamma4 = k * gamma4;

    const auto fact = kernel_float::where((abs(TF(1.) - k_mu*k_mu) > tmin<TF>()), TF(1.) - k_mu*k_mu, tmin<TF>());
    const auto rt_term2 = ssa[idx] * rt_term / fact;

    *r_dir = rt_term2  * ((TF(1.) - k_mu) * (alpha2 + k_gamma3)   -
                          (TF(1.) + k_mu) * (alpha2 - k_gamma3) * exp_minus2ktau -
                          TF(2.) * (k_gamma3 - alpha2 * k_mu)  * exp_minusktau * t_noscat);

    *t_dir = -rt_term2 * ((TF(1.) + k_mu) * (alpha1 + k_gamma4) * t_noscat   -
                          (TF(1.) - k_mu) * (alpha1 - k_gamma4) * exp_minus2ktau * t_noscat -
                          TF(2.) * (k_gamma4 + alpha1 * k_mu)  * exp_minusktau);

    // fix thanks to peter ukkonen (see https://github.com/earth-system-radiation/rte-rrtmgp/pull/39#issuecomment-1026698541)
    *r_dir = max(tmin<TF>(), min(*r_dir, one_minus_t_noscat));
    *t_dir = max(tmin<TF>(), min(*t_dir, one_minus_t_noscat - *r_dir));
}


#pragma kernel problem_size(ncol, ngpt)
#pragma kernel block_size(block_size_x, block_size_y)
#pragma kernel grid_divisor(block_size_x * vector_size, block_size_y)
#pragma kernel buffer(tau_ptr[ngpt*nlay*ncol])
#pragma kernel buffer(ssa_ptr[ngpt*nlay*ncol])
#pragma kernel buffer(g_ptr[ngpt*nlay*ncol])
#pragma kernel buffer(mu0_ptr[ncol])
#pragma kernel buffer(r_dif_ptr[ncol*nlay*ngpt])
#pragma kernel buffer(t_dif_ptr[ncol*nlay*ngpt])
#pragma kernel buffer(sfc_alb_dir_ptr[ngpt*ncol])
#pragma kernel buffer(sfc_alb_dif_ptr[ngpt*ncol])
#pragma kernel buffer(source_up_ptr[ngpt*nlay*ncol])
#pragma kernel buffer(source_dn_ptr[ngpt*nlay*ncol])
#pragma kernel buffer(source_sfc_ptr[ngpt*ncol])
#pragma kernel buffer(flux_up_ptr[ngpt*(nlay+1)*ncol])
#pragma kernel buffer(flux_dn_ptr[ngpt*(nlay+1)*ncol])
#pragma kernel buffer(flux_dir_ptr[ngpt*(nlay+1)*ncol])
#pragma kernel buffer(albedo_ptr[ncol*(nlay+1)*ngpt])
#pragma kernel buffer(src_ptr[ncol*(nlay+1)*ngpt])
#pragma kernel buffer(denom_ptr[ngpt*ncol*nlay])
template<
        int top_at_1 = 0,
        int block_size_x,
        int block_size_y,
        int vector_size,
        int loop_unroll_factor_nlay,
        typename float_type,
        typename stream_type,
        typename compute_type,
        typename tau_type,
        typename source_type,
        typename surface_type,
        typename flux_type,
        typename intermediate_type
>
__global__
void sw_solver_kernel(
        const int ncol, const int nlay, const int ngpt,
        const tau_type* __restrict__ tau_ptr,
        const tau_type* __restrict__ ssa_ptr,
        const tau_type* __restrict__ g_ptr,
        const float_type* __restrict__ mu0_ptr,
        intermediate_type* __restrict__ r_dif_ptr,
        intermediate_type* __restrict__ t_dif_ptr,
        const surface_type* __restrict__ sfc_alb_dir_ptr,
        const surface_type* __restrict__ sfc_alb_dif_ptr,
        source_type* __restrict__ source_up_ptr,
        source_type* __restrict__ source_dn_ptr,
        surface_type* __restrict__ source_sfc_ptr,
        flux_type* __restrict__ flux_up_ptr,
        flux_type* __restrict__ flux_dn_ptr,
        flux_type* __restrict__ flux_dir_ptr,
        intermediate_type* __restrict__ albedo_ptr,
        intermediate_type* __restrict__ src_ptr,
        intermediate_type* __restrict__ denom_ptr
)
__launch_bounds__(block_size_x * block_size_y)
{
    using Float = compute_type;
    using vec = kernel_float::vec<compute_type, vector_size>;

    const int ivcol = blockIdx.x*blockDim.x + threadIdx.x;
    const int igpt = blockIdx.y*blockDim.y + threadIdx.y;
    const int nvcol = ncol / vector_size;

    auto tau = kernel_float::wrap_ptr<compute_type, vector_size>(tau_ptr);
    auto ssa = kernel_float::wrap_ptr<compute_type, vector_size>(ssa_ptr);
    auto g = kernel_float::wrap_ptr<compute_type, vector_size>(g_ptr);
    auto mu0 = kernel_float::wrap_ptr<compute_type, vector_size>(mu0_ptr);
    auto r_dif = kernel_float::wrap_ptr<compute_type, vector_size>(r_dif_ptr);
    auto t_dif = kernel_float::wrap_ptr<compute_type, vector_size>(t_dif_ptr);
    auto sfc_alb_dir = kernel_float::wrap_ptr<compute_type, vector_size>(sfc_alb_dir_ptr);
    auto sfc_alb_dif = kernel_float::wrap_ptr<compute_type, vector_size>(sfc_alb_dif_ptr);
    auto source_up = kernel_float::wrap_ptr<compute_type, vector_size>(source_up_ptr);
    auto source_dn = kernel_float::wrap_ptr<compute_type, vector_size>(source_dn_ptr);
    auto source_sfc = kernel_float::wrap_ptr<compute_type, vector_size>(source_sfc_ptr);
    auto flux_up = kernel_float::wrap_ptr<compute_type, vector_size>(flux_up_ptr);
    auto flux_dn = kernel_float::wrap_ptr<compute_type, vector_size>(flux_dn_ptr);
    auto flux_dir = kernel_float::wrap_ptr<compute_type, vector_size>(flux_dir_ptr);
    auto albedo = kernel_float::wrap_ptr<compute_type, vector_size>(albedo_ptr);
    auto src = kernel_float::wrap_ptr<compute_type, vector_size>(src_ptr);
    auto denom = kernel_float::wrap_ptr<compute_type, vector_size>(denom_ptr);

    if ( (ivcol < nvcol) && (igpt < ngpt) )
    {
        if (top_at_1)
        {
            auto flux_dir_loc = kernel_float::into_vec(flux_dir[ivcol + igpt*(nlay+1)*nvcol]);

#pragma unroll loop_unroll_factor_nlay
            for (int ilay=0; ilay<nlay; ++ilay)
            {
                vec r_dir, t_dir, t_noscat;
                sw_2stream_function<stream_type, vector_size, compute_type, tau_type, intermediate_type>(
                        ivcol, ilay, igpt,
                        nvcol, nlay, ngpt,
                        tau, ssa, g, mu0,
                        r_dif, t_dif,
                        kernel_float::wrap_ptr<compute_type, vector_size>(r_dir.data()),
                        kernel_float::wrap_ptr<compute_type, vector_size>(t_dir.data()),
                        kernel_float::wrap_ptr<compute_type, vector_size>(t_noscat.data()));

                const int idx_lay  = ivcol + ilay*nvcol + igpt*nlay*nvcol;
                const int idx_lev2 = ivcol + (ilay+1)*nvcol + igpt*(nlay+1)*nvcol;

                source_up[idx_lay] = r_dir * flux_dir_loc;
                source_dn[idx_lay] = t_dir * flux_dir_loc;

                flux_dir_loc = t_noscat * flux_dir_loc;
                flux_dir[idx_lev2] = flux_dir_loc;
            }

            const int sfc_idx = ivcol + igpt*nvcol;
            auto src_loc = flux_dir_loc * sfc_alb_dir[ivcol];
            auto albedo_loc = sfc_alb_dif[sfc_idx];
            source_sfc[sfc_idx] = src_loc;

            const int sfc_idx_3d = ivcol + nlay*nvcol + igpt*(nlay+1)*nvcol;
            albedo[sfc_idx_3d] = albedo_loc;
            src[sfc_idx_3d] = src_loc;

#pragma unroll loop_unroll_factor_nlay
            for (int ilay=nlay-1; ilay >= 0; --ilay)
            {
                const int lay_idx  = ivcol + ilay*nvcol + igpt*nvcol*nlay;
                const int lev_idx1 = ivcol + ilay*nvcol + igpt*nvcol*(nlay+1);

                auto denom_loc = Float(1.)/(Float(1.) - r_dif[lay_idx] * albedo_loc);
                auto albedo_next = r_dif[lay_idx] + t_dif[lay_idx].read() * t_dif[lay_idx]
                                                    * albedo_loc * denom_loc;
                auto src_next = source_up[lay_idx] + t_dif[lay_idx] * denom_loc *
                                                     (src_loc + albedo_loc * source_dn[lay_idx]);

                albedo_loc = albedo_next;
                src_loc = src_next;

                denom[lay_idx] = denom_loc;
                albedo[lev_idx1] = albedo_loc;
                src[lev_idx1] = src_loc;
            }

            const int top_idx = ivcol + igpt*(nlay+1)*nvcol;
            auto flux_dn_loc = kernel_float::into_vec(flux_dn[top_idx]);

            flux_dn[top_idx] = flux_dn_loc + flux_dir[top_idx];
            flux_up[top_idx] = flux_dn_loc * albedo_loc + src_loc;

#pragma unroll loop_unroll_factor_nlay
            for (int ilay=0; ilay < nlay; ++ilay)
            {
                const int lev_idx1 = ivcol + (ilay+1)*nvcol + igpt*(nlay+1)*nvcol;
                const int lay_idx = ivcol + ilay*nvcol + igpt*(nlay)*nvcol;

                flux_dn_loc = (t_dif[lay_idx] * flux_dn_loc +
                               r_dif[lay_idx].read() * src[lev_idx1] +
                               source_dn[lay_idx]) * denom[lay_idx];

                flux_dn[lev_idx1] = flux_dn_loc + flux_dir[lev_idx1];
                flux_up[lev_idx1] = kernel_float::fma(flux_dn_loc, albedo[lev_idx1], src[lev_idx1]);
            }
        }
        else
        {
            auto flux_dir_loc = kernel_float::into_vec(flux_dir[ivcol + nlay*nvcol + igpt*(nlay+1)*nvcol]);

#pragma unroll loop_unroll_factor_nlay
            for (int ilay=nlay-1; ilay>=0; --ilay)
            {
                vec r_dir, t_dir, t_noscat;
                sw_2stream_function<stream_type, vector_size, compute_type, tau_type, intermediate_type>(
                        ivcol, ilay, igpt,
                        nvcol, nlay, ngpt,
                        tau, ssa, g, mu0,
                        r_dif, t_dif,
                        kernel_float::wrap_ptr<compute_type, vector_size>(r_dir.data()),
                        kernel_float::wrap_ptr<compute_type, vector_size>(t_dir.data()),
                        kernel_float::wrap_ptr<compute_type, vector_size>(t_noscat.data()));

                const int idx_lay  = ivcol + ilay*nvcol + igpt*nlay*nvcol;
                const int idx_lev1 = ivcol + (ilay)*nvcol + igpt*(nlay+1)*nvcol;

                source_up[idx_lay] = r_dir * flux_dir_loc;
                source_dn[idx_lay] = t_dir * flux_dir_loc;

                flux_dir_loc = t_noscat * flux_dir_loc;
                flux_dir[idx_lev1] = flux_dir_loc;
            }

            const int sfc_idx = ivcol + igpt*nvcol;
            const int flx_idx = ivcol + igpt*(nlay+1)*nvcol;
            auto albedo_loc = sfc_alb_dif[sfc_idx];
            auto src_loc = flux_dir[flx_idx] * sfc_alb_dir[ivcol];
            source_sfc[sfc_idx] = src_loc;

            const int sfc_idx_3d = ivcol + igpt*(nlay+1)*nvcol;
            albedo[sfc_idx_3d] = albedo_loc;
            src[sfc_idx_3d] = src_loc;

#pragma unroll loop_unroll_factor_nlay
            for (int ilay=0; ilay<nlay; ++ilay)
            {
                const int lay_idx  = ivcol + ilay*nvcol + igpt*nvcol*nlay;
                const int lev_idx2 = ivcol + (ilay+1)*nvcol + igpt*nvcol*(nlay+1);

                auto denom_loc = Float(1.)/(Float(1.) - r_dif[lay_idx] * albedo_loc);
                auto albedo_next = r_dif[lay_idx] + (t_dif[lay_idx].read() * t_dif[lay_idx] *
                                                     albedo_loc * denom_loc);
                auto src_next = source_up[lay_idx] + t_dif[lay_idx].read() * denom_loc *
                                                     (src_loc+albedo_loc*source_dn[lay_idx]);

                albedo_loc = albedo_next;
                src_loc = src_next;

                denom[lay_idx] = denom_loc;
                albedo[lev_idx2] = albedo_loc;
                src[lev_idx2] = src_loc;
            }

            const int top_idx = ivcol + nlay*nvcol + igpt*(nlay+1)*nvcol;
            auto flux_dn_loc = kernel_float::into_vec(flux_dn[top_idx]);

            flux_dn[top_idx] = flux_dn_loc + flux_dir[top_idx];
            flux_up[top_idx] = flux_dn_loc * albedo_loc + src_loc;

#pragma unroll loop_unroll_factor_nlay
            for (int ilay=nlay-1; ilay >= 0; --ilay) {
                const int lay_idx = ivcol + ilay * nvcol + igpt * nlay * nvcol;
                const int lev_idx1 = ivcol + ilay * nvcol + igpt * (nlay + 1) * nvcol;

                flux_dn_loc = (t_dif[lay_idx] * flux_dn_loc +
                               r_dif[lay_idx].read() * src[lev_idx1] +
                               source_dn[lay_idx]) * denom[lay_idx];

                flux_dn[lev_idx1] = flux_dn_loc + flux_dir[lev_idx1];
                flux_up[lev_idx1] = flux_dn_loc * albedo[lev_idx1] + src[lev_idx1];
            }
        }
    }
}

