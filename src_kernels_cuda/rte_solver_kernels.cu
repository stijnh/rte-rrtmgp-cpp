#include <float.h>

#include "types.h"


const int loop_unroll_factor_nlay = 4;


template<typename TF> __device__ constexpr TF k_min();
template<> __device__ constexpr double k_min() { return 1.e-12; }
template<> __device__ constexpr float k_min() { return 1.e-4f; }


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

__device__
void lw_transport_noscat_kernel(
        const int icol, const int igpt, const int ncol, const int nlay, const int ngpt, const Bool top_at_1,
        const Float* __restrict__ tau, const INTERMEDIATE_TYPE* __restrict__ trans, const Float sfc_albedo,
        const INTERMEDIATE_TYPE* __restrict__ source_dn, const INTERMEDIATE_TYPE* __restrict__ source_up, Float source_sfc,
        FLUX_TYPE* __restrict__ radn_up, FLUX_TYPE* __restrict__ radn_dn, Float source_sfc_jac, Float* __restrict__ radn_up_jac,
        Float radn_dn_top, Float scaling)
{
    if (top_at_1)
    {
        const int idx_top = icol + igpt*ncol*(nlay+1);
        Float radn_dn_loc = radn_dn_top;
        radn_dn[idx_top] = static_cast<FLUX_TYPE>(radn_dn_loc * scaling);

        #pragma unroll loop_unroll_factor_nlay
        for (int ilev=0; ilev<(nlay); ++ilev)
        {
            const int idx1 = icol + (ilev+1)*ncol + igpt*ncol*(nlay+1);
            const int idx3 = icol + ilev*ncol + igpt*ncol*nlay;
            radn_dn_loc = static_cast<Float>(trans[idx3]) * radn_dn_loc + static_cast<Float>(source_dn[idx3]);
            radn_dn[idx1] = static_cast<FLUX_TYPE>(radn_dn_loc * scaling);
        }

        Float radn_up_loc = radn_dn_loc * sfc_albedo + source_sfc;
        Float radn_jac_loc = source_sfc_jac;

        const int idx_bot = icol + nlay*ncol + igpt*ncol*(nlay+1);
        radn_up[idx_bot] = static_cast<FLUX_TYPE>(radn_up_loc * scaling);
        radn_up_jac[idx_bot] = radn_jac_loc * scaling;

        #pragma unroll loop_unroll_factor_nlay
        for (int ilev=nlay-1; ilev>=0; --ilev)
        {
            const int idx3 = icol + ilev*ncol + igpt*ncol*nlay;
            radn_up_loc = static_cast<Float>(trans[idx3]) * radn_up_loc + static_cast<Float>(source_up[idx3]);
            radn_jac_loc = static_cast<Float>(trans[idx3]) * radn_jac_loc;

            const int idx1 = icol + ilev*ncol + igpt*ncol*(nlay+1);
            radn_up[idx1] = static_cast<FLUX_TYPE>(radn_up_loc * scaling);
            radn_up_jac[idx1] = radn_jac_loc * scaling;
        }
    }
    else
    {
        const int idx_top = icol + nlay*ncol + igpt*ncol*(nlay+1);
        Float radn_dn_loc = radn_dn_top;
        radn_dn[idx_top] = static_cast<FLUX_TYPE>(radn_dn_loc * scaling);

        #pragma unroll loop_unroll_factor_nlay
        for (int ilev=(nlay-1); ilev>=0; --ilev)
        {
            const int idx1 = icol + ilev*ncol + igpt*ncol*(nlay+1);
            const int idx3 = icol + ilev*ncol + igpt*ncol*nlay;
            radn_dn_loc = static_cast<Float>(trans[idx3]) * radn_dn_loc + static_cast<Float>(source_dn[idx3]);
            radn_dn[idx1] = static_cast<FLUX_TYPE>(radn_dn_loc * scaling);
        }

        Float radn_up_loc = radn_dn_loc * sfc_albedo + source_sfc;
        Float radn_jac_loc = source_sfc_jac;

        const int idx_bot = icol + igpt*ncol*(nlay+1);
        radn_up[idx_bot] = static_cast<FLUX_TYPE>(radn_up_loc * scaling);
        radn_up_jac[idx_bot] = radn_jac_loc * scaling;

        #pragma unroll loop_unroll_factor_nlay
        for (int ilev=0; ilev<nlay; ++ilev)
        {
            const int idx3 = icol + ilev*ncol + igpt*ncol*nlay;
            radn_up_loc = static_cast<Float>(trans[idx3]) * radn_up_loc + static_cast<Float>(source_up[idx3]);
            radn_jac_loc = static_cast<Float>(trans[idx3]) * radn_jac_loc;

            const int idx1 = icol + (ilev+1)*ncol + igpt*ncol*(nlay+1);
            radn_up[idx1] = static_cast<FLUX_TYPE>(radn_up_loc * scaling);
            radn_up_jac[idx1] = radn_jac_loc * scaling;
        }
    }
}

#pragma kernel problem_size(ncol, ngpt)
#pragma kernel block_size(32, 4)
#pragma kernel buffer(D[ncol*ngpt])
#pragma kernel buffer(weight[1])
#pragma kernel buffer(tau[ncol*nlay*ngpt])
#pragma kernel buffer(lay_source[ncol*nlay*ngpt])
#pragma kernel buffer(lev_source[ngpt * ncol * (nlay + 1)])
#pragma kernel buffer(sfc_emis[ngpt*ncol])
#pragma kernel buffer(sfc_src[ngpt*ncol])
#pragma kernel buffer(radn_up[ngpt*ncol*(nlay+1)])
#pragma kernel buffer(radn_dn[ngpt*ncol*(nlay+1)])
#pragma kernel buffer(sfc_src_jac[ngpt*ncol])
#pragma kernel buffer(radn_up_jac[ngpt*ncol*(nlay+1)])
#pragma kernel buffer(trans[ncol*nlay*ngpt])
#pragma kernel buffer(source_dn[ncol*nlay*ngpt])
#pragma kernel buffer(source_up[ncol*nlay*ngpt])
template <Bool top_at_1>
__global__
void lw_solver_noscat_kernel(
        const int ncol, const int nlay, const int ngpt, const Float tau_thres,
        const Float* __restrict__ D, const Float* __restrict__ weight, const Float* __restrict__ tau, const Float* __restrict__ lay_source,
        const Float* __restrict__ lev_source, const Float* __restrict__ sfc_emis,
        const Float* __restrict__ sfc_src, FLUX_TYPE* __restrict__ radn_up, FLUX_TYPE* __restrict__ radn_dn,
        const Float* __restrict__ sfc_src_jac, Float* __restrict__ radn_up_jac,
        INTERMEDIATE_TYPE* __restrict__ trans, INTERMEDIATE_TYPE* __restrict__ source_dn, INTERMEDIATE_TYPE* __restrict__ source_up)
{
    const int icol = blockIdx.x*blockDim.x + threadIdx.x;
    const int igpt = blockIdx.y*blockDim.y + threadIdx.y;

    if ( (icol < ncol) && (igpt < ngpt) )
    {
        for (int ilay=0; ilay<nlay; ++ilay) {
            const int idx_lay = icol + ilay * ncol + igpt * ncol * nlay;
            const int idx_lev = icol + ilay * ncol + igpt * ncol * (nlay + 1);
            const int idx_lev_p = icol + (ilay + 1) * ncol + igpt * ncol * (nlay + 1);

            const int idx_D = icol + igpt * ncol;

            Float tau_loc = tau[idx_lay] * D[idx_D];
            Float trans_loc = exp(-tau_loc);
            Float trans_loc_inv = -expm1(-tau_loc); // `1 - trans_loc`

            const Float fact =
                    tau_loc > tau_thres ?
                    trans_loc_inv / tau_loc - trans_loc :
                    tau_loc *
                    (Float(.5) + tau_loc * (Float(-1. / 3.) + tau_loc * Float(1. / 8.)));

            Float src_inc = trans_loc_inv * lev_source[idx_lev_p] +
                            Float(2.) * fact * (lay_source[idx_lay] - lev_source[idx_lev_p]);
            Float src_dec = trans_loc_inv * lev_source[idx_lev] +
                            Float(2.) * fact * (lay_source[idx_lay] - lev_source[idx_lev]);

            trans[idx_lay] = static_cast<INTERMEDIATE_TYPE>(trans_loc);
            source_dn[idx_lay] = static_cast<INTERMEDIATE_TYPE>(top_at_1 ? src_inc : src_dec);
            source_up[idx_lay] = static_cast<INTERMEDIATE_TYPE>(top_at_1 ? src_dec : src_inc);
        }

        const int idx2d = icol + igpt*ncol;
        Float sfc_albedo = Float(1.) - sfc_emis[idx2d];
        Float source_sfc = sfc_emis[idx2d] * sfc_src[idx2d];
        Float source_sfc_jac = sfc_emis[idx2d] * sfc_src_jac[idx2d];

        const Float pi = acos(Float(-1.));
        Float scaling = pi * weight[0];
        const int idx_top = icol + (top_at_1 ? 0 : nlay)*ncol + igpt*ncol*(nlay+1);
        const Float radn_dn_top = static_cast<Float>(radn_dn[idx_top]) / (Float(2.) * pi * weight[0]);

        lw_transport_noscat_kernel(
                icol, igpt, ncol, nlay, ngpt, top_at_1, tau, trans, sfc_albedo, source_dn,
                source_up, source_sfc, radn_up, radn_dn, source_sfc_jac, radn_up_jac, radn_dn_top, scaling);
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
void apply_BC_kernel_lw(const int isfc, int ncol, const int nlay, const int ngpt, const Bool top_at_1, const FLUX_TYPE* __restrict__ inc_flux, FLUX_TYPE* __restrict__ flux_dn)
{
    const int icol = blockIdx.x*blockDim.x + threadIdx.x;
    const int igpt = blockIdx.y*blockDim.y + threadIdx.y;

    if ( (icol < ncol) && (igpt < ngpt) )
    {
        const int idx_in  = icol + isfc*ncol + igpt*ncol*(nlay+1);
        const int idx_out = (top_at_1) ? icol + igpt*ncol*(nlay+1) : icol + nlay*ncol + igpt*ncol*(nlay+1);
        flux_dn[idx_out] = static_cast<FLUX_TYPE>(inc_flux[idx_in]);
    }
}

__global__
void apply_BC_kernel(const int ncol, const int nlay, const int ngpt, const Bool top_at_1, const FLUX_TYPE* __restrict__ inc_flux, FLUX_TYPE* __restrict__ flux_dn)
{
    const int icol = blockIdx.x*blockDim.x + threadIdx.x;
    const int igpt = blockIdx.y*blockDim.y + threadIdx.y;
    if ( (icol < ncol) && (igpt < ngpt) )
    {
        const int idx_out = icol + ((top_at_1 ? 0 : (nlay * ncol))) + (igpt * ncol * (nlay + 1));
        const int idx_in = icol + (igpt * ncol);
        flux_dn[idx_out] = static_cast<FLUX_TYPE>(inc_flux[idx_in]);
    }
}

__global__
void apply_BC_kernel(const int ncol, const int nlay, const int ngpt, const Bool top_at_1, const FLUX_TYPE* __restrict__ inc_flux, const Float* __restrict__ factor, FLUX_TYPE* __restrict__ flux_dn)
{
    const int icol = blockIdx.x*blockDim.x + threadIdx.x;
    const int igpt = blockIdx.y*blockDim.y + threadIdx.y;
    if ( (icol < ncol) && (igpt < ngpt) )
    {
        const int idx_out = icol + ((top_at_1 ? 0 : (nlay * ncol))) + (igpt * ncol * (nlay + 1));
        const int idx_in = icol + (igpt * ncol);

        flux_dn[idx_out] = static_cast<FLUX_TYPE>(static_cast<Float>(inc_flux[idx_in]) * factor[icol]);
    }
}

__global__
void apply_BC_kernel(const int ncol, const int nlay, const int ngpt, const Bool top_at_1, FLUX_TYPE* __restrict__ flux_dn)
{
    const int icol = blockIdx.x*blockDim.x + threadIdx.x;
    const int igpt = blockIdx.y*blockDim.y + threadIdx.y;
    if ( (icol < ncol) && (igpt < ngpt) )
    {
        const int idx_out = icol + ((top_at_1 ? 0 : (nlay * ncol))) + (igpt * ncol * (nlay + 1));
        flux_dn[idx_out] = static_cast<FLUX_TYPE>(0);
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


__device__
void sw_2stream_function(
        const int icol, const int ilay, const int igpt,
        const int ncol, const int nlay, const int ngpt,
        const Float* __restrict__ tau, const Float* __restrict__ ssa,
        const Float* __restrict__ g, const Float* __restrict__ mu0,
        Float* __restrict__ r_dif, Float* __restrict__ t_dif,
        Float* __restrict__ r_dir, Float* __restrict__ t_dir,
        Float* __restrict__ t_noscat_out)
{
        const int idx = icol + ilay*ncol + igpt*nlay*ncol;

        const Float mu0_inv = Float(1.)/mu0[icol];
        const Float gamma1 = (Float(8.) - ssa[idx] * (Float(5.) + Float(3.) * g[idx])) * Float(.25);
        const Float gamma2 = Float(3.) * (ssa[idx] * (Float(1.) - g[idx])) * Float(.25);
        const Float gamma3 = (Float(2.) - Float(3.) * mu0[icol] * g[idx])  * Float(.25);
        const Float gamma4 = Float(1.) - gamma3;

        const Float alpha1 = gamma1 * gamma4 + gamma2 * gamma3;
        const Float alpha2 = gamma1 * gamma3 + gamma2 * gamma4;

        const Float k = sqrt(max((gamma1 - gamma2) * (gamma1 + gamma2), k_min<Float>()));
        const Float exp_minusktau = exp(-tau[idx] * k);
        const Float exp_minus2ktau = exp_minusktau * exp_minusktau;
        const Float one_minus_exp_minus2ktau = -expm1(Float(-2) * tau[idx] * k);

        const Float rt_term = Float(1.) / (k      * (Float(1.) + exp_minus2ktau) +
                                     gamma1 * one_minus_exp_minus2ktau);
        r_dif[idx] = rt_term * gamma2 * one_minus_exp_minus2ktau;
        t_dif[idx] = rt_term * Float(2.) * k * exp_minusktau;

        Float t_noscat = exp(-tau[idx] * mu0_inv);
        Float one_minus_t_noscat = -expm1(-tau[idx] * mu0_inv);
        *t_noscat_out = t_noscat;

        const Float k_mu     = k * mu0[icol];
        const Float k_gamma3 = k * gamma3;
        const Float k_gamma4 = k * gamma4;

        const Float fact = (abs(Float(1.) - k_mu*k_mu) > tmin<Float>()) ? Float(1.) - k_mu*k_mu : tmin<Float>();
        const Float rt_term2 = ssa[idx] * rt_term / fact;

        *r_dir = rt_term2  * ((Float(1.) - k_mu) * (alpha2 + k_gamma3)   -
                                  (Float(1.) + k_mu) * (alpha2 - k_gamma3) * exp_minus2ktau -
                                   Float(2.) * (k_gamma3 - alpha2 * k_mu)  * exp_minusktau * t_noscat);

        *t_dir = -rt_term2 * ((Float(1.) + k_mu) * (alpha1 + k_gamma4) * t_noscat   -
                                  (Float(1.) - k_mu) * (alpha1 - k_gamma4) * exp_minus2ktau * t_noscat -
                                   Float(2.) * (k_gamma4 + alpha1 * k_mu)  * exp_minusktau);

        // fix thanks to peter ukkonen (see https://github.com/earth-system-radiation/rte-rrtmgp/pull/39#issuecomment-1026698541)
        *r_dir = max(tmin<Float>(), min(*r_dir, one_minus_t_noscat));
        *t_dir = max(tmin<Float>(), min(*t_dir, one_minus_t_noscat - *r_dir));
}

#pragma kernel problem_size(ncol, ngpt)
#pragma kernel block_size(32, 4)
#pragma kernel buffer(tau[ngpt*nlay*ncol])
#pragma kernel buffer(ssa[ngpt*nlay*ncol])
#pragma kernel buffer(g[ngpt*nlay*ncol])
#pragma kernel buffer(mu0[ncol])
#pragma kernel buffer(r_dif[ncol*nlay*ngpt])
#pragma kernel buffer(t_dif[ncol*nlay*ngpt])
#pragma kernel buffer(sfc_alb_dir[ngpt*ncol])
#pragma kernel buffer(sfc_alb_dif[ngpt*ncol])
#pragma kernel buffer(source_up[ngpt*nlay*ncol])
#pragma kernel buffer(source_dn[ngpt*nlay*ncol])
#pragma kernel buffer(source_sfc[ngpt*ncol])
#pragma kernel buffer(flux_up[ngpt*(nlay+1)*ncol])
#pragma kernel buffer(flux_dn[ngpt*(nlay+1)*ncol])
#pragma kernel buffer(flux_dir[ngpt*(nlay+1)*ncol])
#pragma kernel buffer(albedo[ncol*(nlay+1)*ngpt])
#pragma kernel buffer(src[ncol*(nlay+1)*ngpt])
#pragma kernel buffer(denom[ngpt*ncol*nlay])
template<Bool top_at_1> __global__
void sw_solver_kernel(
        const int ncol, const int nlay, const int ngpt,
        const Float* __restrict__ tau, const Float* __restrict__ ssa,
        const Float* __restrict__ g, const Float* __restrict__ mu0,
        Float* __restrict__ r_dif, Float* __restrict__ t_dif,
        const Float* __restrict__ sfc_alb_dir, const Float* __restrict__ sfc_alb_dif,
        Float* __restrict__ source_up, Float* __restrict__ source_dn, Float* __restrict__ source_sfc,
        Float* __restrict__ flux_up, Float* __restrict__ flux_dn, Float* __restrict__ flux_dir,
        Float* __restrict__ albedo, Float* __restrict__ src, Float* __restrict__ denom)
{
    const int icol = blockIdx.x*blockDim.x + threadIdx.x;
    const int igpt = blockIdx.y*blockDim.y + threadIdx.y;

    if ( (icol < ncol) && (igpt < ngpt) )
    {
        if (top_at_1)
        {
            Float flux_dir_loc = flux_dir[icol + igpt*(nlay+1)*ncol];

            for (int ilay=0; ilay<nlay; ++ilay)
            {
                Float r_dir, t_dir, t_noscat;
                sw_2stream_function(icol, ilay, igpt,
                                    ncol, nlay, ngpt,
                                    tau, ssa, g, mu0,
                                    r_dif, t_dif, &r_dir, &t_dir, &t_noscat);

                const int idx_lay  = icol + ilay*ncol + igpt*nlay*ncol;
                const int idx_lev2 = icol + (ilay+1)*ncol + igpt*(nlay+1)*ncol;

                source_up[idx_lay] = r_dir * flux_dir_loc;
                source_dn[idx_lay] = t_dir * flux_dir_loc;

                flux_dir_loc = t_noscat * flux_dir_loc;
                flux_dir[idx_lev2] = flux_dir_loc;
            }

            const int sfc_idx = icol + igpt*ncol;
            source_sfc[sfc_idx] = flux_dir_loc * sfc_alb_dir[icol];

            const int sfc_idx_2d = icol + igpt*ncol;
            Float albedo_loc = sfc_alb_dif[sfc_idx_2d];
            Float src_loc = source_sfc[sfc_idx_2d];

            const int sfc_idx_3d = icol + nlay*ncol + igpt*(nlay+1)*ncol;
            albedo[sfc_idx_3d] = albedo_loc;
            src[sfc_idx_3d] = src_loc;

#pragma unroll loop_unroll_factor_nlay
            for (int ilay=nlay-1; ilay >= 0; --ilay)
            {
                const int lay_idx  = icol + ilay*ncol + igpt*ncol*nlay;
                const int lev_idx1 = icol + ilay*ncol + igpt*ncol*(nlay+1);

                Float denom_loc = Float(1.)/(Float(1.) - r_dif[lay_idx] * albedo_loc);
                Float albedo_next = r_dif[lay_idx] + t_dif[lay_idx] * t_dif[lay_idx]
                                                    * albedo_loc * denom_loc;
                Float src_next = source_up[lay_idx] + t_dif[lay_idx] * denom_loc *
                                                     (src_loc + albedo_loc * source_dn[lay_idx]);

                albedo_loc = albedo_next;
                src_loc = src_next;

                denom[lay_idx] = denom_loc;
                albedo[lev_idx1] = albedo_loc;
                src[lev_idx1] = src_loc;
            }

            const int top_idx = icol + igpt*(nlay+1)*ncol;
            Float flux_dn_loc = flux_dn[top_idx];

            flux_dn[top_idx] = flux_dn_loc + flux_dir[top_idx];
            flux_up[top_idx] = flux_dn_loc * albedo_loc + src_loc;

            for (int ilay=0; ilay < nlay; ++ilay)
            {
                const int lev_idx1 = icol + (ilay+1)*ncol + igpt*(nlay+1)*ncol;
                const int lay_idx = icol + ilay*ncol + igpt*(nlay)*ncol;

                flux_dn_loc = (t_dif[lay_idx] * flux_dn_loc +
                               r_dif[lay_idx] * src[lev_idx1] +
                               source_dn[lay_idx]) * denom[lay_idx];

                flux_dn[lev_idx1] = flux_dn_loc + flux_dir[lev_idx1];
                flux_up[lev_idx1] = flux_dn_loc * albedo[lev_idx1] + src[lev_idx1];
            }
        }
        else
        {
            Float flux_dir_loc = flux_dir[icol + nlay*ncol + igpt*(nlay+1)*ncol];

            for (int ilay=nlay-1; ilay>=0; --ilay)
            {
                Float r_dir, t_dir, t_noscat;
                sw_2stream_function(icol, ilay, igpt,
                                    ncol, nlay, ngpt,
                                    tau, ssa, g, mu0,
                                    r_dif, t_dif, &r_dir, &t_dir, &t_noscat);

                const int idx_lay  = icol + ilay*ncol + igpt*nlay*ncol;
                const int idx_lev1 = icol + (ilay)*ncol + igpt*(nlay+1)*ncol;

                source_up[idx_lay] = r_dir * flux_dir_loc;
                source_dn[idx_lay] = t_dir * flux_dir_loc;

                flux_dir_loc = t_noscat * flux_dir_loc;
                flux_dir[idx_lev1] = flux_dir_loc;
            }

            const int sfc_idx = icol + igpt*ncol;
            const int flx_idx = icol + igpt*(nlay+1)*ncol;
            source_sfc[sfc_idx] = flux_dir[flx_idx] * sfc_alb_dir[icol];

            const int sfc_idx_2d = icol + igpt*ncol;
            Float albedo_loc = sfc_alb_dif[sfc_idx_2d];
            Float src_loc = source_sfc[sfc_idx_2d];

            const int sfc_idx_3d = icol + igpt*(nlay+1)*ncol;
            albedo[sfc_idx_3d] = albedo_loc;
            src[sfc_idx_3d] = src_loc;

#pragma unroll loop_unroll_factor_nlay
            for (int ilay=0; ilay<nlay; ++ilay)
            {
                const int lay_idx  = icol + ilay*ncol + igpt*ncol*nlay;
                const int lev_idx2 = icol + (ilay+1)*ncol + igpt*ncol*(nlay+1);

                Float denom_loc = Float(1.)/(Float(1.) - r_dif[lay_idx] * albedo_loc);
                Float albedo_next = r_dif[lay_idx] + (t_dif[lay_idx] * t_dif[lay_idx] *
                                                     albedo_loc * denom_loc);
                Float src_next = source_up[lay_idx] + t_dif[lay_idx]*denom_loc *
                                                     (src_loc+albedo_loc*source_dn[lay_idx]);

                albedo_loc = albedo_next;
                src_loc = src_next;

                denom[lay_idx] = denom_loc;
                albedo[lev_idx2] = albedo_loc;
                src[lev_idx2] = src_loc;
            }

            const int top_idx = icol + nlay*ncol + igpt*(nlay+1)*ncol;
            Float flux_dn_loc = flux_dn[top_idx];

            flux_dn[top_idx] = flux_dn_loc + flux_dir[top_idx];
            flux_up[top_idx] = flux_dn_loc * albedo_loc + src_loc;

            for (int ilay=nlay-1; ilay >= 0; --ilay) {
                const int lay_idx = icol + ilay * ncol + igpt * nlay * ncol;
                const int lev_idx1 = icol + ilay * ncol + igpt * (nlay + 1) * ncol;

                flux_dn_loc = (t_dif[lay_idx] * flux_dn_loc +
                               r_dif[lay_idx] * src[lev_idx1] +
                               source_dn[lay_idx]) * denom[lay_idx];

                flux_dn[lev_idx1] = flux_dn_loc + flux_dir[lev_idx1];
                flux_up[lev_idx1] = flux_dn_loc * albedo[lev_idx1] + src[lev_idx1];
            }
        }
    }
}

