#include "types.h"
#include "kernel_float.h"

__device__
Float interpolate1D(
        const Float val,
        const Float offset,
        const Float delta,
        const int len,
        const Float* __restrict__ table)
{
    Float val0 = (val - offset) * (Float(1) / delta);
    Float frac = val0 - int(val0);
    int idx = min(len-1, max(1, int(val0)+1));
    return table[idx-1] + frac * (table[idx] - table[idx-1]);
}


__device__ __forceinline__
void interpolate2D_byflav_kernel(const FloatFMinor* __restrict__ fminor,
                                 const Float* __restrict__ kin,
                                 const int gpt_start, const int gpt_end,
                                 Float* __restrict__ k,
                                 const int* __restrict__ jeta,
                                 const int jtemp,
                                 const int ngpt,
                                 const int neta)
{
    const int band_gpt = gpt_end-gpt_start;
    const int j0 = jeta[0];
    const int j1 = jeta[1];

    #pragma unroll
    for (int igpt=0; igpt<band_gpt; ++igpt)
    {
        k[igpt] = Float(fminor[0]) * kin[igpt + (j0-1)*ngpt + (jtemp-1)*neta*ngpt] +
                  Float(fminor[1]) * kin[igpt +  j0   *ngpt + (jtemp-1)*neta*ngpt] +
                  Float(fminor[2]) * kin[igpt + (j1-1)*ngpt + jtemp    *neta*ngpt] +
                  Float(fminor[3]) * kin[igpt +  j1   *ngpt + jtemp    *neta*ngpt];
    }
}


__device__
void interpolate3D_byflav_kernel(
        const Float* __restrict__ scaling,
        const FloatFMajor* __restrict__ fmajor,
        const Float* __restrict__ k,
        const int gpt_start, const int gpt_end,
        const int* __restrict__ jeta,
        const int jtemp,
        const int jpress,
        const int ngpt,
        const int neta,
        const int npress,
        FloatTau* __restrict__ tau_major)
{
    const int band_gpt = gpt_end-gpt_start;
    const int j0 = jeta[0];
    const int j1 = jeta[1];

    #pragma unroll
    for (int igpt=0; igpt<band_gpt; ++igpt)
    {
        auto result = scaling[0]*
                          (Float(fmajor[0]) * k[igpt + (j0-1)*ngpt + (jpress-1)*neta*ngpt + (jtemp-1)*neta*ngpt*npress] +
                           Float(fmajor[1]) * k[igpt +  j0   *ngpt + (jpress-1)*neta*ngpt + (jtemp-1)*neta*ngpt*npress] +
                           Float(fmajor[2]) * k[igpt + (j0-1)*ngpt + jpress*neta*ngpt     + (jtemp-1)*neta*ngpt*npress] +
                           Float(fmajor[3]) * k[igpt +  j0   *ngpt + jpress*neta*ngpt     + (jtemp-1)*neta*ngpt*npress])
                        + scaling[1]*
                          (Float(fmajor[4]) * k[igpt + (j1-1)*ngpt + (jpress-1)*neta*ngpt + jtemp*neta*ngpt*npress] +
                            Float(fmajor[5]) * k[igpt +  j1   *ngpt + (jpress-1)*neta*ngpt + jtemp*neta*ngpt*npress] +
                            Float(fmajor[6]) * k[igpt + (j1-1)*ngpt + jpress*neta*ngpt     + jtemp*neta*ngpt*npress] +
                            Float(fmajor[7]) * k[igpt +  j1   *ngpt + jpress*neta*ngpt     + jtemp*neta*ngpt*npress]);

        tau_major[igpt] = FloatTau(result);
    }
}


__global__
void reorder12x21_kernel(
        const int ni, const int nj,
        const Float* __restrict__ arr_in, Float* __restrict__ arr_out)
{
    const int ii = blockIdx.x*blockDim.x + threadIdx.x;
    const int ij = blockIdx.y*blockDim.y + threadIdx.y;

    if ( (ii < ni) && (ij < nj) )
    {
        const int idx_out = ii + ij*ni;
        const int idx_in = ij + ii*nj;

        arr_out[idx_out] = arr_in[idx_in];
    }
}


__global__
void reorder123x321_kernel(
        const int ni, const int nj, const int nk,
        const Float* __restrict__ arr_in, Float* __restrict__ arr_out)
{
    const int ii = blockIdx.x*blockDim.x + threadIdx.x;
    const int ij = blockIdx.y*blockDim.y + threadIdx.y;
    const int ik = blockIdx.z*blockDim.z + threadIdx.z;

    if ( (ii < ni) && (ij < nj) && (ik < nk))
    {
        const int idx_out = ii + ij*ni + ik*nj*ni;
        const int idx_in  = ik + ij*nk + ii*nj*nk;

        arr_out[idx_out] = arr_in[idx_in];
    }
}


template <typename T>
__global__
void zero_array_kernel(
        const int ni, const int nj, const int nk,
        T* __restrict__ arr)
{
    const int ii = blockIdx.x*blockDim.x + threadIdx.x;
    const int ij = blockIdx.y*blockDim.y + threadIdx.y;
    const int ik = blockIdx.z*blockDim.z + threadIdx.z;

    if ( (ii < ni) && (ij < nj) && (ik < nk) )
    {
        const int idx = ii + ij*ni + ik*nj*ni;
        arr[idx] = T(0);
    }
}


template<typename T>
struct Index_1d
{
    __device__ Index_1d(T* __restrict__ data, const int n1) :
        data(data) {}
    __device__ __forceinline__ T& operator()(const int i1) { return data[i1-1]; }
    __device__ __forceinline__ const T& operator()(const int i1) const { return data[i1-1]; }
    T* __restrict__ data;
};


template<typename T>
struct Index_2d
{
    __device__ Index_2d(T* __restrict__ data, const int n1, const int n2) :
        data(data), s2(n1) {}
    __device__ __forceinline__ T& operator()(const int i1, const int i2) { return data[(i1-1) + (i2-1)*s2]; }
    __device__ __forceinline__ const T& operator()(const int i1, const int i2) const { return data[(i1-1) + (i2-1)*s2]; }
    T* __restrict__ data;
    const int s2;
};


template<typename T>
struct Index_3d
{
    __device__ Index_3d(T* __restrict__ data, const int n1, const int n2, const int n3) :
        data(data), s2(n1), s3(n1*n2) {}
    __device__ __forceinline__ T& operator()(const int i1, const int i2, const int i3) { return data[(i1-1) + (i2-1)*s2 + (i3-1)*s3]; }
    __device__ __forceinline__ const T& operator()(const int i1, const int i2, const int i3) const { return data[(i1-1) + (i2-1)*s2 + (i3-1)*s3]; }
    T* __restrict__ data;
    const int s2;
    const int s3;
};


template<typename T>
struct Index_4d
{
    __device__ Index_4d(T* __restrict__ data, const int n1, const int n2, const int n3, const int n4) :
        data(data), s2(n1), s3(n1*n2), s4(n1*n2*n3) {}
    __device__ __forceinline__ T& operator()(const int i1, const int i2, const int i3, const int i4) { return data[(i1-1) + (i2-1)*s2 + (i3-1)*s3 + (i4-1)*s4]; }
    __device__ __forceinline__ const T& operator()(const int i1, const int i2, const int i3, const int i4) const { return data[(i1-1) + (i2-1)*s2 + (i3-1)*s3 + (i4-1)*s4]; }
    T* __restrict__ data;
    const int s2;
    const int s3;
    const int s4;
};


template<typename T>
struct Index_6d
{
    __device__ Index_6d(T* __restrict__ data, const int n1, const int n2, const int n3, const int n4, const int n5, const int n6) :
        data(data), s2(n1), s3(n1*n2), s4(n1*n2*n3), s5(n1*n2*n3*n4), s6(n1*n2*n3*n4*n5) {}
    __device__ __forceinline__ T& operator()(const int i1, const int i2, const int i3, const int i4, const int i5, const int i6) { return data[(i1-1) + (i2-1)*s2 + (i3-1)*s3 + (i4-1)*s4 + (i5-1)*s5 + (i6-1)*s6]; }
    __device__ __forceinline__ const T& operator()(const int i1, const int i2, const int i3, const int i4, const int i5, const int i6) const { return data[(i1-1) + (i2-1)*s2 + (i3-1)*s3 + (i4-1)*s4 + (i5-1)*s5 + (i6-1)*s6]; }
    T* __restrict__ data;
    const int s2;
    const int s3;
    const int s4;
    const int s5;
    const int s6;
};

template <typename T, size_t N>
struct alignas(sizeof(T) * N) vector {
    __device__ __forceinline__ T& operator[](const int i) { return data[i]; }
    __device__ __forceinline__ const T& operator[](const int i) const { return data[i]; }

    T data[N];
};

__global__
void Planck_source_kernel(
        const int ncol,
        const int nlay,
        const int nbnd,
        const int ngpt,
        const int nflav,
        const int neta,
        const int npres,
        const int ntemp,
        const int nPlanckTemp,
        const FloatTemperature* __restrict__ tlay_ptr,
        const FloatTemperature* __restrict__ tlev_ptr,
        const FloatTemperature* __restrict__ tsfc_ptr,
        const int sfc_lay,
        const FloatFMajor* __restrict__ fmajor_ptr,
        const int* __restrict__ jeta_ptr,
        const Bool* __restrict__ tropo_ptr,
        const int* __restrict__ jtemp_ptr,
        const int* __restrict__ jpress_ptr,
        const int* __restrict__ gpoint_bands_ptr,
        const int* __restrict__ band_lims_gpt_ptr,
        const Float* __restrict__ pfracin_ptr,
        const Float temp_ref_min,
        const Float totplnk_delta,
        const Float* __restrict__ totplnk_ptr,
        const int* __restrict__ gpoint_flavor_ptr,
        const Float delta_Tsurf,
        FloatSurface* __restrict__ sfc_src_ptr,
        FloatSource* __restrict__ lay_src_ptr,
        FloatSource* __restrict__ lev_src_ptr,
        Float* __restrict__ sfc_src_jac_ptr)
{
    // THIS KERNEL USES FORTRAN INDEXING TO AVOID MISTAKES.
    const int icol = blockIdx.x*blockDim.x + threadIdx.x + 1;
    const int ilay = blockIdx.y*blockDim.y + threadIdx.y + 1;
    const int igpt = blockIdx.z*blockDim.z + threadIdx.z + 1;

    // Input arrays, use Index functor to simplify index porting from Fortran to CUDA
    const Index_2d<const FloatTemperature> tlay        (tlay_ptr, ncol, nlay);
    const Index_2d<const FloatTemperature> tlev        (tlev_ptr, ncol, nlay+1);
    const Index_1d<const FloatTemperature> tsfc        (tsfc_ptr, ncol);
    const Index_3d<const vector<FloatFMajor, 8>> fmajor (reinterpret_cast<const vector<FloatFMajor, 8>*>(fmajor_ptr), ncol, nlay, nflav);
    const Index_4d<const int> jeta          (jeta_ptr, 2, ncol, nlay, nflav);
    const Index_2d<const Bool> tropo        (tropo_ptr, ncol, nlay);
    const Index_2d<const int> jtemp         (jtemp_ptr, ncol, nlay);
    const Index_2d<const int> jpress        (jpress_ptr, ncol, nlay);
    const Index_1d<const int> gpoint_bands  (gpoint_bands_ptr, ngpt);
    const Index_2d<const int> band_lims_gpt (band_lims_gpt_ptr, 2, nbnd);
    const Index_4d<const Float> pfracin     (pfracin_ptr, ntemp, neta, npres+1, ngpt); 
    const Index_2d<const Float> totplnk     (totplnk_ptr, nPlanckTemp, nbnd);
    const Index_2d<const int> gpoint_flavor (gpoint_flavor_ptr, 2, ngpt);

    // Output arrays
    Index_2d<FloatSurface> sfc_src(sfc_src_ptr, ncol, ngpt);
    Index_3d<FloatSource> lay_src(lay_src_ptr, ncol, nlay, ngpt);
    Index_3d<FloatSource> lev_src(lev_src_ptr, ncol, nlay+1, ngpt);
    Index_2d<Float> sfc_src_jac(sfc_src_jac_ptr, ncol, ngpt);

    if ( (icol <= ncol) && (ilay <= nlay) && (igpt <= ngpt) )
    {
        const int ibnd = gpoint_bands(igpt);
        const int itropo = (tropo(icol, ilay) == Bool(true)) ? 1 : 2;
        const int iflav = gpoint_flavor(itropo, igpt);

        // 3D interp.
        const Float pfrac =
              ( Float(fmajor(icol, ilay, iflav)[0]) * Float(pfracin(jtemp(icol, ilay), jeta(1, icol, ilay, iflav)  , jpress(icol, ilay)-1 + itropo, igpt))
              + Float(fmajor(icol, ilay, iflav)[1]) * Float(pfracin(jtemp(icol, ilay), jeta(1, icol, ilay, iflav)+1, jpress(icol, ilay)-1 + itropo, igpt))
              + Float(fmajor(icol, ilay, iflav)[2]) * Float(pfracin(jtemp(icol, ilay), jeta(1, icol, ilay, iflav)  , jpress(icol, ilay)   + itropo, igpt))
              + Float(fmajor(icol, ilay, iflav)[3]) * Float(pfracin(jtemp(icol, ilay), jeta(1, icol, ilay, iflav)+1, jpress(icol, ilay)   + itropo, igpt)) )

            + ( Float(fmajor(icol, ilay, iflav)[4]) * Float(pfracin(jtemp(icol, ilay)+1, jeta(2, icol, ilay, iflav)  , jpress(icol, ilay)-1 + itropo, igpt))
              + Float(fmajor(icol, ilay, iflav)[5]) * Float(pfracin(jtemp(icol, ilay)+1, jeta(2, icol, ilay, iflav)+1, jpress(icol, ilay)-1 + itropo, igpt))
              + Float(fmajor(icol, ilay, iflav)[6]) * Float(pfracin(jtemp(icol, ilay)+1, jeta(2, icol, ilay, iflav)  , jpress(icol, ilay)   + itropo, igpt))
              + Float(fmajor(icol, ilay, iflav)[7]) * Float(pfracin(jtemp(icol, ilay)+1, jeta(2, icol, ilay, iflav)+1, jpress(icol, ilay)   + itropo, igpt)) );

        Float planck_function_1 = interpolate1D(tlay(icol, ilay), temp_ref_min, totplnk_delta, nPlanckTemp, &totplnk(1, ibnd));
        lay_src(icol, ilay, igpt) = pfrac * planck_function_1;

        planck_function_1 = interpolate1D(tlev(icol, ilay), temp_ref_min, totplnk_delta, nPlanckTemp, &totplnk(1, ibnd));
        if (ilay == 1)
        {
            lev_src(icol, ilay, igpt) = pfrac * planck_function_1;
        }
        else
        {
            const int itropo = (tropo(icol, ilay-1) == Bool(true)) ? 1 : 2;
            const int iflav = gpoint_flavor(itropo, igpt);

            const Float pfrac_m1 =
                  ( Float(fmajor(icol, ilay-1, iflav)[0]) * Float(pfracin(jtemp(icol, ilay-1), jeta(1, icol, ilay-1, iflav)  , jpress(icol, ilay-1)-1 + itropo, igpt))
                  + Float(fmajor(icol, ilay-1, iflav)[1]) * Float(pfracin(jtemp(icol, ilay-1), jeta(1, icol, ilay-1, iflav)+1, jpress(icol, ilay-1)-1 + itropo, igpt))
                  + Float(fmajor(icol, ilay-1, iflav)[2]) * Float(pfracin(jtemp(icol, ilay-1), jeta(1, icol, ilay-1, iflav)  , jpress(icol, ilay-1)   + itropo, igpt))
                  + Float(fmajor(icol, ilay-1, iflav)[3]) * Float(pfracin(jtemp(icol, ilay-1), jeta(1, icol, ilay-1, iflav)+1, jpress(icol, ilay-1)   + itropo, igpt)) )

                + ( Float(fmajor(icol, ilay-1, iflav)[4]) * Float(pfracin(jtemp(icol, ilay-1)+1, jeta(2, icol, ilay-1, iflav)  , jpress(icol, ilay-1)-1 + itropo, igpt))
                  + Float(fmajor(icol, ilay-1, iflav)[5]) * Float(pfracin(jtemp(icol, ilay-1)+1, jeta(2, icol, ilay-1, iflav)+1, jpress(icol, ilay-1)-1 + itropo, igpt))
                  + Float(fmajor(icol, ilay-1, iflav)[6]) * Float(pfracin(jtemp(icol, ilay-1)+1, jeta(2, icol, ilay-1, iflav)  , jpress(icol, ilay-1)   + itropo, igpt))
                  + Float(fmajor(icol, ilay-1, iflav)[7]) * Float(pfracin(jtemp(icol, ilay-1)+1, jeta(2, icol, ilay-1, iflav)+1, jpress(icol, ilay-1)   + itropo, igpt)) );

            lev_src(icol, ilay, igpt) = sqrt(pfrac * pfrac_m1) * planck_function_1;
        }

        if (ilay == nlay)
        {
            planck_function_1 = interpolate1D(tlev(icol, ilay+1), temp_ref_min, totplnk_delta, nPlanckTemp, &totplnk(1, ibnd));
            lev_src(icol, ilay+1, igpt) = pfrac * planck_function_1;
        }

        if (ilay == sfc_lay)
        {
                        planck_function_1 = interpolate1D(Float(tsfc(icol))              , temp_ref_min, totplnk_delta, nPlanckTemp, &totplnk(1, ibnd));
            const Float planck_function_2 = interpolate1D(Float(tsfc(icol)) + delta_Tsurf, temp_ref_min, totplnk_delta, nPlanckTemp, &totplnk(1, ibnd));
            sfc_src    (icol, igpt) = pfrac * planck_function_1;
            sfc_src_jac(icol, igpt) = pfrac * (planck_function_2 - planck_function_1);
        }
    }
}


__global__
void interpolation_kernel(
        const int ncol, const int nlay, const int ngas, const int nflav,
        const int neta, const int npres, const int ntemp, const FloatTemperature tmin,
        const int* __restrict__ flavor,
        const FloatPressure* __restrict__ press_ref_log,
        const FloatTemperature* __restrict__ temp_ref,
        FloatPressure press_ref_log_delta,
        FloatTemperature temp_ref_min,
        FloatTemperature temp_ref_delta,
        FloatPressure press_ref_trop_log,
        const Float* __restrict__ vmr_ref,
        const FloatPressure* __restrict__ play,
        const FloatTemperature* __restrict__ tlay,
        FloatColGas* __restrict__ col_gas,
        int* __restrict__ jtemp,
        FloatFMajor* __restrict__ fmajor, FloatFMinor* __restrict__ fminor,
        FloatColMix* __restrict__ col_mix,
        Bool* __restrict__ tropo,
        int* __restrict__ jeta,
        int* __restrict__ jpress)
{
    const int icol  = blockIdx.x*blockDim.x + threadIdx.x;
    const int ilay  = blockIdx.y*blockDim.y + threadIdx.y;
    const int iflav = blockIdx.z*blockDim.z + threadIdx.z;

    if ( (icol < ncol) && (ilay < nlay) && (iflav < nflav) )
    {
        const int idx = icol + ilay*ncol;

        jtemp[idx] = int(Float(Float(tlay[idx]) - Float(temp_ref_min-temp_ref_delta)) / Float(temp_ref_delta));
        jtemp[idx] = min(ntemp-1, max(1, jtemp[idx]));
        const Float ftemp = Float(tlay[idx] - temp_ref[jtemp[idx]-1]) / Float(temp_ref_delta);

        const Float locpress = Float(1.) + (log(Float(play[idx])) - Float(press_ref_log[0])) / Float(press_ref_log_delta);
        jpress[idx] = min(npres-1, max(1, int(locpress)));
        const Float fpress = locpress - Float(jpress[idx]);

        tropo[idx] = log(Float(play[idx])) > Float(press_ref_trop_log);
        const int itropo = !tropo[idx];

        const int gas1 = flavor[2*iflav  ];
        const int gas2 = flavor[2*iflav+1];

        for (int itemp=0; itemp<2; ++itemp)
        {
            const int vmr_base_idx = itropo + (jtemp[idx]+itemp-1) * (ngas+1) * 2;
            const int colmix_idx = itemp + 2*(icol + ilay*ncol + iflav*ncol*nlay);
            const int colgas1_idx = icol + ilay*ncol + gas1*nlay*ncol;
            const int colgas2_idx = icol + ilay*ncol + gas2*nlay*ncol;
            const Float ratio_eta_half = vmr_ref[vmr_base_idx + 2*gas1] /
                                      vmr_ref[vmr_base_idx + 2*gas2];
            Float col_mix_res = Float(col_gas[colgas1_idx]) + ratio_eta_half * Float(col_gas[colgas2_idx]);
            col_mix[colmix_idx] = FloatColMix(col_mix_res);

            Float eta;
            if (col_mix_res > Float(2.)*Float(tmin))
                eta = Float(col_gas[colgas1_idx]) / col_mix_res;
            else
                eta = Float(0.5);

            const Float loceta = eta * Float(neta-1);
            jeta[colmix_idx] = min(int(loceta)+1, neta-1);
            const Float feta = fmod(loceta, Float(1.));
            const Float ftemp_term = Float(1-itemp) + Float(2*itemp-1)*ftemp;

            // Compute interpolation fractions needed for minor species.
            const int fminor_idx = 2*(itemp + 2*(icol + ilay*ncol + iflav*ncol*nlay));
            Float fminor_p = (Float(1.)-feta) * ftemp_term;
            Float fminor_n = feta * ftemp_term;

            fminor[fminor_idx  ] = FloatFMinor(fminor_p);
            fminor[fminor_idx+1] = FloatFMinor(fminor_n);

            // Compute interpolation fractions needed for major species.
            const int fmajor_idx = 2*2*(itemp + 2*(icol + ilay*ncol + iflav*ncol*nlay));
            fmajor[fmajor_idx  ] = FloatFMajor((Float(1.)-fpress) * fminor_p);
            fmajor[fmajor_idx+1] = FloatFMajor((Float(1.)-fpress) * fminor_n);
            fmajor[fmajor_idx+2] = FloatFMajor(fpress * fminor_p);
            fmajor[fmajor_idx+3] = FloatFMajor(fpress * fminor_n);
        }
    }
}

#pragma kernel problem_size(ncol, nlay, ngpt)
#pragma kernel block_size(block_size_x, block_size_y, block_size_z)
#pragma kernel buffer(gpoint_flavor[2*ngpt])
#pragma kernel buffer(band_lims_gpt[1])
#pragma kernel buffer(kmajor_ptr[ntemp * (npres+1) * neta * ngpt])
#pragma kernel buffer(col_mix_ptr[2*(nflav*ncol*nlay)])
#pragma kernel buffer(fmajor_ptr[2 * 2 * 2 * (nflav*ncol*nlay)])
#pragma kernel buffer(jeta[2*(nflav*ncol*nlay)])
#pragma kernel buffer(tropo[nlay*ncol])
#pragma kernel buffer(jtemp[nlay*ncol])
#pragma kernel buffer(jpress[nlay*ncol])
#pragma kernel buffer(tau_ptr[ngpt*nlay*ncol])
template<
        int block_size_x,
        int block_size_y,
        int block_size_z,
        int vector_size,
        typename compute_type,
        typename FloatKMajor,
        typename FloatColMix,
        typename FloatFMajor,
        typename FloatTau
>
__global__
void gas_optical_depths_major_kernel(
        const int ncol, const int nlay, const int nband, const int ngpt,
        const int nflav, const int neta, const int npres, const int ntemp,
        const int* __restrict__ gpoint_flavor,
        const int* __restrict__ band_lims_gpt,
        const FloatKMajor* __restrict__ kmajor_ptr,
        const FloatColMix* __restrict__ col_mix_ptr,
        const FloatFMajor* __restrict__ fmajor_ptr,
        const int* __restrict__ jeta,
        const Bool* __restrict__ tropo,
        const int* __restrict__ jtemp,
        const int* __restrict__ jpress,
        FloatTau* __restrict__ tau_ptr)
__launch_bounds__(block_size_x * block_size_y * block_size_z)
{
    const int ivcol = blockIdx.x * blockDim.x + threadIdx.x;
    const int ilay = blockIdx.y * blockDim.y + threadIdx.y;
    const int igpt = blockIdx.z * blockDim.z + threadIdx.z;
    const int nvcol = ncol / vector_size;

    using Float = compute_type;
    auto fmajor = kernel_float::make_vec_ptr<compute_type, 4>(fmajor_ptr);
    auto kmajor = kernel_float::make_vec_ptr<compute_type>(kmajor_ptr);
    auto col_mix = kernel_float::make_vec_ptr<compute_type, 2>(col_mix_ptr);
    auto tau = kernel_float::make_vec_ptr<compute_type, vector_size>(tau_ptr);

    if ( (ivcol * vector_size < ncol) && (ilay < nlay) && (igpt < ngpt) )
    {
        auto result = kernel_float::zeros<Float, vector_size>();

        const int idx_collay = ivcol * vector_size + ilay*ncol;
        const int npress = npres+1;

        const auto itropo = kernel_float::cast<int>(!kernel_float::read_aligned<vector_size>(&tropo[idx_collay]));
        const auto ljtemp = kernel_float::read_aligned<vector_size>(&jtemp[idx_collay]);
        const auto jpressi = kernel_float::read_aligned<vector_size>(&jpress[idx_collay]) + itropo;

        // Major gases.
#pragma unroll
        for (int i = 0; i < vector_size; i++) {
#pragma unroll
            for (int j=0; j<2; ++j)
            {
                const int icol = ivcol * vector_size + i;
                const int iflav = kernel_float::read_aligned<2>(&gpoint_flavor[2*igpt])[itropo[i]] - 1;

                const int idx_fcl = (icol + ilay*ncol + iflav*ncol*nlay);
                auto v0 = fmajor[2 * idx_fcl + j];

                auto jetai = jeta[2 * idx_fcl + j];
                auto v1 = kernel_float::concat(
                        kmajor[(ljtemp[i]-1+j) + (jetai-1)*ntemp + (jpressi[i]-1)*ntemp*neta + igpt*ntemp*neta*npress],
                        kmajor[(ljtemp[i]-1+j) +  jetai   *ntemp + (jpressi[i]-1)*ntemp*neta + igpt*ntemp*neta*npress],
                        kmajor[(ljtemp[i]-1+j) + (jetai-1)*ntemp + jpressi[i]    *ntemp*neta + igpt*ntemp*neta*npress],
                        kmajor[(ljtemp[i]-1+j) +  jetai   *ntemp + jpressi[i]    *ntemp*neta + igpt*ntemp*neta*npress]
                );

                result[i] += col_mix[idx_fcl][j] * kernel_float::dot(v0, v1);
            }
        }

        const int idx_out = ivcol + ilay*nvcol + igpt*nvcol*nlay;
        tau[idx_out] += result;
    }
}




#pragma kernel problem_size(ncol, nlay)
#pragma kernel block_size(block_size_x, block_size_y, block_size_z)
#pragma kernel buffer(gpoint_flavor[2*ngpt])
#pragma kernel buffer(kminor_ptr[nminork*ntemp*neta])
#pragma kernel buffer(minor_limits_gpt[2*nminor])
#pragma kernel buffer(minor_scales_with_density[nminor])
#pragma kernel buffer(scale_by_complement[nminor])
#pragma kernel buffer(idx_minor[nminor])
#pragma kernel buffer(idx_minor_scaling[nminor])
#pragma kernel buffer(kminor_start[nminor])
#pragma kernel buffer(play_ptr[nlay*ncol])
#pragma kernel buffer(tlay_ptr[nlay*ncol])
#pragma kernel buffer(col_gas_ptr[ncol*nlay*ngas])
#pragma kernel buffer(fminor_ptr[2 * 2 * (nflav*ncol*nlay)])
#pragma kernel buffer(jeta[2*nflav*ncol*nlay])
#pragma kernel buffer(jtemp[nlay*ncol])
#pragma kernel buffer(tropo_ptr[nlay*ncol])
#pragma kernel buffer(tau_ptr[ngpt*ncol*nlay])
template<
        int block_size_x,
        int block_size_y,
        int block_size_z,
        int vector_size=1,
        bool use_smem,
        typename Float,
        typename FloatKMinor,
        typename FloatPressure,
        typename FloatTemperature,
        typename FloatColGas,
        typename FloatFMinor,
        typename FloatTau,
        typename Policy
> __global__
void gas_optical_depths_minor_kernel(
        const int ncol, const int nlay, const int ngpt,
        const int ngas, const int nflav, const int ntemp, const int neta,
        const int nminor,
        const int nminork,
        const int idx_h2o, const int idx_tropo,
        const int* __restrict__ gpoint_flavor,
        const FloatKMinor* __restrict__ kminor_ptr,
        const int* __restrict__ minor_limits_gpt,
        const Bool* __restrict__ minor_scales_with_density,
        const Bool* __restrict__ scale_by_complement,
        const int* __restrict__ idx_minor,
        const int* __restrict__ idx_minor_scaling,
        const int* __restrict__ kminor_start,
        const FloatPressure* __restrict__ play_ptr,
        const FloatTemperature * __restrict__ tlay_ptr,
        const FloatColGas* __restrict__ col_gas_ptr,
        const FloatFMinor* __restrict__ fminor_ptr,
        const int2* __restrict__ jeta,
        const int* __restrict__ jtemp,
        const Bool* __restrict__ tropo_ptr,
        FloatTau* __restrict__ tau_ptr)
__launch_bounds__(block_size_x * block_size_y * block_size_z)
{
    const int ivcol = blockIdx.x * block_size_x + threadIdx.x;
    const int ilay = blockIdx.y * block_size_y + threadIdx.y;
    const int nvcol = ncol / vector_size;

    __shared__ kernel_float::vector_storage<Float, vector_size> scalings[block_size_y][block_size_x];

    auto play = kernel_float::make_vec_ptr<Float, vector_size>(play_ptr);
    auto tlay = kernel_float::make_vec_ptr<Float, vector_size>(tlay_ptr);
    auto col_gas = kernel_float::make_vec_ptr<Float, vector_size>(col_gas_ptr);
    auto tau = kernel_float::make_vec_ptr<Float, vector_size>(tau_ptr);
    auto tropo = kernel_float::make_vec_ptr<Bool, vector_size>(tropo_ptr);

    auto kminor = kernel_float::make_vec_ptr<Float>(kminor_ptr);
    auto fminor = kernel_float::make_vec_ptr<Float, 4>(fminor_ptr);

    if ( (ivcol < nvcol) && (ilay < nlay) )
    {
        const int idx_vcollay = ivcol + ilay*nvcol;
        const auto tropoi = tropo[idx_vcollay] == idx_tropo;

        if (!kernel_float::any(tropoi)) {
            return;
        }

        for (int imnr=0; imnr<nminor; ++imnr)
        {
            kernel_float::vec<Float, vector_size> scaling = 0;

            if (!use_smem || threadIdx.z == 0)
            {
                const int nvcl = nvcol * nlay;
                scaling = col_gas[idx_vcollay + idx_minor[imnr] * nvcl];

                if (minor_scales_with_density[imnr])
                {
                    const Float PaTohPa = 0.01;
                    scaling *= PaTohPa * kernel_float::divide<Policy>(play[idx_vcollay], tlay[idx_vcollay]);

                    if (idx_minor_scaling[imnr] > 0)
                    {
                        const int idx_vcollaywv = ivcol + ilay*nvcol + idx_h2o*nvcl;
                        auto vmr_fact = kernel_float::rcp<Policy>(col_gas[idx_vcollay]);
                        auto dry_fact = kernel_float::rcp<Policy>(kernel_float::fma(col_gas[idx_vcollaywv], vmr_fact, 1));

                        auto weight = col_gas[idx_vcollay + idx_minor_scaling[imnr] * nvcl] * vmr_fact * dry_fact;

                        if (scale_by_complement[imnr])
                            scaling *= 1 - weight;
                        else
                            scaling *= weight;
                    }
                }

                // Set entries to zero
                if constexpr (vector_size > 1) {
                    scaling *= tropoi;
                }
            }

            if constexpr (use_smem) {
                if (threadIdx.z == 0) {
                    scalings[threadIdx.y][threadIdx.x] = scaling;
                }

                __syncthreads();

                scaling = scalings[threadIdx.y][threadIdx.x];

                __syncthreads();
            }

            const int gpt_start = kernel_float::read_aligned<2>(&minor_limits_gpt[2*imnr])[0]-1;
            const int gpt_end = kernel_float::read_aligned<2>(&minor_limits_gpt[2*imnr])[1];
            const int gpt_offs = 1-idx_tropo;
            const int band_gpt = gpt_end-gpt_start;
            const int gpt_offset = kminor_start[imnr]-1;
            const int iflav = kernel_float::read_aligned<2>(&gpoint_flavor[2*gpt_start])[gpt_offs]-1;

            for (int igpt=threadIdx.z; igpt<band_gpt; igpt+=block_size_z)
            {
                kernel_float::vec<Float, vector_size> ltau_minor;

#pragma unroll
                for (int k = 0; k < vector_size; k++) {
                    const int icol = ivcol * vector_size + k;
                    const int idx_fcl = (icol + ilay*ncol) + iflav*ncol*nlay;
                    const int j0 = jeta[idx_fcl].x;
                    const int j1 = jeta[idx_fcl].y;
                    const int kjtemp = jtemp[icol + ilay*ncol];

                    const auto v0 = fminor[idx_fcl];
                    const auto v1 = kernel_float::concat(
                            kminor[(kjtemp-1) + (j0-1)*ntemp + (igpt+gpt_offset)*ntemp*neta],
                            kminor[(kjtemp-1) +  j0   *ntemp + (igpt+gpt_offset)*ntemp*neta],
                            kminor[kjtemp     + (j1-1)*ntemp + (igpt+gpt_offset)*ntemp*neta],
                            kminor[kjtemp     +  j1   *ntemp + (igpt+gpt_offset)*ntemp*neta]);

                    ltau_minor[k] = kernel_float::dot(v0, v1);
                }

                const int idx_out = ivcol + ilay*nvcol + (igpt+gpt_start)*nvcol*nlay;
                tau[idx_out] = kernel_float::fma(ltau_minor, scaling, tau[idx_out]);
            }
        }
    }
}

/*
#pragma kernel buffer(gpoint_flavor[2*ngpt])
#pragma kernel buffer(kminor_ptr[nminork*ntemp*neta])
#pragma kernel buffer(minor_limits_gpt[2*nminor])
#pragma kernel buffer(minor_scales_with_density[nminor])
#pragma kernel buffer(scale_by_complement[nminor])
#pragma kernel buffer(idx_minor[nminor])
#pragma kernel buffer(idx_minor_scaling[nminor])
#pragma kernel buffer(kminor_start[nminor])
#pragma kernel buffer(play_ptr[nlay*ncol])
#pragma kernel buffer(tlay_ptr[nlay*ncol])
#pragma kernel buffer(col_gas_ptr[ncol*nlay*ngas])
#pragma kernel buffer(fminor_ptr[2 * 2 * (nflav*ncol*nlay)])
#pragma kernel buffer(jeta[2*nflav*ncol*nlay])
#pragma kernel buffer(jtemp[nlay*ncol])
#pragma kernel buffer(tropo_ptr[nlay*ncol])
#pragma kernel buffer(tau_ptr[ngpt*ncol*nlay])
 */

/*
#pragma kernel problem_size(ncol, nlay)
#pragma kernel block_size(block_size_x, block_size_y, block_size_z)
template<
        int block_size_x,
        int block_size_y,
        int block_size_z,
        int vector_size=1,
        bool use_smem,
        typename Float,
        typename FloatKMinor,
        typename FloatPressure,
        typename FloatTemperature,
        typename FloatColGas,
        typename FloatFMinor,
        typename FloatTau,
        typename Policy
> __global__
void gas_optical_depths_minor_kernel(
        const int ncol, const int nlay, const int ngpt,
        const int ngas, const int nflav, const int ntemp, const int neta,
        const int nminor,
        const int nminork,
        const int idx_h2o, const int idx_tropo,
        const int* __restrict__ gpoint_flavor,
        const FloatKMinor* __restrict__ kminor,
        const int* __restrict__ minor_limits_gpt,
        const Bool* __restrict__ minor_scales_with_density,
        const Bool* __restrict__ scale_by_complement,
        const int* __restrict__ idx_minor,
        const int* __restrict__ idx_minor_scaling,
        const int* __restrict__ kminor_start,
        const FloatPressure* __restrict__ play,
        const FloatTemperature * __restrict__ tlay,
        const FloatColGas* __restrict__ col_gas,
        const FloatFMinor* __restrict__ fminor,
        const int2* __restrict__ jeta,
        const int* __restrict__ jtemp,
        const Bool* __restrict__ tropo,
        FloatTau* __restrict__ tau)
{
    const int icol = blockIdx.x * block_size_x + threadIdx.x;
    const int ilay = blockIdx.y * block_size_y + threadIdx.y;

    __shared__ Float scalings[block_size_y][block_size_x];

    if ( (icol < ncol) && (ilay < nlay) )
    {
        const int idx_collay = icol + ilay*ncol;

        if (tropo[idx_collay] == idx_tropo)
        {
            for (int imnr=0; imnr<nminor; ++imnr)
            {
                Float scaling = Float(0.);

                if (block_size_z == 1 || threadIdx.z == 0)
                {
                    const int ncl = ncol * nlay;
                    scaling = col_gas[idx_collay + idx_minor[imnr] * ncl];

                    if (minor_scales_with_density[imnr])
                    {
                        const Float PaTohPa = 0.01;
                        scaling *= PaTohPa * Float(play[idx_collay]) / Float(tlay[idx_collay]);

                        if (idx_minor_scaling[imnr] > 0)
                        {
                            const int idx_collaywv = icol + ilay*ncol + idx_h2o*ncl;
                            Float vmr_fact = Float(1.) / col_gas[idx_collay];
                            Float dry_fact = Float(1.) / (Float(1.) + col_gas[idx_collaywv] * vmr_fact);

                            if (scale_by_complement[imnr])
                                scaling *= (Float(1.) - col_gas[idx_collay + idx_minor_scaling[imnr] * ncl] * vmr_fact * dry_fact);
                            else
                                scaling *= col_gas[idx_collay + idx_minor_scaling[imnr] * ncl] * vmr_fact * dry_fact;
                        }
                    }
                }

                if constexpr (block_size_z > 1) {
                    if (threadIdx.z == 0) {
                        scalings[threadIdx.y][threadIdx.x] = scaling;
                    }

                    __syncthreads();

                    scaling = scalings[threadIdx.y][threadIdx.x];

                    __syncthreads();
                }

                const int gpt_start = minor_limits_gpt[2*imnr]-1;
                const int gpt_end = minor_limits_gpt[2*imnr+1];
                const int gpt_offs = 1-idx_tropo;
                const int iflav = gpoint_flavor[2*gpt_start + gpt_offs]-1;

                const int idx_fcl2 = 2 * 2 * (icol + ilay*ncol + iflav*ncol*nlay);
                const int idx_fcl1 = 2 * (icol + ilay*ncol + iflav*ncol*nlay);

                const FloatFMinor* kfminor = &fminor[idx_fcl2];
                const FloatKMinor* kin = &kminor[0];

                const int j0 = jeta[idx_fcl1/2].x;
                const int j1 = jeta[idx_fcl1/2].y;
                const int kjtemp = jtemp[idx_collay];
                const int band_gpt = gpt_end-gpt_start;
                const int gpt_offset = kminor_start[imnr]-1;

                for (int igpt=threadIdx.z; igpt<band_gpt; igpt+=block_size_z)
                {
                    auto ltau_minor = Float(kfminor[0]) * Float(kin[(kjtemp-1) + (j0-1)*ntemp + (igpt+gpt_offset)*ntemp*neta]) +
                                        Float(kfminor[1]) * Float(kin[(kjtemp-1) +  j0   *ntemp + (igpt+gpt_offset)*ntemp*neta]) +
                                        Float(kfminor[2]) * Float(kin[kjtemp     + (j1-1)*ntemp + (igpt+gpt_offset)*ntemp*neta]) +
                                        Float(kfminor[3]) * Float(kin[kjtemp     +  j1   *ntemp + (igpt+gpt_offset)*ntemp*neta]);

                    const int idx_out = icol + ilay*ncol + (igpt+gpt_start)*ncol*nlay;
                    tau[idx_out] += FloatTau(ltau_minor * scaling);
                }
            }
        }
    }
}
*/

/*
__global__
void gas_optical_depths_minor_reference_kernel(
        const int ncol, const int nlay, const int ngpt,
        const int ngas, const int nflav, const int ntemp, const int neta,
        const int nscale,
        const int nminor,
        const int nminork,
        const int idx_h2o, const int idx_tropo,
        const int* __restrict__ gpoint_flavor,
        const Float* __restrict__ kminor,
        const int* __restrict__ minor_limits_gpt,
        const Bool* __restrict__ minor_scales_with_density,
        const Bool* __restrict__ scale_by_complement,
        const int* __restrict__ idx_minor,
        const int* __restrict__ idx_minor_scaling,
        const int* __restrict__ kminor_start,
        const Float* __restrict__ play,
        const Float* __restrict__ tlay,
        const Float* __restrict__ col_gas,
        const FloatFMinor* __restrict__ fminor,
        const int* __restrict__ jeta,
        const int* __restrict__ jtemp,
        const Bool* __restrict__ tropo,
        Float* __restrict__ tau,
        Float* __restrict__ tau_minor)
{
    // Fetch the three coordinates.
    const int icol = blockIdx.x * blockDim.x + threadIdx.x;
    const int ilay = blockIdx.y * blockDim.y + threadIdx.y;

    const Float PaTohPa = 0.01;
    const int ncl = ncol * nlay;

    if ((icol < ncol) && (ilay < nlay))
    {
        const int idx_collay = icol + ilay*ncol;
        const int idx_collaywv = icol + ilay*ncol + idx_h2o*ncl;

        if (tropo[idx_collay] == idx_tropo)
        {
            for (int imnr = 0; imnr < nscale; ++imnr)
            {
                Float scaling = col_gas[idx_collay + idx_minor[imnr] * ncl];

                if (minor_scales_with_density[imnr])
                {
                    scaling *= PaTohPa * play[idx_collay] / tlay[idx_collay];

                    if (idx_minor_scaling[imnr] > 0)
                    {
                        Float vmr_fact = Float(1.) / col_gas[idx_collay];
                        Float dry_fact = Float(1.) / (Float(1.) + col_gas[idx_collaywv] * vmr_fact);

                        if (scale_by_complement[imnr])
                            scaling *= (Float(1.) - col_gas[idx_collay + idx_minor_scaling[imnr] * ncl] * vmr_fact * dry_fact);
                        else
                            scaling *= col_gas[idx_collay + idx_minor_scaling[imnr] * ncl] * vmr_fact * dry_fact;
                    }
                }

                const int gpt_start = minor_limits_gpt[2*imnr]-1;
                const int gpt_end = minor_limits_gpt[2*imnr+1];
                const int gpt_offs = 1-idx_tropo;
                const int iflav = gpoint_flavor[2*gpt_start + gpt_offs]-1;
                const int idx_fcl2 = 2 * 2 * (iflav + icol*nflav + ilay*ncol*nflav);
                const int idx_fcl1 = 2 * (iflav + icol*nflav + ilay*ncol*nflav);

                const Float* kfminor = &fminor[idx_fcl2];
                const Float* kin = &kminor[kminor_start[imnr]-1];
                const int j0 = jeta[idx_fcl1];
                const int j1 = jeta[idx_fcl1+1];
                const int kjtemp = jtemp[idx_collay];
                const int band_gpt = gpt_end-gpt_start;

                for (int igpt=0; igpt<band_gpt; ++igpt) {
                    Float ltau_minor = kfminor[0] * kin[igpt + (j0-1)*nminork + (kjtemp-1)*neta*nminork] +
                                    kfminor[1] * kin[igpt +  j0   *nminork + (kjtemp-1)*neta*nminork] +
                                    kfminor[2] * kin[igpt + (j1-1)*nminork + kjtemp    *neta*nminork] +
                                    kfminor[3] * kin[igpt +  j1   *nminork + kjtemp    *neta*nminork];

                    const int idx_out = (igpt+gpt_start) + ilay*ngpt + icol*nlay*ngpt;
                    tau[idx_out] += ltau_minor * scaling;
                }

            }
        }
    }
}
*/


__global__
void compute_tau_rayleigh_kernel(
        const int ncol, const int nlay, const int nbnd, const int ngpt,
        const int ngas, const int nflav, const int neta, const int npres, const int ntemp,
        const int* __restrict__ gpoint_flavor,
        const int* __restrict__ band_lims_gpt,
        const Float* __restrict__ krayl,
        int idx_h2o, const FloatColDry* __restrict__ col_dry, const FloatColGas* __restrict__ col_gas,
        const FloatFMinor* __restrict__ fminor, const int* __restrict__ jeta,
        const Bool* __restrict__ tropo, const int* __restrict__ jtemp,
        FloatTau* __restrict__ tau_rayleigh)
{
    // Fetch the three coordinates.
    const int icol = blockIdx.x*blockDim.x + threadIdx.x;
    const int ilay = blockIdx.y*blockDim.y + threadIdx.y;

    if ( (icol < ncol) && (ilay < nlay) )
    {
        const int idx_collay = icol + ilay*ncol;
        const int idx_collaywv = icol + ilay*ncol + idx_h2o*nlay*ncol;
        const int itropo = !tropo[idx_collay];

        const int idx_krayl = itropo*ntemp*neta*ngpt;
        const int jtempl = jtemp[idx_collay];

        for (int igpt=0; igpt<ngpt; ++igpt)
        {
            const int iflav = gpoint_flavor[itropo+2*igpt]-1;

            const int idx_fcl2 = 2*2*(icol + ilay*ncol + iflav*ncol*nlay);
            const int idx_fcl1 =   2*(icol + ilay*ncol + iflav*ncol*nlay);

            const int j0 = jeta[idx_fcl1  ];
            const int j1 = jeta[idx_fcl1+1];

            const Float kloc = Float(fminor[idx_fcl2+0]) * krayl[idx_krayl + (jtempl-1) + (j0-1)*ntemp + igpt*ntemp*neta] +
                               Float(fminor[idx_fcl2+1]) * krayl[idx_krayl + (jtempl-1) +  j0   *ntemp + igpt*ntemp*neta] +
                               Float(fminor[idx_fcl2+2]) * krayl[idx_krayl + (jtempl  ) + (j1-1)*ntemp + igpt*ntemp*neta] +
                               Float(fminor[idx_fcl2+3]) * krayl[idx_krayl + (jtempl  ) +  j1   *ntemp + igpt*ntemp*neta];

            const int idx_out = icol + ilay*ncol + igpt*ncol*nlay;
            auto result = kloc * (Float(col_gas[idx_collaywv]) + Float(col_dry[idx_collay]));
            tau_rayleigh[idx_out] = FloatTau(result);
        }
    }
}


__global__
void combine_abs_and_rayleigh_kernel(
        const int ncol, const int nlay, const int ngpt, const Float tmin,
        const FloatTau* __restrict__ tau_abs, const FloatTau* __restrict__ tau_rayleigh,
        FloatTau* __restrict__ tau, FloatOptical* __restrict__ ssa, FloatOptical* __restrict__ g)
{
    // Fetch the three coordinates.
    const int icol = blockIdx.x*blockDim.x + threadIdx.x;
    const int ilay = blockIdx.y*blockDim.y + threadIdx.y;
    const int igpt = blockIdx.z*blockDim.z + threadIdx.z;

    if ( (icol < ncol) && (ilay < nlay) && (igpt < ngpt) )
    {
        const int idx = icol + ilay*ncol + igpt*ncol*nlay;

        const Float tau_tot = Float(tau_abs[idx]) + Float(tau_rayleigh[idx]);

        tau[idx] = FloatTau(tau_tot);
        g  [idx] = Float(0.);

        if (Float(tau_tot)>(Float(2.)*tmin))
            ssa[idx] = FloatOptical(Float(tau_rayleigh[idx])/tau_tot);
        else
            ssa[idx] = FloatOptical(0.);
    }
}
