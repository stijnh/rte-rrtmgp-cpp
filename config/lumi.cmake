# LUMI
#
# module --force purge
# module load LUMI/25.03
# module load buildtools/25.03
# module load partition/G
# module load lumi-CPEtools
# module load PrgEnv-cray
# module load craype-x86-trento
# module load craype-accel-amd-gfx90a
# module load cray-libsci_acc
# module load rocm
# module load Boost
# module load cray-hdf5
# module load cray-netcdf
# module load Szip

set(CMAKE_POLICY_DEFAULT_CMP0077 NEW)

set(USEHIP TRUE)
set(USECUDA FALSE)
set(USEINTEL FALSE)

set(KMM_USE_HIP ON)
set(KERNEL_FLOAT_LANGUAGE "HIP")

set(AMDGPU_TARGETS "gfx90a")
set(GPU_TARGETS "gfx90a")
set(CMAKE_HIP_ARCHITECTURES "gfx90a")
set(USER_CXX_FLAGS "-std=c++17")
set(USER_CXX_FLAGS_RELEASE "-Ofast")
set(USER_CXX_FLAGS_DEBUG "-O0 -g -Wall -Wno-unknown-pragmas")
set(USER_HIP_FLAGS "-std=c++17")
set(USER_HIP_FLAGS_RELEASE "-Ofast")
set(USER_HIP_FLAGS_DEBUG "-O0 -g -G -Wall")

set(NETCDF_LIB_C "netcdf")
set(HDF5_LIB "hdf5")
set(SZIP_LIB "sz")
set(LIBS ${NETCDF_LIB_C} ${HDF5_LIB} ${SZIP_LIB})
include_directories(/opt/cray/pe/netcdf/4.9.0.17/include)

add_definitions(-DRTE_USE_CBOOL)
