#
# This file is part of a C++ interface to the Radiative Transfer for Energetics (RTE)
#
FILE(GLOB sourcefiles_cuda_rt "../src_cuda_rt/*.cu")
include_directories("../include" "../include_rt" "../include_rt_kernels" SYSTEM ${INCLUDE_DIRS})

add_library(rte_rrtmgp_cuda_rt STATIC ${sourcefiles_cuda_rt})

if(USEHIP)
    find_package(HIP REQUIRED)
    target_link_libraries(rte_rrtmgp_cuda PUBLIC hip::host)
    target_link_libraries(rte_rrtmgp_cuda PUBLIC hip::device)
    set_source_files_properties(${sourcefiles_cuda_rt} PROPERTIES LANGUAGE HIP)
endif()

target_link_libraries(rte_rrtmgp_cuda_rt PRIVATE rte_rrtmgp_kernels_cuda_rt kmm kernel_float)
