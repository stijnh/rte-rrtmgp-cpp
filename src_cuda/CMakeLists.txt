#
# This file is part of a C++ interface to the Radiative Transfer for Energetics (RTE)
#
FILE(GLOB sourcefiles_cuda "../src_cuda/*.cu")
include_directories("../include" "../include_kernels_cuda" SYSTEM ${INCLUDE_DIRS})

add_library(rte_rrtmgp_cuda STATIC ${sourcefiles_cuda})
target_link_libraries(rte_rrtmgp_cuda PUBLIC rte_rrtmgp_kernels_cuda)

if(USEHIP)
    find_package(HIP REQUIRED)
    target_link_libraries(rte_rrtmgp_cuda PUBLIC hip::host)
    target_link_libraries(rte_rrtmgp_cuda PUBLIC hip::device)
    set_source_files_properties(${sourcefiles_cuda} PROPERTIES LANGUAGE HIP)
endif()

target_link_libraries(rte_rrtmgp_cuda PRIVATE kmm kernel_float)
