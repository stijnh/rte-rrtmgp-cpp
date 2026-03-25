#
# This file is part of a C++ interface to the Radiative Transfer for Energetics (RTE)
#
FILE(GLOB sourcefiles_cuda_rt "../src_cuda_rt/*.cu")
include_directories("../include" "../include_rt" "../include_rt_kernels" SYSTEM ${INCLUDE_DIRS})

add_library(rte_rrtmgp_cuda_rt STATIC ${sourcefiles_cuda_rt})

if(USECUDA)
  message(STATUS "CUDA: Enabled.")
  target_compile_definitions(rte-rrtmgp_cuda_rt PRIVATE USECUDA)
elseif(USEHIP)
  message(STATUS "HIP: Enabled.")
  target_compile_definitions(rte-rrtmgp_cuda_rt PRIVATE USEHIP)
endif()

# Set the precision of the build.
if(ACCURACY)
  message(STATUS "Accuracy level: ${ACCURACY}")
  target_compile_definitions(rte-rrtmgp_cuda_rt PRIVATE RTE_ACCURACY=${ACCURACY})
else()
  message(STATUS "Accuracy level: not set, default to level 0")
  target_compile_definitions(rte-rrtmgp_cuda_rt PRIVATE RTE_ACCURACY=0)
endif()

if(USEHIP)
    find_package(HIP REQUIRED)
    target_link_libraries(rte_rrtmgp_cuda_rt PUBLIC hip::host)
    target_link_libraries(rte_rrtmgp_cuda_rt PUBLIC hip::device)
    set_source_files_properties(${sourcefiles_cuda_rt} PROPERTIES LANGUAGE HIP)
endif()

target_link_libraries(rte_rrtmgp_cuda_rt PRIVATE rte_rrtmgp_kernels_cuda_rt kmm kernel_float)
