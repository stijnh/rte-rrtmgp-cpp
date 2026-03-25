#
# This file is part of a C++ interface to the Radiative Transfer for Energetics (RTE)
#
FILE(GLOB sourcefiles_cuda "../src_cuda/*.cu")
include_directories("../include" "../include_kernels_cuda" SYSTEM ${INCLUDE_DIRS})

add_library(rte_rrtmgp_cuda STATIC ${sourcefiles_cuda})

if(USECUDA)
  message(STATUS "CUDA: Enabled.")
  target_compile_definitions(rte_rrtmgp_cuda PRIVATE USECUDA)
elseif(USEHIP)
  message(STATUS "HIP: Enabled.")
  target_compile_definitions(rte_rrtmgp_cuda PRIVATE USEHIP)
endif()

# Set the precision of the build.
if(ACCURACY)
  message(STATUS "Accuracy level: ${ACCURACY}")
  target_compile_definitions(rte_rrtmgp_cuda PRIVATE RTE_ACCURACY=${ACCURACY})
else()
  message(STATUS "Accuracy level: not set, default to level 0")
  target_compile_definitions(rte_rrtmgp_cuda PRIVATE RTE_ACCURACY=0)
endif()

if(USEHIP)
    find_package(HIP REQUIRED)
    target_link_libraries(rte_rrtmgp_cuda PUBLIC hip::host)
    target_link_libraries(rte_rrtmgp_cuda PUBLIC hip::device)
    set_source_files_properties(${sourcefiles_cuda} PROPERTIES LANGUAGE HIP)
endif()

target_link_libraries(rte_rrtmgp_cuda PRIVATE rte_rrtmgp_kernels_cuda kmm kernel_float)
