FILE(GLOB sourcefiles
    "../src_kernels_cuda/gas_optics_rrtmgp_kernels_launchers.cu"
    "../src_kernels_cuda/optical_props_kernels_launchers.cu"
    "../src_kernels_cuda/rte_solver_kernels_launchers.cu"
    "../src_kernels_cuda/subset_kernels_launchers.cu"
    "../src_kernels_cuda/fluxes_kernels_launchers.cu")

include_directories("../include" "../include_kernels_cuda" SYSTEM ${INCLUDE_DIRS})

add_library(rte_rrtmgp_kernels_cuda STATIC ${sourcefiles})

if(USECUDA)
  message(STATUS "CUDA: Enabled.")
  target_compile_definitions(rte_rrtmgp_kernels_cuda PRIVATE USECUDA)
elseif(USEHIP)
  message(STATUS "HIP: Enabled.")
  target_compile_definitions(rte_rrtmgp_kernels_cuda PRIVATE USEHIP)
endif()

# Set the precision of the build.
if(ACCURACY)
  message(STATUS "Accuracy level: ${ACCURACY}")
  target_compile_definitions(rte_rrtmgp_kernels_cuda PRIVATE RTE_ACCURACY=${ACCURACY})
else()
  message(STATUS "Accuracy level: not set, default to level 0")
  target_compile_definitions(rte_rrtmgp_kernels_cuda PRIVATE RTE_ACCURACY=0)
endif()

if(USEHIP)
    find_package(HIP REQUIRED)
    target_link_libraries(rte_rrtmgp_kernels_cuda PUBLIC hip::host)
    target_link_libraries(rte_rrtmgp_kernels_cuda PUBLIC hip::device)
    set_source_files_properties(${sourcefiles} PROPERTIES LANGUAGE HIP)
endif()

target_link_libraries(rte_rrtmgp_kernels_cuda PRIVATE kernel_float kmm)
