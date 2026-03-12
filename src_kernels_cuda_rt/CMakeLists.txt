FILE(GLOB sourcefiles_rt
    "../src_kernels_cuda_rt/gas_optics_rrtmgp_kernels_launchers_rt.cu"
    "../src_kernels_cuda_rt/optical_props_kernels_launchers_rt.cu"
    "../src_kernels_cuda_rt/rte_solver_kernels_launchers_rt.cu"
    "../src_kernels_cuda_rt/subset_kernels_launchers_rt.cu"
    "../src_kernels_cuda_rt/gpt_combine_kernels_launchers_rt.cu"
    "../src_kernels_cuda_rt/raytracer_kernels.cu"
    "../src_kernels_cuda_rt/raytracer_kernels_bw.cu")


if(USECUDA)
    if(CMAKE_BUILD_TYPE STREQUAL "RELEASE")
        set_source_files_properties("../src_kernels_cuda_rt/raytracer_kernels.cu" PROPERTIES COMPILE_FLAGS "-use_fast_math")
        set_source_files_properties("../src_kernels_cuda_rt/raytracer_kernels_bw.cu" PROPERTIES COMPILE_FLAGS "-use_fast_math")
    endif()
elseif(USEHIP)
    if(CMAKE_BUILD_TYPE STREQUAL "RELEASE")
        set_source_files_properties("../src_kernels_cuda_rt/raytracer_kernels.cu" PROPERTIES COMPILE_FLAGS "-ffast-math")
        set_source_files_properties("../src_kernels_cuda_rt/raytracer_kernels_bw.cu" PROPERTIES COMPILE_FLAGS "-ffast-math")
    endif()
endif()

include_directories("../include" "../include_rt" "../include_rt_kernels" SYSTEM ${INCLUDE_DIRS})

add_library(rte_rrtmgp_kernels_cuda_rt STATIC ${sourcefiles_rt})

if(USEHIP)
    enable_language(HIP)
    set_source_files_properties(${sourcefiles_rt} PROPERTIES LANGUAGE HIP)
endif()

target_link_libraries(rte_rrtmgp_kernels_cuda_rt PRIVATE kmm kernel_float)


