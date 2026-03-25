#
# This file is part of a C++ interface to the Radiative Transfer for Energetics (RTE)
#
FILE(GLOB sourcefiles "../src/*.cpp")
include_directories("../include" SYSTEM ${INCLUDE_DIRS})

add_library(rte_rrtmgp STATIC ${sourcefiles} Aerosol_optics.cpp ../include/Aerosol_optics.h)

if(USECUDA)
  message(STATUS "CUDA: Enabled.")
  target_compile_definitions(rte_rrtmgp PRIVATE USECUDA)
elseif(USEHIP)
  message(STATUS "HIP: Enabled.")
  target_compile_definitions(rte_rrtmgp PRIVATE USEHIP)
endif()

# Set the precision of the build.
if(ACCURACY)
  message(STATUS "Accuracy level: ${ACCURACY}")
  target_compile_definitions(rte_rrtmgp PRIVATE RTE_ACCURACY=${ACCURACY})
else()
  message(STATUS "Accuracy level: not set, default to level 0")
  target_compile_definitions(rte_rrtmgp PRIVATE RTE_ACCURACY=0)
endif()

target_link_libraries(rte_rrtmgp PRIVATE kmm rte_rrtmgp_kernels kernel_float)
