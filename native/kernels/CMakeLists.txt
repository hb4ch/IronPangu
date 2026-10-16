cmake_minimum_required(VERSION 3.16)
project(inferfabric_cache_kernels LANGUAGES CXX)
set(SOC_VERSION "Ascend910_9391" CACHE STRING "Ascend SoC")
set(ASCEND_CANN_PACKAGE_PATH "/usr/local/Ascend/cann" CACHE PATH "CANN")
include(${ASCEND_CANN_PACKAGE_PATH}/tools/tikcpp/ascendc_kernel_cmake/ascendc.cmake)
ascendc_library(inferfabric_cache STATIC cache_quantization.cpp)

