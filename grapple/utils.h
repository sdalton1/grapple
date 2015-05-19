/*
*  Copyright 2008-2013 NVIDIA Corporation
*
*  Licensed under the Apache License, Version 2.0 (the "License");
*  you may not use this file except in compliance with the License.
*  You may obtain a copy of the License at
*
*      http://www.apache.org/licenses/LICENSE-2.0
*
*  Unless required by applicable law or agreed to in writing, software
*  distributed under the License is distributed on an "AS IS" BASIS,
*  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
*  See the License for the specific language governing permissions and
*  limitations under the License.
*/
#pragma once

#include <thrust/device_vector.h>
#include <thrust/device_allocator.h>
#include <thrust/execution_policy.h>
#include <thrust/host_vector.h>
#include <thrust/version.h>

#if THRUST_DEVICE_SYSTEM == THRUST_DEVICE_SYSTEM_CUDA
// ensure thrust::cuda::free and thrust::cuda::pointer are available
#include <thrust/system/cuda/vector.h>
#include <thrust/system/cuda/execution_policy.h>
#endif

#include <algorithm>
#include <cstdarg>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <string>

#include <cstdio>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>

namespace grapple
{

std::string stringprintf(const char* format, ...) {
    va_list args;
    va_start(args, format);
    int len = vsnprintf(0, 0, format, args);
    va_end(args);

    // allocate space.
    std::string text;
    text.resize(len);

    va_start(args, format);
    vsnprintf(&text[0], len + 1, format, args);
    va_end(args);

    return text;
}

#if THRUST_DEVICE_COMPILER == THRUST_DEVICE_COMPILER_NVCC
__global__ void KernelVersionShim() { }

std::string getDeviceString(void)
{
    int _ordinal = -1;
    if(cudaGetDevice(&_ordinal) != cudaSuccess) {
        fprintf(stderr, "ERROR RETRIEVING CURRENT DEVICE ORDINAL %d\n", _ordinal);
        exit(0);
    }

    size_t freeMem, totalMem;
    if(cudaMemGetInfo(&freeMem, &totalMem) != cudaSuccess) {
        fprintf(stderr, "ERROR RETRIEVING MEM INFO FOR CUDA DEVICE %d\n", _ordinal);
        exit(0);
    }

    cudaDeviceProp _prop;
    if(cudaGetDeviceProperties(&_prop, _ordinal) != cudaSuccess) {
        fprintf(stderr, "ERROR RETRIEVING DEVICE %d PROPERTIES\n", _ordinal);
        exit(0);
    }

		// Get the compiler version for this device.
		cudaFuncAttributes attr;
		if(cudaFuncGetAttributes(&attr, KernelVersionShim) != cudaSuccess)
		{
			printf("NOT COMPILED WITH COMPATIBLE PTX VERSION FOR DEVICE %d\n", _ordinal);
			// The module wasn't compiled with support for this device.
      exit(0);
		}

		int _ptxVersion = 10 * attr.ptxVersion;

    double memBandwidth = (_prop.memoryClockRate * 1000.0) *
                          (_prop.memoryBusWidth / 8 * 2) / 1.0e9;

    return stringprintf(
               "%s : %8.3lf Mhz   (Ordinal %d)\n"
               "%d SMs enabled. Compute Capability sm_%d%d\n"
               "FreeMem: %6dMB   TotalMem: %6dMB   %2d-bit pointers.\n"
               "Mem Clock: %8.3lf Mhz x %d bits   (%5.1lf GB/s)\n"
               "ECC %s\n PTX Version : sm_%d\n\n",
               _prop.name, _prop.clockRate / 1000.0, _ordinal,
               _prop.multiProcessorCount, _prop.major, _prop.minor,
               (int)(freeMem / (1<< 20)), (int)(totalMem / (1<< 20)), 8 * sizeof(int*),
               _prop.memoryClockRate / 1000.0, _prop.memoryBusWidth, memBandwidth,
               _prop.ECCEnabled ? "Enabled" : "Disabled", _ptxVersion / 10);
}
#endif // end THRUST_DEVICE_COMPILER if

void print_config(void)
{
#if THRUST_DEVICE_COMPILER == THRUST_DEVICE_COMPILER_NVCC
    std::cout << " CUDA   v" << (CUDA_VERSION / 1000) << "." <<
              (CUDA_VERSION % 1000) / 10 << std::endl;
    std::cout << getDeviceString();
#endif // end THRUST_DEVICE_COMPILER if

#if THRUST_HOST_COMPILER == THRUST_HOST_COMPILER_MSVC
    std::cout << " MSVC   v" <<
#if _MSC_VER == 1800
              "12.0 (Visual Studio 2013)"
#elif _MSC_VER == 1700
              "11.0 (Visual Studio 2012)"
#elif _MSC_VER == 1600
              "10.0 (Visual Studio 2010)"
#elif _MSC_VER == 1500
              " 9.0 (Visual Studio 2008)"
#endif
              << std::endl;
#elif THRUST_HOST_COMPILER == THRUST_HOST_COMPILER_CLANG
    std::cout << " CLANG  v" << (THRUST_CLANG_VERSION / 10000) << "." <<
              (THRUST_CLANG_VERSION % 10000) / 100  << "." <<
              (THRUST_CLANG_VERSION % 10) << std::endl;
#elif THRUST_HOST_COMPILER == THRUST_HOST_COMPILER_GCC
    std::cout << " GCC    v" << (THRUST_GCC_VERSION / 10000) << "." <<
              (THRUST_GCC_VERSION % 10000) / 100  << "." <<
              (THRUST_GCC_VERSION % 10) << std::endl;
#else
    std::cout << " Unknown compiler " << std::endl;
#endif

    std::cout << " Thrust v" << THRUST_MAJOR_VERSION << "."
                             << THRUST_MINOR_VERSION << "."
                             << THRUST_SUBMINOR_VERSION
                             << std::endl;

    std::cout << std::endl;
}

} // end namespace grapple


