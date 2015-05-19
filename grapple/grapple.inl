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
#elif THRUST_DEVICE_SYSTEM == THRUST_DEVICE_SYSTEM_OMP
#include <thrust/system/omp/execution_policy.h>
#elif THRUST_DEVICE_SYSTEM == THRUST_DEVICE_SYSTEM_TBB
#include <thrust/system/tbb/execution_policy.h>
#endif

#include <grapple/map.h>
#include <grapple/system_select.h>
#include <grapple/thrust.h>
#include <grapple/utils.h>

#ifdef __CUDACC__
#include <grapple/gputimer.h>
#else
#include <grapple/cputimer.h>
#endif

#include <algorithm>
#include <cassert>
#include <cstdarg>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <map>
#include <stack>
#include <typeinfo>
#include <vector>

namespace grapple
{

class grapple_data
{
private:

    grapple_type system;
    int   func_id;
    int   stack_frame;
    float time;

public:

    int mem_size;

    grapple_data(void)
      : system(GRAPPLE_CPP), func_id(-1), stack_frame(0), time(0.0), mem_size(0)
    {}

    void set_data(grapple_type stack_system, int stack_index, int func_index, float elapsed_time)
    {
        system = stack_system;
        stack_frame = stack_index;
        func_id = func_index;
        time = elapsed_time;
    }

    friend std::ostream &operator<<( std::ostream &output,
                                     const grapple_data &data )
    {
        std::string funcname(grapple_thrust_map.find(data.func_id));
        std::string sysname(grapple_system_map.find(data.system));

        output << "[" << sysname << "] "
               << std::string(data.stack_frame, '\t')
               << std::setw(23) << funcname       << " : "
               << std::setw( 8) << data.time      << " (ms), allocated : "
               << std::setw(10) << data.mem_size  << " bytes";

        return output;
    }
};

grapple_system::grapple_system(void)
    : Parent(), stack_frame(0), abs_index(0), system(GRAPPLE_CPP)
{
    print_config();
    data.reserve(100);
}

grapple_system::~grapple_system(void)
{
    print();
}

void grapple_system::start(const int func_num)
{
    func_index[stack_frame] = func_num;
    tlist[stack_frame++].Start();

    data.push_back(grapple_data());
    stack.push(abs_index++);
}

void grapple_system::stop(void)
{
    tlist[--stack_frame].Stop();
    float elapsed_time = tlist[stack_frame].milliseconds_elapsed();

    int index = stack.top();
    data[index].set_data(system, stack_frame, func_index[stack_frame], elapsed_time);
    stack.pop();
}

char * grapple_system::allocate(std::ptrdiff_t num_bytes, bool service)
{
    int index = stack.top();
    data[index].mem_size += num_bytes;
    char* ret = NULL;

    if(service)
    {
        switch(system)
        {
        #ifdef __CUDACC__
        case GRAPPLE_CUDA :
            ret = thrust::device_malloc<char>(num_bytes).get();
            break;
        #endif
        default:
            ret = thrust::malloc<char>(thrust::cpp::tag(), num_bytes).get();
        }
    }

    return ret;
}

void grapple_system::deallocate(char *ptr, size_t num_bytes)
{
    switch(system)
    {
    #ifdef __CUDACC__
    case GRAPPLE_CUDA :
        thrust::cuda::free(thrust::cuda::pointer<char>(ptr));
        break;
    #endif
    default:
        thrust::free(thrust::cpp::tag(), ptr);
    }
}

typename grapple_system::Parent&
grapple_system::policy(void)
{
    return static_cast<Parent&>(*this);
}

thrust::detail::execute_with_allocator<grapple_system, thrust::system::cpp::detail::execution_policy>
grapple_system::policy(thrust::cpp::tag)
{
    system = GRAPPLE_CPP;
    return thrust::cpp::par(*this);
}

#if THRUST_DEVICE_SYSTEM == THRUST_DEVICE_SYSTEM_CUDA
thrust::detail::execute_with_allocator<grapple_system, thrust::system::cuda::detail::execute_on_stream_base>
grapple_system::policy(thrust::cuda::tag)
{
    system = GRAPPLE_CUDA;
    return thrust::cuda::par(*this);
}

template<typename System>
thrust::system::cuda::detail::cross_system<thrust::cuda::tag,System>
grapple_system::policy(thrust::system::cuda::detail::cross_system<thrust::cuda::tag,System> policy)
{
    system = GRAPPLE_D2H;
    return policy;
}

template<typename System>
thrust::system::cuda::detail::cross_system<System,thrust::cuda::tag>
grapple_system::policy(thrust::system::cuda::detail::cross_system<System,thrust::cuda::tag> policy)
{
    system = GRAPPLE_H2D;
    return policy;
}
#elif THRUST_DEVICE_SYSTEM == THRUST_DEVICE_SYSTEM_OMP
thrust::detail::execute_with_allocator<grapple_system, thrust::system::omp::detail::execution_policy>
grapple_system::policy(thrust::omp::tag)
{
    system = GRAPPLE_OMP;
    return thrust::omp::par(*this);
}
#elif THRUST_DEVICE_SYSTEM == THRUST_DEVICE_SYSTEM_TBB
thrust::detail::execute_with_allocator<grapple_system, thrust::system::tbb::detail::execution_policy>
grapple_system::policy(thrust::tbb::tag)
{
    system = GRAPPLE_TBB;
    return thrust::tbb::par(*this);
}
#endif

void grapple_system::print(void)
{
    for(size_t i = 0; i < data.size(); i++)
        std::cout << std::right << "[" << std::setw(2) << i << "]" << std::left << data[i] << std::endl;
}

} // end namespace grapple

