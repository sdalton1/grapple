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

#include <cuda.h>

#include <thrust/execution_policy.h>

#include <thrust/system/cuda/vector.h>
#include <thrust/system/cuda/execution_policy.h>
#include <thrust/system/omp/execution_policy.h>
#include <thrust/system/tbb/execution_policy.h>

#include <cstdlib>
#include <stack>
#include <vector>

#include <grapple/map.h>

namespace grapple
{

// forward definitions
class grapple_data;

class grapple_system : public thrust::execution_policy<grapple_system>
{
private:

    typedef thrust::execution_policy<grapple_system> Parent;

public:

    typedef char value_type;

    std::vector<grapple_data> data;

    grapple_system(void);

    ~grapple_system(void);

    void start(const int func_num);

    void stop(void);

    char *allocate(std::ptrdiff_t num_bytes, bool service = true);

    void deallocate(char *ptr, size_t num_bytes);

    void print(void);

    thrust::detail::execute_with_allocator<grapple_system, thrust::system::cpp::detail::execution_policy>
    policy(thrust::cpp::tag);

    thrust::detail::execute_with_allocator<grapple_system, thrust::system::omp::detail::execution_policy>
    policy(thrust::omp::tag);

    thrust::detail::execute_with_allocator<grapple_system, thrust::system::tbb::detail::execution_policy>
    policy(thrust::tbb::tag);

    template<typename System>
    thrust::system::cuda::detail::cross_system<thrust::cuda::tag,System>
    policy(thrust::system::cuda::detail::cross_system<thrust::cuda::tag,System> policy);

    template<typename System>
    thrust::system::cuda::detail::cross_system<System,thrust::cuda::tag>
    policy(thrust::system::cuda::detail::cross_system<System,thrust::cuda::tag> policy);

    thrust::detail::execute_with_allocator<grapple_system, thrust::system::cuda::detail::execute_on_stream_base>
    policy(thrust::cuda::tag);

protected:

    const static size_t STACK_SIZE = 100;
    cudaEvent_t tstart[STACK_SIZE];
    cudaEvent_t tstop[STACK_SIZE];

    int func_index[STACK_SIZE];
    int stack_frame;
    int abs_index;
    grapple_type system;

    std::stack<int> stack;
};

} // end grapple namespace

#include <grapple/grapple.inl>

