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

#include <cuda.h>

#include <thrust/execution_policy.h>
#include <thrust/host_vector.h>

#include <thrust/system/cuda/vector.h>
#include <thrust/system/cuda/execution_policy.h>
#include <thrust/system/omp/execution_policy.h>
#include <thrust/system/tbb/execution_policy.h>

#include <cassert>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <map>
#include <stack>
#include <vector>

#include "grapple_map.h"

enum grapple_type
{
    GRAPPLE_CPP,
    GRAPPLE_OMP,
    GRAPPLE_TBB,
    GRAPPLE_CUDA,
    GRAPPLE_CROSS,
};

class grapple_data
{
public:
    grapple_type system;
    int   func_id;
    int   stack_frame;
    int   mem_size;
    float time;
    cudaStream_t stream;

    grapple_data(void) : system(GRAPPLE_CPP), func_id(-1), stack_frame(0), mem_size(0), time(0) {}

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
        std::string funcname(grapple_map.find(data.func_id));
        std::string sysname;

        switch(data.system)
        {
          case GRAPPLE_CPP :
              sysname = "cpp ";
              break;
          case GRAPPLE_TBB :
              sysname = "tbb ";
              break;
          case GRAPPLE_OMP :
              sysname = "omp ";
              break;
          case GRAPPLE_CUDA :
              sysname = "cuda";
              break;
          case GRAPPLE_CROSS :
              sysname = "d->h";
              break;
          default:
              sysname = "unk ";
        }

        output << "[" << sysname << "] "
               << std::string(data.stack_frame, '\t')
               << std::setw(23) << funcname       << " : "
               << std::setw( 8) << data.time      << " (ms), allocated : "
               << std::setw(10) << data.mem_size  << " bytes";

        return output;
    }
};

struct grapple_system : public thrust::detail::execution_policy_base<grapple_system>
{
private:

    typedef thrust::detail::execution_policy_base<grapple_system> Parent;

    const static size_t STACK_SIZE = 100;
    cudaEvent_t tstart[STACK_SIZE];
    cudaEvent_t tstop[STACK_SIZE];

    int func_index[STACK_SIZE];
    int stack_frame;
    int abs_index;
    grapple_type system;

    std::stack<int> stack;

public:

    typedef char value_type;

    std::vector<grapple_data> data;

    grapple_system(void) : Parent(), stack_frame(0), abs_index(0), system(GRAPPLE_CPP)
    {
        data.reserve(100);
    }

    ~grapple_system(void)
    {
        print();
    }

    void start(const int func_num)
    {
        func_index[stack_frame] = func_num;

        cudaEventCreate(&tstart[stack_frame]);
        cudaEventCreate(&tstop[stack_frame]);
        cudaEventRecord(tstart[stack_frame++], 0);

        data.push_back(grapple_data());
        stack.push(abs_index++);
    }

    void stop(void)
    {
        float elapsed_time;
        cudaEventRecord(tstop[--stack_frame], 0);
        cudaEventSynchronize(tstop[stack_frame]);
        cudaEventElapsedTime(&elapsed_time, tstart[stack_frame], tstop[stack_frame]);

        int index = stack.top();
        data[index].set_data(system, stack_frame, func_index[stack_frame], elapsed_time);
        stack.pop();
    }

    char *allocate(std::ptrdiff_t num_bytes)
    {
        int index = stack.top();
        data[index].mem_size += num_bytes;
        char* ret;

        switch(system)
        {
          case GRAPPLE_CUDA :
              ret = thrust::cuda::malloc<char>(num_bytes).get();
              break;
          default:
              ret = thrust::malloc<char>(thrust::cpp::tag(), num_bytes).get();
        }

        return ret;
    }

    void deallocate(char *ptr, size_t num_bytes)
    {
        switch(system)
        {
          case GRAPPLE_CUDA :
              thrust::cuda::free(thrust::cuda::pointer<char>(ptr));
              break;
          default:
              thrust::free(thrust::cpp::tag(), ptr);
        }
    }

    template<typename System1, typename System2>
    thrust::system::cuda::detail::cross_system<System1,System2>
    policy(thrust::system::cuda::detail::cross_system<System1,System2> policy)
    {
        system = GRAPPLE_CROSS;
        return policy;
    }

    thrust::detail::execute_with_allocator<grapple_system, thrust::system::cpp::detail::execution_policy>
    policy(thrust::cpp::tag)
    {
        system = GRAPPLE_CPP;
        return thrust::cpp::par(*this);
    }

    thrust::detail::execute_with_allocator<grapple_system, thrust::system::omp::detail::execution_policy>
    policy(thrust::omp::tag)
    {
        system = GRAPPLE_OMP;
        return thrust::omp::par(*this);
    }

    thrust::detail::execute_with_allocator<grapple_system, thrust::system::tbb::detail::execution_policy>
    policy(thrust::tbb::tag)
    {
        system = GRAPPLE_TBB;
        return thrust::tbb::par(*this);
    }

    thrust::detail::execute_with_allocator<grapple_system, thrust::system::cuda::detail::execute_on_stream_base>
    policy(thrust::cuda::tag)
    {
        system = GRAPPLE_CUDA;
        return thrust::cuda::par(*this);
    }

    void print(void)
    {
        for(size_t i = 0; i < data.size(); i++)
            std::cout << std::right << "[" << std::setw(2) << i << "]" << std::left << data[i] << std::endl;
    }
};

#include "system_select.h"
#include "grapple_includes.h"
