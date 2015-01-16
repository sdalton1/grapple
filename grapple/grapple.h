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

#include <cassert>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <map>
#include <stack>
#include <vector>

#include "grapple_map.h"

class grapple_data
{
  public:
    int   func_id;
    int   stack_frame;
    int   mem_size;
    float time;
    cudaStream_t stream;

    grapple_data(void) : func_id(-1), stack_frame(0), mem_size(0), time(0) {}

    void set_data(int stack_index, int func_index, float elapsed_time)
    {
      stack_frame = stack_index;
      func_id = func_index;
      time = elapsed_time;
    }

    friend std::ostream &operator<<( std::ostream &output,
                                     const grapple_data &data )
    {
        std::string name(grapple_map.find(data.func_id));

        output << std::string(data.stack_frame, '\t')
               << std::setw(23) << name           << " : "
               << std::setw( 8) << data.time      << " (ms), allocated : "
               << std::setw(10) << data.mem_size  << " bytes";

        return output;
    }
};

struct grapple_system
  : public thrust::detail::execution_policy_base<grapple_system>
{
private:

    typedef thrust::detail::execution_policy_base<grapple_system> Parent;

    const static size_t STACK_SIZE = 100;
    cudaEvent_t tstart[STACK_SIZE];
    cudaEvent_t tstop[STACK_SIZE];

    int func_index[STACK_SIZE];
    int stack_frame;
    int abs_index;

    std::stack<int> stack;

public:

    typedef char value_type;

    std::vector<grapple_data> data;

    grapple_system(void) : Parent(), stack_frame(0), abs_index(0)
    {
        data.reserve(100);
    }

    ~grapple_system(void) {}

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
        data[index].set_data(stack_frame, func_index[stack_frame], elapsed_time);
        stack.pop();
    }

    char *allocate(std::ptrdiff_t num_bytes)
    {
        data.back().mem_size += num_bytes;
        return thrust::cuda::malloc<char>(num_bytes).get();
    }

    void deallocate(char *ptr, size_t num_bytes)
    {
        thrust::cuda::free(thrust::cuda::pointer<char>(ptr));
    }

    Parent& policy(void)
    {
        return reinterpret_cast<Parent&>(*this);
    }

    void print(void)
    {
        std::cout << std::left;
        for(size_t i = 0; i < data.size(); i++)
            std::cout << "[" << i << "]" << data[i] << std::endl;
    }
};

#include "grapple_includes.h"
