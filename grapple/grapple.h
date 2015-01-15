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

#include <thrust/system/cuda/vector.h>
#include <thrust/system/cuda/execution_policy.h>

#include <cstdlib>
#include <iostream>
#include <map>
#include <cassert>

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

    friend std::ostream &operator<<( std::ostream &output,
                                     const grapple_data &data )
    {
        std::string name(thrust_mapper::thrustReverseMap.find(data.func_id)->second);

        output << "(on stream " << data.stream << ") "
               << std::string(data.stack_frame, '\t')
               << std::setw(23) << name           << " : "
               << std::setw( 8) << data.time      << " (ms), allocated : "
               << std::setw(10) << data.mem_size  << " bytes";

        return output;
    }
};

struct grapple_system
  : public thrust::system::cuda::detail::execute_on_stream_base<grapple_system>
{
private:

    typedef thrust::system::cuda::detail::execute_on_stream_base<grapple_system> Parent;
    typedef thrust::detail::execute_with_allocator<grapple_system, thrust::system::cuda::detail::execution_policy> Allocator;

    cudaEvent_t tstart;
    cudaEvent_t tstop;
    cudaStream_t s;
    int func_index;
    int stack_frame;

public:

    typedef char value_type;

    thrust::host_vector< grapple_data > data;

    grapple_system(void) : Parent(), func_index(-1), stack_frame(0)
    {
        data.reserve(100);
    }

    ~grapple_system(void) {}

    void start(const int func_num)
    {
        s = stream(*this);
        func_index = func_num;
        stack_frame++;

        cudaEventCreate(&tstart);
        cudaEventCreate(&tstop);
        cudaEventRecord(tstart, 0);

        data.push_back(grapple_data());
    }

    void stop(void)
    {
        float elapsed_time;
        cudaEventRecord(tstop, 0);
        cudaEventSynchronize(tstop);
        cudaEventElapsedTime(&elapsed_time, tstart, tstop);

        data.back().func_id = func_index;
        data.back().stack_frame = --stack_frame;
        data.back().time = elapsed_time;
        data.back().stream = s;
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

    Allocator policy(void)
    {
        return Allocator(*this);
    }

    void print(void)
    {
        std::cout << std::left;

        for(size_t i = 0; i < data.size(); i++)
            std::cout << "[" << i << "]" << data[i] << std::endl;
    }
};

#include "grapple_includes.h"
