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

#include <thrust/detail/config.h>
#include <thrust/generate.h>

namespace grapple
{

template<typename ForwardIterator,
         typename Generator>
void generate(grapple_system &exec,
              ForwardIterator first,
              ForwardIterator last,
              Generator gen)
{
    exec.start(THRUST_GENERATE);
    thrust::generate(exec.policy(get_system(first)), first, last, gen);
    exec.stop();
}

template<typename OutputIterator,
         typename Size,
         typename Generator>
__host__ __device__
OutputIterator generate_n(grapple_system &exec,
                          OutputIterator first,
                          Size n,
                          Generator gen)
{
    exec.start(THRUST_GENERATE_N);
    OutputIterator ret = thrust::generate_n(exec.policy(get_system(first)), first, n, gen);
    exec.stop();

    return ret;
}

}
