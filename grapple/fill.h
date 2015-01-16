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
#include <thrust/fill.h>

template<typename ForwardIterator, typename T>
void fill(grapple_system &exec,
          ForwardIterator first,
          ForwardIterator last,
          const T &value)
{
    exec.start(THRUST_FILL);
    thrust::fill(thrust::cuda::par(exec), first, last, value);
    exec.stop();
}

template<typename OutputIterator, typename Size, typename T>
OutputIterator fill_n(grapple_system &exec,
                      OutputIterator first,
                      Size n,
                      const T &value)
{
    exec.start(THRUST_FILL_N);
    OutputIterator ret = thrust::fill_n(thrust::cuda::par(exec), first, n, value);
    exec.stop();

    return ret;
}

