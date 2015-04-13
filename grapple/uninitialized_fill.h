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
#include <thrust/uninitialized_fill.h>

namespace grapple
{

template<typename ForwardIterator, typename T>
void uninitialized_fill(grapple_system &exec,
                        ForwardIterator first,
                        ForwardIterator last,
                        const T &x)
{
    exec.start(THRUST_UNINITIALIZED_FILL);
    ForwardIterator ret = thrust::uninitialized_fill(exec.policy(get_system(first)), first, last, x);
    exec.stop();

    return ret;
}


template<typename ForwardIterator, typename Size, typename T>
ForwardIterator uninitialized_fill_n(grapple_system &exec,
                                     ForwardIterator first,
                                     Size n,
                                     const T &x)
{
    exec.start(THRUST_UNINITIALIZED_FILL_N);
    ForwardIterator ret = thrust::uninitialized_fill_n(exec.policy(get_system(first)), first, n, x);
    exec.stop();

    return ret;
}

}

