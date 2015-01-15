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
#include <thrust/reverse.h>

template<typename DerivedPolicy, typename BidirectionalIterator>
void reverse(grapple_system &exec,
             BidirectionalIterator first,
             BidirectionalIterator last)
{
    exec.start(THRUST_REVERSE);
    thrust::reverse(exec.policy(), first, last);
    exec.stop();
}

template<typename BidirectionalIterator, typename OutputIterator>
OutputIterator reverse_copy(grapple_system &exec,
                            BidirectionalIterator first,
                            BidirectionalIterator last,
                            OutputIterator result)
{
    exec.start(THRUST_REVERSE_COPY);
    OutputIterator ret = thrust::reverse_copy(exec.policy(), first, last, result);
    exec.stop();

    return ret;
}

