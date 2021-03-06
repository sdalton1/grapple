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
#include <thrust/adjacent_difference.h>

namespace grapple
{

template<typename InputIterator, typename OutputIterator>
OutputIterator adjacent_difference(grapple_system &exec,
                                   InputIterator first, InputIterator last,
                                   OutputIterator result)
{
    exec.start(THRUST_ADJACENT_DIFFERENCE);
    OutputIterator ret = thrust::adjacent_difference(exec.policy(get_system(first,result)), first, last, result);
    exec.stop();

    return ret;
}

template<typename InputIterator, typename OutputIterator, typename BinaryFunction>
OutputIterator adjacent_difference(grapple_system &exec,
                                   InputIterator first, InputIterator last,
                                   OutputIterator result,
                                   BinaryFunction binary_op)
{
    exec.start(THRUST_ADJACENT_DIFFERENCE);
    OutputIterator ret = thrust::adjacent_difference(exec.policy(get_system(first,result)), first, last, result, binary_op);
    exec.stop();

    return ret;
}

}

