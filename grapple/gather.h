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
#include <thrust/gather.h>

namespace grapple
{

template<typename InputIterator,
         typename RandomAccessIterator,
         typename OutputIterator>
OutputIterator gather(grapple_system &exec,
                      InputIterator                                               map_first,
                      InputIterator                                               map_last,
                      RandomAccessIterator                                        input_first,
                      OutputIterator                                              result)
{
    exec.start(THRUST_GATHER);
    OutputIterator ret = thrust::gather(exec.policy(get_system(map_first,input_first,result)), map_first, map_last, input_first, result);
    exec.stop();

    return ret;
}

template<typename InputIterator1,
         typename InputIterator2,
         typename RandomAccessIterator,
         typename OutputIterator>
OutputIterator gather_if(grapple_system &exec,
                         InputIterator1                                              map_first,
                         InputIterator1                                              map_last,
                         InputIterator2                                              stencil,
                         RandomAccessIterator                                        input_first,
                         OutputIterator                                              result)
{
    exec.start(THRUST_GATHER_IF);
    OutputIterator ret = thrust::gather_if(exec.policy(get_system(map_first,stencil,input_first,result)), map_first, map_last, stencil, input_first, result);
    exec.stop();

    return ret;
}

template<typename InputIterator1,
         typename InputIterator2,
         typename RandomAccessIterator,
         typename OutputIterator,
         typename Predicate>
OutputIterator gather_if(grapple_system &exec,
                         InputIterator1                                              map_first,
                         InputIterator1                                              map_last,
                         InputIterator2                                              stencil,
                         RandomAccessIterator                                        input_first,
                         OutputIterator                                              result,
                         Predicate                                                   pred)
{
    exec.start(THRUST_GATHER_IF);
    OutputIterator ret = thrust::gather_if(exec.policy(get_system(map_first,stencil,input_first,result)), map_first, map_last, stencil, input_first, result, pred);
    exec.stop();

    return ret;
}

}

