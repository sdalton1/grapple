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
#include <thrust/copy.h>

template<typename InputIterator, typename OutputIterator>
OutputIterator copy(grapple_system &exec,
                    InputIterator first,
                    InputIterator last,
                    OutputIterator result)
{
    exec.start(THRUST_COPY);
    OutputIterator ret = thrust::copy(exec.policy(get_system(first,result)), first, last, result);
    exec.stop();

    return ret;
}

template<typename InputIterator, typename Size, typename OutputIterator>
OutputIterator copy_n(grapple_system &exec,
                      InputIterator first,
                      Size n,
                      OutputIterator result)
{
    exec.start(THRUST_COPY_N);
    OutputIterator ret = thrust::copy_n(exec.policy(get_system(first,result)), first, n, result);
    exec.stop();

    return ret;
}

template<typename InputIterator, typename OutputIterator, typename Predicate>
OutputIterator copy_if(grapple_system &exec,
                       InputIterator first,
                       InputIterator last,
                       OutputIterator result,
                       Predicate pred)
{
    exec.start(THRUST_COPY_IF);
    OutputIterator ret = thrust::copy_if(exec.policy(get_system(first,result)), first, last, result, pred);
    exec.stop();

    return ret;
}

template<typename InputIterator1, typename InputIterator2, typename OutputIterator, typename Predicate>
OutputIterator copy_if(grapple_system &exec,
                       InputIterator1 first,
                       InputIterator1 last,
                       InputIterator2 stencil,
                       OutputIterator result,
                       Predicate pred)
{
    exec.start(THRUST_COPY_IF);
    OutputIterator ret = thrust::copy_if(exec.policy(get_system(first,stencil,result)), first, last, stencil, result, pred);
    exec.stop();

    return ret;
}

