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
#include <thrust/extrema.h>

template<typename ForwardIterator>
ForwardIterator min_element(grapple_system &exec, ForwardIterator first, ForwardIterator last)
{
    exec.start(THRUST_MIN_ELEMENT);
    ForwardIterator ret = thrust::min_element(exec.policy(get_system(first)), first, last);
    exec.stop();

    return ret;
}

template<typename ForwardIterator, typename BinaryPredicate>
ForwardIterator min_element(grapple_system &exec, ForwardIterator first, ForwardIterator last, BinaryPredicate comp)
{
    exec.start(THRUST_MIN_ELEMENT);
    ForwardIterator ret = thrust::min_element(exec.policy(get_system(first)), first, last, comp);
    exec.stop();

    return ret;
}

template<typename ForwardIterator>
ForwardIterator max_element(grapple_system &exec, ForwardIterator first, ForwardIterator last)
{
    exec.start(THRUST_MAX_ELEMENT);
    ForwardIterator ret = thrust::max_element(exec.policy(get_system(first)), first, last);
    exec.stop();

    return ret;
}

template<typename ForwardIterator, typename BinaryPredicate>
ForwardIterator max_element(grapple_system &exec, ForwardIterator first, ForwardIterator last, BinaryPredicate comp)
{
    exec.start(THRUST_MAX_ELEMENT);
    ForwardIterator ret = thrust::max_element(exec.policy(get_system(first)), first, last, comp);
    exec.stop();

    return ret;
}

template<typename ForwardIterator>
thrust::pair<ForwardIterator,ForwardIterator> minmax_element(grapple_system &exec, ForwardIterator first, ForwardIterator last)
{
    exec.start(THRUST_MINMAX_ELEMENT);
    thrust::pair<ForwardIterator,ForwardIterator> ret = thrust::minmax_element(exec.policy(get_system(first)), first, last);
    exec.stop();

    return ret;
}

template<typename ForwardIterator, typename BinaryPredicate>
thrust::pair<ForwardIterator,ForwardIterator> minmax_element(grapple_system &exec, ForwardIterator first, ForwardIterator last, BinaryPredicate comp)
{
    exec.start(THRUST_MINMAX_ELEMENT);
    thrust::pair<ForwardIterator,ForwardIterator> ret = thrust::minmax_element(exec.policy(get_system(first)), first, last, comp);
    exec.stop();

    return ret;
}

