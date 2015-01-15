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
#include <thrust/count.h>

template<typename InputIterator, typename EqualityComparable>
typename thrust::iterator_traits<InputIterator>::difference_type
count(grapple_system &exec, InputIterator first, InputIterator last, const EqualityComparable& value)
{
    exec.start(thrust_mapper::thrustMap.find(__FUNCTION__)->second);
    typename thrust::iterator_traits<InputIterator>::difference_type ret =
      thrust::count(exec.policy(), first, last, value);
    exec.stop();

    return ret;
}

template<typename InputIterator, typename Predicate>
typename thrust::iterator_traits<InputIterator>::difference_type
count_if(grapple_system &exec, InputIterator first, InputIterator last, Predicate pred)
{
    exec.start(thrust_mapper::thrustMap.find(__FUNCTION__)->second);
    typename thrust::iterator_traits<InputIterator>::difference_type ret =
      thrust::count_if(exec.policy(), first, last, pred);
    exec.stop();

    return ret;
}

