/*
 *  Copyright 2008-2013 NVIDIA Corporation
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 * *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 */


#pragma once

#include <thrust/detail/config.h>
#include <thrust/for_each.h>

template<typename InputIterator,
         typename UnaryFunction>
InputIterator for_each(grapple_system &exec,
                       InputIterator first,
                       InputIterator last,
                       UnaryFunction f)
{
    exec.start(thrust_mapper::thrustMap.find(__FUNCTION__)->second);
    InputIterator ret = thrust::for_each(exec.policy(), first, last, f);
    exec.stop();

    return ret;
}

template<typename InputIterator,
         typename Size,
         typename UnaryFunction>
InputIterator for_each_n(grapple_system &exec,
                         InputIterator first,
                         Size n,
                         UnaryFunction f)
{
    exec.start(thrust_mapper::thrustMap.find(__FUNCTION__)->second);
    InputIterator ret = thrust::for_each_n(exec.policy(), first, n, f);
    exec.stop();

    return ret;
}

