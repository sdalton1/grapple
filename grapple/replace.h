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
#include <thrust/replace.h>

template<typename ForwardIterator, typename T>
void replace(grapple_system &exec,
             ForwardIterator first, ForwardIterator last,
             const T &old_value,
             const T &new_value)
{
    exec.start(thrust_mapper::thrustMap.find(__FUNCTION__)->second);
    thrust::replace(exec.policy(), first, last, old_value, new_value);
    exec.stop();
}

template<typename ForwardIterator, typename Predicate, typename T>
void replace_if(grapple_system &exec,
                ForwardIterator first, ForwardIterator last,
                Predicate pred,
                const T &new_value)
{
    exec.start(thrust_mapper::thrustMap.find(__FUNCTION__)->second);
    thrust::replace_if(exec.policy(), first, last, pred, new_value);
    exec.stop();
}

template<typename ForwardIterator, typename InputIterator, typename Predicate, typename T>
void replace_if(grapple_system &exec,
                ForwardIterator first, ForwardIterator last,
                InputIterator stencil,
                Predicate pred,
                const T &new_value)
{
    exec.start(thrust_mapper::thrustMap.find(__FUNCTION__)->second);
    thrust::replace_if(exec.policy(), first, last, stencil, pred, new_value);
    exec.stop();
}

template<typename InputIterator, typename OutputIterator, typename T>
OutputIterator replace_copy(grapple_system &exec,
                            InputIterator first, InputIterator last,
                            OutputIterator result,
                            const T &old_value,
                            const T &new_value)
{
    exec.start(thrust_mapper::thrustMap.find(__FUNCTION__)->second);
    OutputIterator ret = thrust::replace_copy(exec.policy(), first, last, result, old_value, new_value);
    exec.stop();

    return ret;
}

template<typename InputIterator, typename OutputIterator, typename Predicate, typename T>
OutputIterator replace_copy_if(grapple_system &exec,
                               InputIterator first, InputIterator last,
                               OutputIterator result,
                               Predicate pred,
                               const T &new_value)
{
    exec.start(thrust_mapper::thrustMap.find(__FUNCTION__)->second);
    OutputIterator ret = thrust::replace_copy_if(exec.policy(), first, last, result, pred, new_value);
    exec.stop();

    return ret;
}

template<typename InputIterator1, typename InputIterator2, typename OutputIterator, typename Predicate, typename T>
OutputIterator replace_copy_if(grapple_system &exec,
                               InputIterator1 first, InputIterator1 last,
                               InputIterator2 stencil,
                               OutputIterator result,
                               Predicate pred,
                               const T &new_value)
{
    exec.start(thrust_mapper::thrustMap.find(__FUNCTION__)->second);
    OutputIterator ret = thrust::replace_copy_if(exec.policy(), first, last, stencil, result, pred, new_value);
    exec.stop();

    return ret;
}

