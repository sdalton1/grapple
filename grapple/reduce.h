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
#include <thrust/reduce.h>

namespace grapple
{

template<typename InputIterator>
typename thrust::iterator_traits<InputIterator>::value_type
reduce(grapple_system &exec, InputIterator first, InputIterator last)
{
    exec.start(THRUST_REDUCE);
    typename thrust::iterator_traits<InputIterator>::value_type ret =
        thrust::reduce(exec.policy(get_system(first)), first, last);
    exec.stop();

    return ret;
}

template<typename InputIterator, typename T>
T reduce(grapple_system &exec,
         InputIterator first,
         InputIterator last,
         T init)
{
    exec.start(THRUST_REDUCE);
    T ret = thrust::reduce(exec.policy(get_system(first)), first, last, init);
    exec.stop();

    return ret;
}

template<typename InputIterator,
         typename T,
         typename BinaryFunction>
T reduce(grapple_system &exec,
         InputIterator first,
         InputIterator last,
         T init,
         BinaryFunction binary_op)
{
    exec.start(THRUST_REDUCE);
    T ret = thrust::reduce(exec.policy(get_system(first)), first, last, init, binary_op);
    exec.stop();

    return ret;
}

template<typename InputIterator1,
         typename InputIterator2,
         typename OutputIterator1,
         typename OutputIterator2>
thrust::pair<OutputIterator1,OutputIterator2>
reduce_by_key(grapple_system &exec,
              InputIterator1 keys_first,
              InputIterator1 keys_last,
              InputIterator2 values_first,
              OutputIterator1 keys_output,
              OutputIterator2 values_output)
{
    exec.start(THRUST_REDUCE_BY_KEY);
    thrust::pair<OutputIterator1,OutputIterator2> ret = thrust::reduce_by_key(exec.policy(get_system(keys_first,values_first,keys_output,values_output)),
            keys_first, keys_last, values_first, keys_output, values_output);
    exec.stop();

    return ret;
}

template<typename InputIterator1,
         typename InputIterator2,
         typename OutputIterator1,
         typename OutputIterator2,
         typename BinaryPredicate>
thrust::pair<OutputIterator1,OutputIterator2>
reduce_by_key(grapple_system &exec,
              InputIterator1 keys_first,
              InputIterator1 keys_last,
              InputIterator2 values_first,
              OutputIterator1 keys_output,
              OutputIterator2 values_output,
              BinaryPredicate binary_pred)
{
    exec.start(THRUST_REDUCE_BY_KEY);
    thrust::pair<OutputIterator1,OutputIterator2> ret = thrust::reduce_by_key(exec.policy(get_system(keys_first,values_first,keys_output,values_output)),
            keys_first, keys_last, values_first, keys_output, values_output, binary_pred);
    exec.stop();

    return ret;
}

template<typename InputIterator1,
         typename InputIterator2,
         typename OutputIterator1,
         typename OutputIterator2,
         typename BinaryPredicate,
         typename BinaryFunction>
thrust::pair<OutputIterator1,OutputIterator2>
reduce_by_key(grapple_system &exec,
              InputIterator1 keys_first,
              InputIterator1 keys_last,
              InputIterator2 values_first,
              OutputIterator1 keys_output,
              OutputIterator2 values_output,
              BinaryPredicate binary_pred,
              BinaryFunction binary_op)
{
    exec.start(THRUST_REDUCE_BY_KEY);
    thrust::pair<OutputIterator1,OutputIterator2> ret = thrust::reduce_by_key(exec.policy(get_system(keys_first,values_first,keys_output,values_output)),
            keys_first, keys_last, values_first, keys_output, values_output, binary_pred, binary_op);
    exec.stop();

    return ret;
}

}
