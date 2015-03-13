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
#include <thrust/unique.h>

template<typename ForwardIterator>
ForwardIterator unique(grapple_system &exec,
                       ForwardIterator first,
                       ForwardIterator last)
{
    exec.start(THRUST_UNIQUE);
    ForwardIterator ret = thrust::unique(exec.policy(get_system(first)), first, last);
    exec.stop();

    return ret;
}


template<typename ForwardIterator,
         typename BinaryPredicate>
ForwardIterator unique(grapple_system &exec,
                       ForwardIterator first,
                       ForwardIterator last,
                       BinaryPredicate binary_pred)
{
    exec.start(THRUST_UNIQUE);
    ForwardIterator ret = thrust::unique(exec.policy(get_system(first)), first, last, binary_pred);
    exec.stop();

    return ret;
}

template<typename InputIterator,
         typename OutputIterator>
OutputIterator unique_copy(grapple_system &exec,
                           InputIterator first,
                           InputIterator last,
                           OutputIterator result)
{
    exec.start(THRUST_UNIQUE_COPY);
    OutputIterator ret = thrust::unique_copy(exec.policy(get_system(first,result)), first, last, result);
    exec.stop();

    return ret;
}


template<typename InputIterator,
         typename OutputIterator,
         typename BinaryPredicate>
OutputIterator unique_copy(grapple_system &exec,
                           InputIterator first,
                           InputIterator last,
                           OutputIterator result,
                           BinaryPredicate binary_pred)
{
    exec.start(THRUST_UNIQUE_COPY);
    OutputIterator ret = thrust::unique_copy(exec.policy(get_system(first,result)), first, last, result, binary_pred);
    exec.stop();

    return ret;
}

template<typename ForwardIterator1,
         typename ForwardIterator2>
thrust::pair<ForwardIterator1,ForwardIterator2>
unique_by_key(grapple_system &exec,
              ForwardIterator1 keys_first,
              ForwardIterator1 keys_last,
              ForwardIterator2 values_first)
{
    exec.start(THRUST_UNIQUE_BY_KEY);
    thrust::pair<ForwardIterator1,ForwardIterator2> ret = thrust::unique_by_key(exec.policy(get_system(keys_first,values_first)),
                                                        keys_first, keys_last, values_first);
    exec.stop();

    return ret;
}

template<typename ForwardIterator1,
         typename ForwardIterator2,
         typename BinaryPredicate>
thrust::pair<ForwardIterator1,ForwardIterator2>
unique_by_key(grapple_system &exec,
              ForwardIterator1 keys_first,
              ForwardIterator1 keys_last,
              ForwardIterator2 values_first,
              BinaryPredicate binary_pred)
{
    exec.start(THRUST_UNIQUE_BY_KEY);
    thrust::pair<ForwardIterator1,ForwardIterator2> ret = thrust::unique_by_key(exec.policy(get_system(keys_first,values_first)),
                                                        keys_first, keys_last, values_first, binary_pred);
    exec.stop();

    return ret;
}

template<typename InputIterator1,
         typename InputIterator2,
         typename OutputIterator1,
         typename OutputIterator2>
thrust::pair<OutputIterator1,OutputIterator2>
unique_by_key_copy(grapple_system &exec,
                   InputIterator1 keys_first,
                   InputIterator1 keys_last,
                   InputIterator2 values_first,
                   OutputIterator1 keys_result,
                   OutputIterator2 values_result)
{
    exec.start(THRUST_UNIQUE_BY_KEY_COPY);
    thrust::pair<OutputIterator1,OutputIterator2> ret = thrust::unique_by_key_copy(exec.policy(get_system(keys_first,values_first,keys_result,values_result)),
                                                        keys_first, keys_last, values_first, keys_result,
                                                        values_result);
    exec.stop();

    return ret;
}

template<typename InputIterator1,
         typename InputIterator2,
         typename OutputIterator1,
         typename OutputIterator2,
         typename BinaryPredicate>
thrust::pair<OutputIterator1,OutputIterator2>
unique_by_key_copy(grapple_system &exec,
                   InputIterator1 keys_first,
                   InputIterator1 keys_last,
                   InputIterator2 values_first,
                   OutputIterator1 keys_result,
                   OutputIterator2 values_result,
                   BinaryPredicate binary_pred)
{
    exec.start(THRUST_UNIQUE_BY_KEY_COPY);
    thrust::pair<OutputIterator1,OutputIterator2> ret = thrust::unique_by_key_copy(exec.policy(get_system(keys_first,values_first,keys_result,values_result)),
                                                        keys_first, keys_last, values_first, keys_result,
                                                        values_result, binary_pred);
    exec.stop();

    return ret;
}

