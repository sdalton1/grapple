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
#include <thrust/scan.h>

template<typename InputIterator,
         typename OutputIterator>
OutputIterator inclusive_scan(grapple_system &exec,
                              InputIterator first,
                              InputIterator last,
                              OutputIterator result)
{
    exec.start(thrust_mapper::thrustMap.find(__FUNCTION__)->second);
    OutputIterator ret = thrust::inclusive_scan(exec.policy(), first, last, result);
    exec.stop();

    return ret;
}

template<typename InputIterator,
         typename OutputIterator,
         typename AssociativeOperator>
OutputIterator inclusive_scan(grapple_system &exec,
                              InputIterator first,
                              InputIterator last,
                              OutputIterator result,
                              AssociativeOperator binary_op)
{
    exec.start(thrust_mapper::thrustMap.find(__FUNCTION__)->second);
    OutputIterator ret = thrust::exclusive_scan(exec.policy(), first, last, result, binary_op);
    exec.stop();

    return ret;
}

template<typename InputIterator,
         typename OutputIterator>
OutputIterator exclusive_scan(grapple_system &exec,
                              InputIterator first,
                              InputIterator last,
                              OutputIterator result)
{
    exec.start(thrust_mapper::thrustMap.find(__FUNCTION__)->second);
    OutputIterator ret = thrust::exclusive_scan(exec.policy(), first, last, result);
    exec.stop();

    return ret;
}

template<typename InputIterator,
         typename OutputIterator,
         typename T>
OutputIterator exclusive_scan(grapple_system &exec,
                              InputIterator first,
                              InputIterator last,
                              OutputIterator result,
                              T init)
{
    exec.start(thrust_mapper::thrustMap.find(__FUNCTION__)->second);
    OutputIterator ret = thrust::exclusive_scan(exec.policy(), first, last, result, init);
    exec.stop();

    return ret;
}

template<typename InputIterator,
         typename OutputIterator,
         typename T,
         typename AssociativeOperator>
OutputIterator exclusive_scan(grapple_system &exec,
                              InputIterator first,
                              InputIterator last,
                              OutputIterator result,
                              T init,
                              AssociativeOperator binary_op)
{
    exec.start(thrust_mapper::thrustMap.find(__FUNCTION__)->second);
    OutputIterator ret = thrust::exclusive_scan(exec.policy(), first, last, result, init, binary_op);
    exec.stop();

    return ret;
}

template<typename InputIterator1,
         typename InputIterator2,
         typename OutputIterator>
OutputIterator inclusive_scan_by_key(grapple_system &exec,
                                     InputIterator1 first1,
                                     InputIterator1 last1,
                                     InputIterator2 first2,
                                     OutputIterator result)
{
    exec.start(thrust_mapper::thrustMap.find(__FUNCTION__)->second);
    OutputIterator ret = thrust::inclusive_scan_by_key(exec.policy(), first1, last1, first2, result);
    exec.stop();

    return ret;
}

template<typename InputIterator1,
         typename InputIterator2,
         typename OutputIterator,
         typename BinaryPredicate>
OutputIterator inclusive_scan_by_key(grapple_system &exec,
                                     InputIterator1 first1,
                                     InputIterator1 last1,
                                     InputIterator2 first2,
                                     OutputIterator result,
                                     BinaryPredicate binary_pred)
{
    exec.start(thrust_mapper::thrustMap.find(__FUNCTION__)->second);
    OutputIterator ret = thrust::inclusive_scan_by_key(exec.policy(), first1, last1, first2, result, binary_pred);
    exec.stop();

    return ret;
}

template<typename InputIterator1,
         typename InputIterator2,
         typename OutputIterator,
         typename BinaryPredicate,
         typename AssociativeOperator>
OutputIterator inclusive_scan_by_key(grapple_system &exec,
                                     InputIterator1 first1,
                                     InputIterator1 last1,
                                     InputIterator2 first2,
                                     OutputIterator result,
                                     BinaryPredicate binary_pred,
                                     AssociativeOperator binary_op)
{
    exec.start(thrust_mapper::thrustMap.find(__FUNCTION__)->second);
    OutputIterator ret = thrust::inclusive_scan_by_key(exec.policy(), first1, last1, first2, result, binary_pred, binary_op);
    exec.stop();

    return ret;
}

template<typename InputIterator1,
         typename InputIterator2,
         typename OutputIterator>
OutputIterator exclusive_scan_by_key(grapple_system &exec,
                                     InputIterator1 first1,
                                     InputIterator1 last1,
                                     InputIterator2 first2,
                                     OutputIterator result)
{
    exec.start(thrust_mapper::thrustMap.find(__FUNCTION__)->second);
    OutputIterator ret = thrust::exclusive_scan_by_key(exec.policy(), first1, last1, first2, result);
    exec.stop();

    return ret;
}

template<typename InputIterator1,
         typename InputIterator2,
         typename OutputIterator,
         typename T>
OutputIterator exclusive_scan_by_key(grapple_system &exec,
                                     InputIterator1 first1,
                                     InputIterator1 last1,
                                     InputIterator2 first2,
                                     OutputIterator result,
                                     T init)
{
    exec.start(thrust_mapper::thrustMap.find(__FUNCTION__)->second);
    OutputIterator ret = thrust::exclusive_scan_by_key(exec.policy(), first1, last1, first2, result, init);
    exec.stop();

    return ret;
}

template<typename InputIterator1,
         typename InputIterator2,
         typename OutputIterator,
         typename T,
         typename BinaryPredicate>
OutputIterator exclusive_scan_by_key(grapple_system &exec,
                                     InputIterator1 first1,
                                     InputIterator1 last1,
                                     InputIterator2 first2,
                                     OutputIterator result,
                                     T init,
                                     BinaryPredicate binary_pred)
{
    exec.start(thrust_mapper::thrustMap.find(__FUNCTION__)->second);
    OutputIterator ret = thrust::exclusive_scan_by_key(exec.policy(), first1, last1, first2, result, init, binary_pred);
    exec.stop();

    return ret;
}

template<typename InputIterator1,
         typename InputIterator2,
         typename OutputIterator,
         typename T,
         typename BinaryPredicate,
         typename AssociativeOperator>
OutputIterator exclusive_scan_by_key(grapple_system &exec,
                                     InputIterator1 first1,
                                     InputIterator1 last1,
                                     InputIterator2 first2,
                                     OutputIterator result,
                                     T init,
                                     BinaryPredicate binary_pred,
                                     AssociativeOperator binary_op)
{
    exec.start(thrust_mapper::thrustMap.find(__FUNCTION__)->second);
    OutputIterator ret = thrust::exclusive_scan_by_key(exec.policy(), first1, last1, first2, result, init, binary_pred, binary_op);
    exec.stop();

    return ret;
}

