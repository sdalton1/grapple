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
#include <thrust/partition.h>

template<typename ForwardIterator,
         typename Predicate>
ForwardIterator partition(grapple_system &exec,
                          ForwardIterator first,
                          ForwardIterator last,
                          Predicate pred)
{
    exec.start(THRUST_PARTITION);
    ForwardIterator ret = thrust::partition(thrust::cuda::par(exec), first, last, pred);
    exec.stop();

    return ret;
}

template<typename ForwardIterator,
         typename InputIterator,
         typename Predicate>
ForwardIterator partition(grapple_system &exec,
                          ForwardIterator first,
                          ForwardIterator last,
                          InputIterator stencil,
                          Predicate pred)
{
    exec.start(THRUST_PARTITION);
    ForwardIterator ret = thrust::partition(thrust::cuda::par(exec), first, last, stencil, pred);
    exec.stop();

    return ret;
}

template<typename InputIterator,
         typename OutputIterator1,
         typename OutputIterator2,
         typename Predicate>
thrust::pair<OutputIterator1,OutputIterator2>
partition_copy(grapple_system &exec,
               InputIterator first,
               InputIterator last,
               OutputIterator1 out_true,
               OutputIterator2 out_false,
               Predicate pred)
{
    exec.start(THRUST_PARTITION_COPY);
    thrust::pair<OutputIterator1,OutputIterator2> ret = thrust::partition_copy(thrust::cuda::par(exec), first, last, out_true, out_false, pred);
    exec.stop();

    return ret;
}

template<typename InputIterator1,
         typename InputIterator2,
         typename OutputIterator1,
         typename OutputIterator2,
         typename Predicate>
thrust::pair<OutputIterator1,OutputIterator2>
partition_copy(grapple_system &exec,
               InputIterator1 first,
               InputIterator1 last,
               InputIterator2 stencil,
               OutputIterator1 out_true,
               OutputIterator2 out_false,
               Predicate pred)
{
    exec.start(THRUST_PARTITION_COPY);
    thrust::pair<OutputIterator1,OutputIterator2> ret = thrust::partition_copy(thrust::cuda::par(exec), first, last, stencil, out_true, out_false, pred);
    exec.stop();

    return ret;
}

template<typename ForwardIterator,
         typename Predicate>
ForwardIterator stable_partition(grapple_system &exec,
                                 ForwardIterator first,
                                 ForwardIterator last,
                                 Predicate pred)
{
    exec.start(THRUST_STABLE_PARTITION);
    ForwardIterator ret = thrust::stable_partition(thrust::cuda::par(exec), first, last, pred);
    exec.stop();

    return ret;
}

template<typename ForwardIterator,
         typename InputIterator,
         typename Predicate>
ForwardIterator stable_partition(grapple_system &exec,
                                 ForwardIterator first,
                                 ForwardIterator last,
                                 InputIterator stencil,
                                 Predicate pred)
{
    exec.start(THRUST_STABLE_PARTITION);
    ForwardIterator ret = thrust::stable_partition(thrust::cuda::par(exec), first, last, stencil, pred);
    exec.stop();

    return ret;
}

template<typename InputIterator,
         typename OutputIterator1,
         typename OutputIterator2,
         typename Predicate>
thrust::pair<OutputIterator1,OutputIterator2>
stable_partition_copy(grapple_system &exec,
                      InputIterator first,
                      InputIterator last,
                      OutputIterator1 out_true,
                      OutputIterator2 out_false,
                      Predicate pred)
{
    exec.start(THRUST_STABLE_PARTITION_COPY);
    thrust::pair<OutputIterator1,OutputIterator2> ret = thrust::stable_partition_copy(thrust::cuda::par(exec), first, last, out_true, out_false, pred);
    exec.stop();

    return ret;
}

template<typename InputIterator1,
         typename InputIterator2,
         typename OutputIterator1,
         typename OutputIterator2,
         typename Predicate>
thrust::pair<OutputIterator1,OutputIterator2>
stable_partition_copy(grapple_system &exec,
                      InputIterator1 first,
                      InputIterator1 last,
                      InputIterator2 stencil,
                      OutputIterator1 out_true,
                      OutputIterator2 out_false,
                      Predicate pred)
{
    exec.start(THRUST_STABLE_PARTITION_COPY);
    thrust::pair<OutputIterator1,OutputIterator2> ret = thrust::stable_partition_copy(thrust::cuda::par(exec), first, last, stencil, out_true, out_false, pred);
    exec.stop();

    return ret;
}

template<typename ForwardIterator, typename Predicate>
ForwardIterator partition_point(grapple_system &exec,
                                ForwardIterator first,
                                ForwardIterator last,
                                Predicate pred)
{
    exec.start(THRUST_PARTITION_POINT);
    ForwardIterator ret = thrust::partition_point(thrust::cuda::par(exec), first, last, pred);
    exec.stop();

    return ret;
}

template<typename InputIterator, typename Predicate>
bool is_partitioned(grapple_system &exec,
                    InputIterator first,
                    InputIterator last,
                    Predicate pred)
{
    exec.start(THRUST_IS_PARTITIONED);
    bool ret = thrust::is_partitioned(thrust::cuda::par(exec), first, last, pred);
    exec.stop();

    return ret;
}

