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
#include <thrust/transform.h>

template<typename InputIterator,
         typename OutputIterator,
         typename UnaryFunction>
OutputIterator transform(grapple_system &exec,
                         InputIterator first, InputIterator last,
                         OutputIterator result,
                         UnaryFunction op)
{
    exec.start(THRUST_TRANSFORM);
    OutputIterator ret = thrust::transform_if(thrust::cuda::par(exec), first, last, result, op);
    exec.stop();

    return ret;
}

template<typename InputIterator1,
         typename InputIterator2,
         typename OutputIterator,
         typename BinaryFunction>
OutputIterator transform(grapple_system &exec,
                         InputIterator1 first1, InputIterator1 last1,
                         InputIterator2 first2,
                         OutputIterator result,
                         BinaryFunction op)
{
    exec.start(THRUST_TRANSFORM);
    OutputIterator ret = thrust::transform_if(thrust::cuda::par(exec), first1, last1, first2, result, op);
    exec.stop();

    return ret;
}

template<typename InputIterator,
         typename ForwardIterator,
         typename UnaryFunction,
         typename Predicate>
ForwardIterator transform_if(grapple_system &exec,
                             InputIterator first, InputIterator last,
                             ForwardIterator result,
                             UnaryFunction op,
                             Predicate pred)
{
    exec.start(THRUST_TRANSFORM_IF);
    ForwardIterator ret = thrust::transform_if(thrust::cuda::par(exec), first, last, result, op, pred);
    exec.stop();

    return ret;
}

template<typename InputIterator1,
         typename InputIterator2,
         typename ForwardIterator,
         typename UnaryFunction,
         typename Predicate>
ForwardIterator transform_if(grapple_system &exec,
                             InputIterator1 first, InputIterator1 last,
                             InputIterator2 stencil,
                             ForwardIterator result,
                             UnaryFunction op,
                             Predicate pred)
{
    exec.start(THRUST_TRANSFORM_IF);
    ForwardIterator ret = thrust::transform_if(thrust::cuda::par(exec), first, last, stencil, result, op, pred);
    exec.stop();

    return ret;
}

template<typename InputIterator1,
         typename InputIterator2,
         typename InputIterator3,
         typename ForwardIterator,
         typename BinaryFunction,
         typename Predicate>
ForwardIterator transform_if(grapple_system &exec,
                             InputIterator1 first1, InputIterator1 last1,
                             InputIterator2 first2,
                             InputIterator3 stencil,
                             ForwardIterator result,
                             BinaryFunction binary_op,
                             Predicate pred)
{
    exec.start(THRUST_TRANSFORM_IF);
    ForwardIterator ret = thrust::transform_if(thrust::cuda::par(exec), first1, last1, first2, stencil, result, binary_op, pred);
    exec.stop();

    return ret;
}

