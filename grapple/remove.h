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
#include <thrust/remove.h>

namespace grapple
{

template<typename ForwardIterator,
         typename T>
ForwardIterator remove(grapple_system &exec,
                       ForwardIterator first,
                       ForwardIterator last,
                       const T &value)
{
    exec.start(THRUST_REMOVE);
    ForwardIterator ret = thrust::remove(exec.policy(get_system(first)), first, last, value);
    exec.stop();

    return ret;
}

template<typename InputIterator,
         typename OutputIterator,
         typename T>
OutputIterator remove_copy(grapple_system &exec,
                           InputIterator first,
                           InputIterator last,
                           OutputIterator result,
                           const T &value)
{
    exec.start(THRUST_REMOVE_COPY);
    OutputIterator ret = thrust::remove_copy(exec.policy(get_system(first,result)), first, last, result, value);
    exec.stop();

    return ret;
}

template<typename ForwardIterator,
         typename Predicate>
ForwardIterator remove_if(grapple_system &exec,
                          ForwardIterator first,
                          ForwardIterator last,
                          Predicate pred)
{
    exec.start(THRUST_REMOVE_IF);
    ForwardIterator ret = thrust::remove_if(exec.policy(get_system(first)), first, last, pred);
    exec.stop();

    return ret;
}

template<typename InputIterator,
         typename OutputIterator,
         typename Predicate>
OutputIterator remove_copy_if(grapple_system &exec,
                              InputIterator first,
                              InputIterator last,
                              OutputIterator result,
                              Predicate pred)
{
    exec.start(THRUST_REMOVE_COPY_IF);
    OutputIterator ret = thrust::remove_copy_if(exec.policy(get_system(first,result)), first, last, result, pred);
    exec.stop();

    return ret;
}

template<typename ForwardIterator,
         typename InputIterator,
         typename Predicate>
ForwardIterator remove_if(grapple_system &exec,
                          ForwardIterator first,
                          ForwardIterator last,
                          InputIterator stencil,
                          Predicate pred)
{
    exec.start(THRUST_REMOVE_IF);
    ForwardIterator ret = thrust::remove_if(exec.policy(get_system(first,stencil)), first, last, stencil, pred);
    exec.stop();

    return ret;
}

template<typename InputIterator1,
         typename InputIterator2,
         typename OutputIterator,
         typename Predicate>
OutputIterator remove_copy_if(grapple_system &exec,
                              InputIterator1 first,
                              InputIterator1 last,
                              InputIterator2 stencil,
                              OutputIterator result,
                              Predicate pred)
{
    exec.start(THRUST_REMOVE_COPY_IF);
    OutputIterator ret = thrust::remove_copy_if(exec.policy(get_system(first,stencil,result)), first, last, stencil, result, pred);
    exec.stop();

    return ret;
}

}

