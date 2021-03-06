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
#include <thrust/scatter.h>

namespace grapple
{

template<typename InputIterator1,
         typename InputIterator2,
         typename RandomAccessIterator>
void scatter(grapple_system &exec,
             InputIterator1 first,
             InputIterator1 last,
             InputIterator2 map,
             RandomAccessIterator result)
{
    exec.start(THRUST_SCATTER);
    thrust::scatter(exec.policy(get_system(first,map,result)), first, last, map, result);
    exec.stop();
}

template<typename InputIterator1,
         typename InputIterator2,
         typename InputIterator3,
         typename RandomAccessIterator>
void scatter_if(grapple_system &exec,
                InputIterator1 first,
                InputIterator1 last,
                InputIterator2 map,
                InputIterator3 stencil,
                RandomAccessIterator output)
{
    exec.start(THRUST_SCATTER_IF);
    thrust::scatter_if(exec.policy(get_system(first,map,stencil,output)), first, last, map, stencil, output);
    exec.stop();
}

template<typename InputIterator1,
         typename InputIterator2,
         typename InputIterator3,
         typename RandomAccessIterator,
         typename Predicate>
void scatter_if(grapple_system &exec,
                InputIterator1 first,
                InputIterator1 last,
                InputIterator2 map,
                InputIterator3 stencil,
                RandomAccessIterator output,
                Predicate pred)
{
    exec.start(THRUST_SCATTER_IF);
    thrust::scatter_if(exec.policy(get_system(first,map,stencil,output)), first, last, map, stencil, output, pred);
    exec.stop();
}

}

