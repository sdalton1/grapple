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
#include <thrust/find.h>

template<typename InputIterator, typename T>
InputIterator find(grapple_system &exec,
                   InputIterator first,
                   InputIterator last,
                   const T& value)
{
    exec.start(THRUST_FIND);
    InputIterator ret = thrust::find(exec.policy(get_system(first)), first, last, value);
    exec.stop();

    return ret;
}

template<typename InputIterator, typename Predicate>
InputIterator find_if(grapple_system &exec,
                      InputIterator first,
                      InputIterator last,
                      Predicate pred)
{
    exec.start(THRUST_FIND_IF);
    InputIterator ret = thrust::find_if(exec.policy(get_system(first)), first, last, pred);
    exec.stop();

    return ret;
}

template<typename InputIterator, typename Predicate>
InputIterator find_if_not(grapple_system &exec,
                          InputIterator first,
                          InputIterator last,
                          Predicate pred)
{
    exec.start(THRUST_FIND_IF_NOT);
    InputIterator ret = thrust::find_if_not(exec.policy(get_system(first)), first, last, pred);
    exec.stop();

    return ret;
}

