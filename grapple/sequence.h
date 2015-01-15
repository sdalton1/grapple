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
#include <thrust/sequence.h>

template<typename ForwardIterator>
void sequence(grapple_system &exec,
              ForwardIterator first,
              ForwardIterator last)
{
    exec.start(thrust_mapper::thrustMap.find(__FUNCTION__)->second);
    thrust::sequence(exec.policy(), first, last);
    exec.stop();
}

template<typename ForwardIterator, typename T>
void sequence(grapple_system &exec,
              ForwardIterator first,
              ForwardIterator last,
              T init)
{
    exec.start(thrust_mapper::thrustMap.find(__FUNCTION__)->second);
    thrust::sequence(exec.policy(), first, last, init);
    exec.stop();
}

template<typename ForwardIterator, typename T>
void sequence(grapple_system &exec,
              ForwardIterator first,
              ForwardIterator last,
              T init,
              T step)
{
    exec.start(thrust_mapper::thrustMap.find(__FUNCTION__)->second);
    thrust::sequence(exec.policy(), first, last, init, step);
    exec.stop();
}

