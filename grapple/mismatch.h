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
#include <thrust/mismatch.h>

template<typename InputIterator1, typename InputIterator2>
thrust::pair<InputIterator1, InputIterator2> mismatch(grapple_system &exec,
                                                      InputIterator1 first1,
                                                      InputIterator1 last1,
                                                      InputIterator2 first2)
{
    exec.start(THRUST_MISMATCH);
    thrust::pair<InputIterator1, InputIterator2> ret = thrust::mismatch(thrust::cuda::par(exec), first1, last1, first2);
    exec.stop();

    return ret;
}

template<typename InputIterator1, typename InputIterator2, typename BinaryPredicate>
thrust::pair<InputIterator1, InputIterator2> mismatch(grapple_system &exec,
                                                      InputIterator1 first1,
                                                      InputIterator1 last1,
                                                      InputIterator2 first2,
                                                      BinaryPredicate pred)
{
    exec.start(THRUST_MISMATCH);
    thrust::pair<InputIterator1, InputIterator2> ret = thrust::mismatch(thrust::cuda::par(exec), first1, last1, first2, pred);
    exec.stop();

    return ret;
}

