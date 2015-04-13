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
#include <thrust/inner_product.h>

namespace grapple
{

template<typename InputIterator1,
         typename InputIterator2,
         typename OutputType>
OutputType inner_product(grapple_system &exec,
                         InputIterator1 first1,
                         InputIterator1 last1,
                         InputIterator2 first2,
                         OutputType init)
{
    exec.start(THRUST_INNER_PRODUCT);
    OutputType ret = thrust::inner_product(exec.policy(get_system(first1,first2)), first1, last1, first2, init);
    exec.stop();

    return ret;
}

template<typename InputIterator1,
         typename InputIterator2,
         typename OutputType,
         typename BinaryFunction1,
         typename BinaryFunction2>
OutputType inner_product(grapple_system &exec,
                         InputIterator1 first1,
                         InputIterator1 last1,
                         InputIterator2 first2,
                         OutputType init,
                         BinaryFunction1 binary_op1,
                         BinaryFunction2 binary_op2)
{
    exec.start(THRUST_INNER_PRODUCT);
    OutputType ret = thrust::inner_product(exec.policy(get_system(first1,first2)), first1, last1, first2, init, binary_op1, binary_op2);
    exec.stop();

    return ret;
}

}

