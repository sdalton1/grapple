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
#include <thrust/set_operations.h>

template<typename InputIterator1,
         typename InputIterator2,
         typename OutputIterator>
OutputIterator set_difference(grapple_system &exec,
                              InputIterator1                                              first1,
                              InputIterator1                                              last1,
                              InputIterator2                                              first2,
                              InputIterator2                                              last2,
                              OutputIterator                                              result)
{
    exec.start(THRUST_SET_DIFFERENCE);
    OutputIterator ret = thrust::set_difference(exec.policy(get_system(first1,first2,result)), first1, last1, first2, last2, result);
    exec.stop();

    return ret;
}

template<typename InputIterator1,
         typename InputIterator2,
         typename OutputIterator,
         typename StrictWeakCompare>
OutputIterator set_difference(grapple_system &exec,
                              InputIterator1                                              first1,
                              InputIterator1                                              last1,
                              InputIterator2                                              first2,
                              InputIterator2                                              last2,
                              OutputIterator                                              result,
                              StrictWeakCompare                                           comp)
{
    exec.start(THRUST_SET_DIFFERENCE);
    OutputIterator ret = thrust::set_difference(exec.policy(get_system(first1,first2,result)), first1, last1, first2, last2, result, comp);
    exec.stop();

    return ret;
}

template<typename InputIterator1,
         typename InputIterator2,
         typename OutputIterator>
OutputIterator set_intersection(grapple_system &exec,
                                InputIterator1                                              first1,
                                InputIterator1                                              last1,
                                InputIterator2                                              first2,
                                InputIterator2                                              last2,
                                OutputIterator                                              result)
{
    exec.start(THRUST_SET_INTERSECTION);
    OutputIterator ret = thrust::set_intersection(exec.policy(get_system(first1,first2,result)), first1, last1, first2, last2, result);
    exec.stop();

    return ret;
}

template<typename InputIterator1,
         typename InputIterator2,
         typename OutputIterator,
         typename StrictWeakCompare>
OutputIterator set_intersection(grapple_system &exec,
                                InputIterator1                                              first1,
                                InputIterator1                                              last1,
                                InputIterator2                                              first2,
                                InputIterator2                                              last2,
                                OutputIterator                                              result,
                                StrictWeakCompare                                           comp)
{
    exec.start(THRUST_SET_INTERSECTION);
    OutputIterator ret = thrust::set_intersection(exec.policy(get_system(first1,first2,result)), first1, last1, first2, last2, result, comp);
    exec.stop();

    return ret;
}

template<typename InputIterator1,
         typename InputIterator2,
         typename OutputIterator>
OutputIterator set_symmetric_difference(grapple_system &exec,
                                        InputIterator1                                              first1,
                                        InputIterator1                                              last1,
                                        InputIterator2                                              first2,
                                        InputIterator2                                              last2,
                                        OutputIterator                                              result)
{
    exec.start(THRUST_SET_SYMMETRIC_DIFFERENCE);
    OutputIterator ret = thrust::set_symmetric_difference(exec.policy(get_system(first1,first2,result)), first1, last1, first2, last2, result);
    exec.stop();

    return ret;
}

template<typename InputIterator1,
         typename InputIterator2,
         typename OutputIterator,
         typename StrictWeakCompare>
OutputIterator set_symmetric_difference(grapple_system &exec,
                                        InputIterator1                                              first1,
                                        InputIterator1                                              last1,
                                        InputIterator2                                              first2,
                                        InputIterator2                                              last2,
                                        OutputIterator                                              result,
                                        StrictWeakCompare                                           comp)
{
    exec.start(THRUST_SET_SYMMETRIC_DIFFERENCE);
    OutputIterator ret = thrust::set_symmetric_difference(exec.policy(get_system(first1,first2,result)), first1, last1, first2, last2, result, comp);
    exec.stop();

    return ret;
}

template<typename InputIterator1,
         typename InputIterator2,
         typename OutputIterator>
OutputIterator set_union(grapple_system &exec,
                         InputIterator1                                              first1,
                         InputIterator1                                              last1,
                         InputIterator2                                              first2,
                         InputIterator2                                              last2,
                         OutputIterator                                              result)
{
    exec.start(THRUST_SET_UNION);
    OutputIterator ret = thrust::set_union(exec.policy(get_system(first1,first2,result)), first1, last1, first2, last2, result);
    exec.stop();

    return ret;
}

template<typename InputIterator1,
         typename InputIterator2,
         typename OutputIterator,
         typename StrictWeakCompare>
OutputIterator set_union(grapple_system &exec,
                         InputIterator1                                              first1,
                         InputIterator1                                              last1,
                         InputIterator2                                              first2,
                         InputIterator2                                              last2,
                         OutputIterator                                              result,
                         StrictWeakCompare                                           comp)
{
    exec.start(THRUST_SET_UNION);
    OutputIterator ret = thrust::set_union(exec.policy(get_system(first1,first2,result)), first1, last1, first2, last2, result, comp);
    exec.stop();

    return ret;
}

template<typename InputIterator1,
         typename InputIterator2,
         typename InputIterator3,
         typename InputIterator4,
         typename OutputIterator1,
         typename OutputIterator2>
thrust::pair<OutputIterator1,OutputIterator2>
set_difference_by_key(grapple_system &exec,
                      InputIterator1                                              keys_first1,
                      InputIterator1                                              keys_last1,
                      InputIterator2                                              keys_first2,
                      InputIterator2                                              keys_last2,
                      InputIterator3                                              values_first1,
                      InputIterator4                                              values_first2,
                      OutputIterator1                                             keys_result,
                      OutputIterator2                                             values_result)
{
    exec.start(THRUST_SET_DIFFERENCE_BY_KEY);
    thrust::pair<OutputIterator1,OutputIterator2> ret = thrust::set_difference_by_key(exec.policy(get_system(keys_first1,keys_first2,values_first1,values_first2,keys_result,values_result)),
                                    keys_first1, keys_last1, keys_first2, keys_last2,
                                    values_first1, values_first2, keys_result, values_result);
    exec.stop();

    return ret;
}


template<typename InputIterator1,
         typename InputIterator2,
         typename InputIterator3,
         typename InputIterator4,
         typename OutputIterator1,
         typename OutputIterator2,
         typename StrictWeakCompare>
thrust::pair<OutputIterator1,OutputIterator2>
set_difference_by_key(grapple_system &exec,
                      InputIterator1                                              keys_first1,
                      InputIterator1                                              keys_last1,
                      InputIterator2                                              keys_first2,
                      InputIterator2                                              keys_last2,
                      InputIterator3                                              values_first1,
                      InputIterator4                                              values_first2,
                      OutputIterator1                                             keys_result,
                      OutputIterator2                                             values_result,
                      StrictWeakCompare                                           comp)
{
    exec.start(THRUST_SET_DIFFERENCE_BY_KEY);
    thrust::pair<OutputIterator1,OutputIterator2> ret = thrust::set_difference_by_key(exec.policy(get_system(keys_first1,keys_first2,values_first1,values_first2,keys_result,values_result)),
                                    keys_first1, keys_last1, keys_first2, keys_last2,
                                    values_first1, values_first2, keys_result, values_result, comp);
    exec.stop();

    return ret;
}


template<typename InputIterator1,
         typename InputIterator2,
         typename InputIterator3,
         typename OutputIterator1,
         typename OutputIterator2>
thrust::pair<OutputIterator1,OutputIterator2>
set_intersection_by_key(grapple_system &exec,
                        InputIterator1                                              keys_first1,
                        InputIterator1                                              keys_last1,
                        InputIterator2                                              keys_first2,
                        InputIterator2                                              keys_last2,
                        InputIterator3                                              values_first1,
                        OutputIterator1                                             keys_result,
                        OutputIterator2                                             values_result)
{
    exec.start(THRUST_SET_INTERSECTION_BY_KEY);
    thrust::pair<OutputIterator1,OutputIterator2> ret = thrust::set_intersection_by_key(exec.policy(get_system(keys_first1,keys_first2,values_first1,keys_result,values_result)),
                                    keys_first1, keys_last1, keys_first2, keys_last2,
                                    values_first1, keys_result, values_result);
    exec.stop();

    return ret;
}


template<typename InputIterator1,
         typename InputIterator2,
         typename InputIterator3,
         typename OutputIterator1,
         typename OutputIterator2,
         typename StrictWeakCompare>
thrust::pair<OutputIterator1,OutputIterator2>
set_intersection_by_key(grapple_system &exec,
                        InputIterator1                                              keys_first1,
                        InputIterator1                                              keys_last1,
                        InputIterator2                                              keys_first2,
                        InputIterator2                                              keys_last2,
                        InputIterator3                                              values_first1,
                        OutputIterator1                                             keys_result,
                        OutputIterator2                                             values_result,
                        StrictWeakCompare                                           comp)
{
    exec.start(THRUST_SET_INTERSECTION_BY_KEY);
    thrust::pair<OutputIterator1,OutputIterator2> ret = thrust::set_intersection_by_key(exec.policy(get_system(keys_first1,keys_first2,values_first1,keys_result,values_result)),
                                    keys_first1, keys_last1, keys_first2, keys_last2,
                                    values_first1, keys_result, values_result, comp);
    exec.stop();

    return ret;
}


template<typename InputIterator1,
         typename InputIterator2,
         typename InputIterator3,
         typename InputIterator4,
         typename OutputIterator1,
         typename OutputIterator2>
thrust::pair<OutputIterator1,OutputIterator2>
set_symmetric_difference_by_key(grapple_system &exec,
                                InputIterator1                                              keys_first1,
                                InputIterator1                                              keys_last1,
                                InputIterator2                                              keys_first2,
                                InputIterator2                                              keys_last2,
                                InputIterator3                                              values_first1,
                                InputIterator4                                              values_first2,
                                OutputIterator1                                             keys_result,
                                OutputIterator2                                             values_result)
{
    exec.start(THRUST_SET_SYMMETRIC_DIFFERENCE_BY_KEY);
    thrust::pair<OutputIterator1,OutputIterator2> ret = thrust::set_symmetric_difference_by_key(exec.policy(get_system(keys_first1,keys_first2,values_first1,values_first2,keys_result,values_result)),
                                    keys_first1, keys_last1, keys_first2, keys_last2,
                                    values_first1, values_first2, keys_result, values_result);
    exec.stop();

    return ret;
}

template<typename InputIterator1,
         typename InputIterator2,
         typename InputIterator3,
         typename InputIterator4,
         typename OutputIterator1,
         typename OutputIterator2,
         typename StrictWeakCompare>
thrust::pair<OutputIterator1,OutputIterator2>
set_symmetric_difference_by_key(grapple_system &exec,
                                InputIterator1                                              keys_first1,
                                InputIterator1                                              keys_last1,
                                InputIterator2                                              keys_first2,
                                InputIterator2                                              keys_last2,
                                InputIterator3                                              values_first1,
                                InputIterator4                                              values_first2,
                                OutputIterator1                                             keys_result,
                                OutputIterator2                                             values_result,
                                StrictWeakCompare                                           comp)
{
    exec.start(THRUST_SET_SYMMETRIC_DIFFERENCE_BY_KEY);
    thrust::pair<OutputIterator1,OutputIterator2> ret = thrust::set_symmetric_difference_by_key(exec.policy(get_system(keys_first1,keys_first2,values_first1,values_first2,keys_result,values_result)),
                                    keys_first1, keys_last1, keys_first2, keys_last2,
                                    values_first1, values_first2, keys_result, values_result, comp);
    exec.stop();

    return ret;
}

template<typename InputIterator1,
         typename InputIterator2,
         typename InputIterator3,
         typename InputIterator4,
         typename OutputIterator1,
         typename OutputIterator2>
thrust::pair<OutputIterator1,OutputIterator2>
set_union_by_key(grapple_system &exec,
                 InputIterator1                                              keys_first1,
                 InputIterator1                                              keys_last1,
                 InputIterator2                                              keys_first2,
                 InputIterator2                                              keys_last2,
                 InputIterator3                                              values_first1,
                 InputIterator4                                              values_first2,
                 OutputIterator1                                             keys_result,
                 OutputIterator2                                             values_result)
{
    exec.start(THRUST_SET_UNION_BY_KEY);
    thrust::pair<OutputIterator1,OutputIterator2> ret = thrust::set_union_by_key(exec.policy(get_system(keys_first1,keys_first2,values_first1,values_first2,keys_result,values_result)),
                                    keys_first1, keys_last1, keys_first2, keys_last2,
                                    values_first1, values_first2, keys_result, values_result);
    exec.stop();

    return ret;
}


template<typename InputIterator1,
         typename InputIterator2,
         typename InputIterator3,
         typename InputIterator4,
         typename OutputIterator1,
         typename OutputIterator2,
         typename StrictWeakCompare>
thrust::pair<OutputIterator1,OutputIterator2>
set_union_by_key(grapple_system &exec,
                 InputIterator1                                              keys_first1,
                 InputIterator1                                              keys_last1,
                 InputIterator2                                              keys_first2,
                 InputIterator2                                              keys_last2,
                 InputIterator3                                              values_first1,
                 InputIterator4                                              values_first2,
                 OutputIterator1                                             keys_result,
                 OutputIterator2                                             values_result,
                 StrictWeakCompare                                           comp)
{
    exec.start(THRUST_SET_UNION_BY_KEY);
    thrust::pair<OutputIterator1,OutputIterator2> ret = thrust::set_union_by_key(exec.policy(get_system(keys_first1,keys_first2,values_first1,values_first2,keys_result,values_result)),
                                    keys_first1, keys_last1, keys_first2, keys_last2,
                                    values_first1, values_first2, keys_result, values_result, comp);
    exec.stop();

    return ret;
}

