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


/*! \file sort.h
*  \brief Functions for reorganizing ranges into sorted order
*/

#pragma once

#include <thrust/detail/config.h>
#include <thrust/sort.h>

namespace grapple
{

template<typename RandomAccessIterator>
void sort(grapple_system &exec,
          RandomAccessIterator first,
          RandomAccessIterator last)
{
    exec.start(THRUST_SORT);
    thrust::sort(exec.policy(get_system(first)), first, last);
    exec.stop();
}

template<typename RandomAccessIterator,
         typename StrictWeakOrdering>
void sort(grapple_system &exec,
          RandomAccessIterator first,
          RandomAccessIterator last,
          StrictWeakOrdering comp)
{
    exec.start(THRUST_SORT);
    thrust::sort(exec.policy(get_system(first)), first, last, comp);
    exec.stop();
}

template<typename RandomAccessIterator>
void stable_sort(grapple_system &exec,
                 RandomAccessIterator first,
                 RandomAccessIterator last)
{
    exec.start(THRUST_STABLE_SORT);
    thrust::stable_sort(exec.policy(get_system(first)), first, last);
    exec.stop();
}

template<typename RandomAccessIterator,
         typename StrictWeakOrdering>
void stable_sort(grapple_system &exec,
                 RandomAccessIterator first,
                 RandomAccessIterator last,
                 StrictWeakOrdering comp)
{
    exec.start(THRUST_STABLE_SORT);
    thrust::stable_sort(exec.policy(get_system(first)), first, last, comp);
    exec.stop();
}

///////////////
// Key Value //
///////////////


template<typename RandomAccessIterator1,
         typename RandomAccessIterator2>
void sort_by_key(grapple_system &exec,
                 RandomAccessIterator1 keys_first,
                 RandomAccessIterator1 keys_last,
                 RandomAccessIterator2 values_first)
{
    exec.start(THRUST_SORT_BY_KEY);
    thrust::sort_by_key(exec.policy(get_system(keys_first,values_first)), keys_first, keys_last, values_first);
    exec.stop();
}

template<typename RandomAccessIterator1,
         typename RandomAccessIterator2,
         typename StrictWeakOrdering>
void sort_by_key(grapple_system &exec,
                 RandomAccessIterator1 keys_first,
                 RandomAccessIterator1 keys_last,
                 RandomAccessIterator2 values_first,
                 StrictWeakOrdering comp)
{
    exec.start(THRUST_SORT_BY_KEY);
    thrust::sort_by_key(exec.policy(get_system(keys_first,values_first)), keys_first, keys_last, values_first, comp);
    exec.stop();
}

template<typename RandomAccessIterator1,
         typename RandomAccessIterator2>
void stable_sort_by_key(grapple_system &exec,
                        RandomAccessIterator1 keys_first,
                        RandomAccessIterator1 keys_last,
                        RandomAccessIterator2 values_first)
{
    exec.start(THRUST_STABLE_SORT_BY_KEY);
    thrust::stable_sort_by_key(exec.policy(get_system(keys_first,values_first)), keys_first, keys_last, values_first);
    exec.stop();
}

template<typename RandomAccessIterator1,
         typename RandomAccessIterator2,
         typename StrictWeakOrdering>
void stable_sort_by_key(grapple_system &exec,
                        RandomAccessIterator1 keys_first,
                        RandomAccessIterator1 keys_last,
                        RandomAccessIterator2 values_first,
                        StrictWeakOrdering comp)
{
    exec.start(THRUST_STABLE_SORT_BY_KEY);
    thrust::stable_sort_by_key(exec.policy(get_system(keys_first,values_first)), keys_first, keys_last, values_first, comp);
    exec.stop();
}

template<typename ForwardIterator>
bool is_sorted(grapple_system &exec,
               ForwardIterator first,
               ForwardIterator last)
{
    exec.start(THRUST_IS_SORTED);
    bool ret = thrust::is_sorted(exec.policy(get_system(first)), first, last);
    exec.stop();

    return ret;
}

template<typename ForwardIterator, typename Compare>
bool is_sorted(grapple_system &exec,
               ForwardIterator first,
               ForwardIterator last,
               Compare comp)
{
    exec.start(THRUST_IS_SORTED);
    bool ret = thrust::is_sorted(exec.policy(get_system(first)), first, last);
    exec.stop();

    return ret;
}

template<typename ForwardIterator>
ForwardIterator is_sorted_until(grapple_system &exec,
                                ForwardIterator first,
                                ForwardIterator last)
{
    exec.start(THRUST_IS_SORTED_UNTIL);
    ForwardIterator ret = thrust::is_sorted_until(exec.policy(get_system(first)), first, last);
    exec.stop();

    return ret;
}

template<typename ForwardIterator, typename Compare>
ForwardIterator is_sorted_until(grapple_system &exec,
                                ForwardIterator first,
                                ForwardIterator last,
                                Compare comp)
{
    exec.start(THRUST_IS_SORTED_UNTIL);
    ForwardIterator ret = thrust::is_sorted_until(exec.policy(get_system(first)), first, last, comp);
    exec.stop();

    return ret;
}

}
