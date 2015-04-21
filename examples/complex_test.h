#pragma once

#include <thrust/execution_policy.h>

template<typename DerivedPolicy, typename Array>
void thrust_example_1(thrust::execution_policy<DerivedPolicy>& exec, Array& keys);

template<typename DerivedPolicy, typename Array>
void thrust_example_1(const thrust::detail::execution_policy_base<DerivedPolicy>& exec, Array& keys);

template<typename DerivedPolicy, typename Array>
void thrust_example_2(thrust::execution_policy<DerivedPolicy>& exec, Array& keys);

template<typename DerivedPolicy, typename Array>
void thrust_example_2(const thrust::detail::execution_policy_base<DerivedPolicy>& exec, Array& keys);

template<typename DerivedPolicy, typename Array>
void thrust_example_3(const thrust::detail::execution_policy_base<DerivedPolicy>& exec, Array& keys);

template<typename Array> void thrust_example_1(Array& keys);
template<typename Array> void thrust_example_2(Array& keys);
template<typename Array> void thrust_example_3(Array& keys);

#include "complex_test.inl"
