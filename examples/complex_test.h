#pragma once

#include <thrust/execution_policy.h>

template<typename Policy, typename Array>
void thrust_example_1(const Policy&, Array&);

template<typename Policy, typename Array>
void thrust_example_2(const Policy&, Array&);

template<typename DerivedPolicy, typename Array>
void thrust_example_3(const thrust::execution_policy<DerivedPolicy>&, Array&);

template<typename Array> void thrust_example_1(Array& keys);
template<typename Array> void thrust_example_2(Array& keys);
template<typename Array> void thrust_example_3(Array& keys);

#include "complex_test.inl"
