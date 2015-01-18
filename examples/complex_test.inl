#include <thrust/adjacent_difference.h>
#include <thrust/device_vector.h>
#include <thrust/random.h>
#include <thrust/reduce.h>
#include <thrust/sort.h>
#include <thrust/unique.h>

// base file
void initialize(thrust::device_vector<float>& v)
{
    thrust::default_random_engine rng(123456);
    thrust::uniform_int_distribution<int> dist(2, 19);
    for(size_t i = 0; i < v.size(); i++)
        v[i] = dist(rng) / 2.0f;
}

template<typename DerivedPolicy, typename Array>
void thrust_example_1(const thrust::execution_policy<DerivedPolicy>& exec, Array& keys)
{
    Array values(keys.size());
    initialize(keys);

    thrust::sort(exec, keys.begin(), keys.end());
    thrust::reduce(exec, keys.begin(), keys.end());
    thrust::adjacent_difference(exec, keys.begin(), keys.end(), values.begin());
}

template<typename DerivedPolicy, typename Array>
void thrust_example_2(const thrust::execution_policy<DerivedPolicy>& exec, Array& keys)
{
    initialize(keys);

    thrust::sort(exec, keys.begin(), keys.end());
    thrust::unique(exec, keys.begin(), keys.end());
}

template<typename DerivedPolicy, typename Array>
void thrust_example_3(const thrust::execution_policy<DerivedPolicy>& exec, Array& keys)
{
    DerivedPolicy& derived(thrust::detail::derived_cast(thrust::detail::strip_const(exec)));

    thrust_example_1(derived, keys);
    thrust_example_2(derived, keys);
}

template<typename Array>
void thrust_example_1(Array& keys)
{
  using thrust::system::detail::generic::select_system;

  typename thrust::iterator_system<typename Array::iterator>::type system;

  thrust_example_1(select_system(system), keys);
}

template<typename Array>
void thrust_example_2(Array& keys)
{
  using thrust::system::detail::generic::select_system;

  typename thrust::iterator_system<typename Array::iterator>::type system;

  thrust_example_2(select_system(system), keys);
}

template<typename Array>
void thrust_example_3(Array& keys)
{
  using thrust::system::detail::generic::select_system;

  typename thrust::iterator_system<typename Array::iterator>::type system;

  thrust_example_3(select_system(system), keys);
}

