#include <thrust/adjacent_difference.h>
#include <thrust/device_vector.h>
#include <thrust/random.h>
#include <thrust/reduce.h>
#include <thrust/sort.h>
#include <thrust/unique.h>

namespace example
{
namespace detail
{

template<typename DerivedPolicy, typename Array>
void initialize(thrust::execution_policy<DerivedPolicy>& exec, Array& v)
{
    thrust::default_random_engine rng(123456);
    thrust::uniform_int_distribution<int> dist(2, 19);
    for(size_t i = 0; i < v.size(); i++)
        v[i] = dist(rng) / 2.0f;
}

template<typename DerivedPolicy, typename Array>
void thrust_example_1(thrust::execution_policy<DerivedPolicy>& exec, Array& keys)
{
    Array values(keys.size());
    example::initialize(thrust::detail::derived_cast(exec), keys);

    thrust::sort(exec, keys.begin(), keys.end());
    thrust::reduce(exec, keys.begin(), keys.end());
    thrust::adjacent_difference(exec, keys.begin(), keys.end(), values.begin());
}

template<typename DerivedPolicy, typename Array>
void thrust_example_2(thrust::execution_policy<DerivedPolicy>& exec, Array& keys)
{
    example::initialize(thrust::detail::derived_cast(exec), keys);

    thrust::sort(exec, keys.begin(), keys.end());
    thrust::unique(exec, keys.begin(), keys.end());
}

template<typename DerivedPolicy, typename Array>
void thrust_example_3(thrust::execution_policy<DerivedPolicy>& exec, Array& keys)
{
    example::thrust_example_1(thrust::detail::derived_cast(exec), keys);
    example::thrust_example_2(thrust::detail::derived_cast(exec), keys);
}

} // end namespace detail

template<typename DerivedPolicy, typename Array>
void initialize(const thrust::detail::execution_policy_base<DerivedPolicy>& exec, Array& keys)
{
    using example::detail::initialize;

    initialize(thrust::detail::derived_cast(thrust::detail::strip_const(exec)), keys);
}

template<typename DerivedPolicy, typename Array>
void thrust_example_1(const thrust::detail::execution_policy_base<DerivedPolicy>& exec, Array& keys)
{
    using example::detail::thrust_example_1;

    thrust_example_1(thrust::detail::derived_cast(thrust::detail::strip_const(exec)), keys);
}

template<typename DerivedPolicy, typename Array>
void thrust_example_2(const thrust::detail::execution_policy_base<DerivedPolicy>& exec, Array& keys)
{
    using example::detail::thrust_example_2;

    thrust_example_2(thrust::detail::derived_cast(thrust::detail::strip_const(exec)), keys);
}

template<typename DerivedPolicy, typename Array>
void thrust_example_3(const thrust::detail::execution_policy_base<DerivedPolicy>& exec, Array& keys)
{
    using example::detail::thrust_example_3;

    thrust_example_3(thrust::detail::derived_cast(thrust::detail::strip_const(exec)), keys);
}

template<typename Array>
void thrust_example_1(Array& keys)
{
    typename thrust::iterator_system<typename Array::iterator>::type system;
    example::thrust_example_1(system, keys);
}

template<typename Array>
void thrust_example_2(Array& keys)
{
    typename thrust::iterator_system<typename Array::iterator>::type system;
    example::thrust_example_2(system, keys);
}

template<typename Array>
void thrust_example_3(Array& keys)
{
    typename thrust::iterator_system<typename Array::iterator>::type system;
    example::thrust_example_3(system, keys);
}

} // end namespace example

