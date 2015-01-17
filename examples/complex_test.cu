#include <grapple/grapple.h>

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
void thrust_example_1(const thrust::detail::execution_policy_base<DerivedPolicy>& exec, Array& keys)
{
    Array values(keys.size());
    initialize(keys);

    thrust::sort(exec, keys.begin(), keys.end());
    thrust::reduce(exec, keys.begin(), keys.end());
    thrust::adjacent_difference(exec, keys.begin(), keys.end(), values.begin());
}

template<typename DerivedPolicy, typename Array>
void thrust_example_2(const thrust::detail::execution_policy_base<DerivedPolicy>& exec, Array& keys)
{
    initialize(keys);

    thrust::sort(exec, keys.begin(), keys.end());
    thrust::unique(exec, keys.begin(), keys.end());
}

template<typename DerivedPolicy, typename Array>
void thrust_example_3(const thrust::detail::execution_policy_base<DerivedPolicy>& exec, Array& keys)
{
    thrust_example_1(exec, keys);
    thrust_example_2(exec, keys);
}

// grapple intercept file
enum
{
    THRUST_EXAMPLE_1,
    THRUST_EXAMPLE_2,
    THRUST_EXAMPLE_3,
};

struct example_grapple_init
{
    example_grapple_init(void)
    {
        grapple_map.insert(THRUST_EXAMPLE_1, "thrust_example_1");
        grapple_map.insert(THRUST_EXAMPLE_2, "thrust_example_2");
        grapple_map.insert(THRUST_EXAMPLE_3, "thrust_example_3");
    }
};
static example_grapple_init static_example;

template<typename Array>
void thrust_example_1(grapple_system& exec, Array& keys)
{
    exec.start(THRUST_EXAMPLE_1);
    thrust_example_1(exec.policy(), keys);
    exec.stop();
}

template<typename Array>
void thrust_example_2(grapple_system& exec, Array& keys)
{
    exec.start(THRUST_EXAMPLE_2);
    thrust_example_2(exec.policy(), keys);
    exec.stop();
}

template<typename Array>
void thrust_example_3(grapple_system& exec, Array& keys)
{
    exec.start(THRUST_EXAMPLE_3);
    thrust_example_3(exec.policy(), keys);
    exec.stop();
}

// main file
int main(void)
{
    size_t N = 1<<18;

    grapple_system grapple;

    thrust::device_vector<float> keys(N);
    thrust_example_1(grapple, keys);
    thrust_example_2(grapple, keys);
    thrust_example_3(grapple, keys);

    return 0;
}

