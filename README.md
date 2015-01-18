<hr>
<h3>Grapple</h3>

Grapple is a specialized Thrust execution policy that intercepts Thrust
functions for performance profiling and debugging. It is especially
useful for tuning the performance of complex Thrust-based libraries.

<br><hr>
<h3>A Simple Example</h3>

~~~{.cpp}
#include <grapple/grapple.h>

#include <thrust/device_vector.h>
#include <thrust/random.h>
#include <thrust/sort.h>

void initialize(thrust::device_vector<float>& v)
{
    thrust::default_random_engine rng(123456);
    thrust::uniform_int_distribution<int> dist(2, 19);
    for(size_t i = 0; i < v.size(); i++)
        v[i] = dist(rng) / 2.0f;
}

int main(void)
{
    size_t N = 1<<18;

    grapple_system grapple;

    thrust::device_vector<float> keys(N);
    initialize(keys);
    thrust::sort(grapple, keys.begin(), keys.end());

    return 0;
}
~~~

In the above example grapple transparently intercepts the Thrust sort
function. Grapple collects data concerning the execution time and memory
allocation and automatically prints this information to the console once
the grapple object goes out of scope.

~~~{.shell}
[ 0]sort                    : 0.790976 (ms), allocated : 1137408    bytes
~~~

<br><hr>
<h3>Code Litter</h3>

Grapple is designed to exploit execution policies in order to avoid
littering Thrust based libraries with spurious performance profiling
code. Coarse-grained profiling of entire functions, such as my_func,
provides a general performance overview but fine-grained profiling
requires altering individual functions for the specific purpose of
profiling.

~~~{.cpp}
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/reduce.h>

template<typename Array>
void my_func(keys)
{
  // initialize some performance timer
  timer tsort;
  thrust::sort(keys.begin(), keys.end());
  // return sort elapsed time
  tsort.milliseconds_elapsed();

  timer treduce;
  thrust::reduce(keys.begin(), keys.end());
  treduce.milliseconds_elapsed();
}

int main(void)
{
    thrust::device_vector<float> keys(10);

    timer tmy_func;
    my_func(keys);
    return tmy_func.milliseconds_elapsed();

    return 0;
}
~~~

Grapple leverages the use of execution policies to provide flexible
profiling and allows to user to control the level of granularity without
adding specialized code to any underlying functions. The cost of this
flexibility is tigher integration with the Thrust dispatch system.

Adding code to integrate tightly with execution polices requires
additional code but the use is much more flexible than performance
specific code.

~~~{.cpp}
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/sort.h>
#include <thrust/reduce.h>

template<typename DerivedPolicy, typename Array>
void my_func(const thrust::detail::execution_policy_base<DerivedPolicy>& exec, Array& keys)
{
  DerivedPolicy& derived(thrust::detail::derived_cast(thrust::detail::strip_const(exec)));

  thrust::sort(derived, keys.begin(), keys.end());
  thrust::reduce(derived, keys.begin(), keys.end());
}

template<typename Array>
void my_func(Array& keys)
{
  using thrust::system::detail::generic::select_system;

  typename thrust::iterator_system<typename Array::iterator>::type system;

  my_func(select_system(system), keys);
}

int main(void)
{
    thrust::device_vector<float> keys(10);

    my_func(keys);

    return 0;
}
~~~
