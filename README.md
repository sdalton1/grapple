<hr>
<h3>Grapple</h3>

Grapple is a specialized Thrust execution policy that intercepts Thrust
functions for performance profiling and debugging. It is especially
useful for tuning the performance of complex Thrust-based libraries.

<br><hr>
<h3>A Simple Example</h3>

```C++
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

    grapple.print();

    return 0;
}
```
