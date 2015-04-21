#include <grapple/grapple.h>
#include "complex_test.h"

// grapple intercept file

// define complex function markers
enum
{
    THRUST_EXAMPLE_1,
    THRUST_EXAMPLE_2,
    THRUST_EXAMPLE_3,
};

// insert example function markers and names
// into global grapple map
struct example_grapple_init
{
    example_grapple_init(void)
    {
        grapple::insert(THRUST_EXAMPLE_1, "thrust_example_1");
        grapple::insert(THRUST_EXAMPLE_2, "thrust_example_2");
        grapple::insert(THRUST_EXAMPLE_3, "thrust_example_3");
    }
};
static example_grapple_init static_example;

// intercept grapple execution policy functions for profiling
template<typename Array>
void thrust_example_1(grapple::grapple_system& exec, Array& keys)
{
    exec.start(THRUST_EXAMPLE_1);
    thrust_example_1(exec.policy(), keys);
    exec.stop();
}

template<typename Array>
void thrust_example_2(grapple::grapple_system& exec, Array& keys)
{
    exec.start(THRUST_EXAMPLE_2);
    thrust_example_2(exec.policy(), keys);
    exec.stop();
}

template<typename Array>
void thrust_example_3(grapple::grapple_system& exec, Array& keys)
{
    exec.start(THRUST_EXAMPLE_3);
    thrust_example_3(exec.policy(), keys);
    exec.stop();
}

