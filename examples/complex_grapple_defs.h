#include <grapple/grapple.h>
#include "complex_test.h"

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

