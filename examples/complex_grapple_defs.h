#include <grapple/grapple.h>
#include "complex_test.h"

// grapple intercept file

// define complex function markers
enum
{
    THRUST_EXAMPLE_INIT,
    THRUST_EXAMPLE_1,
    THRUST_EXAMPLE_2,
    THRUST_EXAMPLE_3,
};

namespace grapple
{
// insert example function markers and names
// into global grapple map
struct example_grapple_init
{
    example_grapple_init(void)
    {
        insert(THRUST_EXAMPLE_INIT, "initialize");
        insert(THRUST_EXAMPLE_1, "thrust_example_1");
        insert(THRUST_EXAMPLE_2, "thrust_example_2");
        insert(THRUST_EXAMPLE_3, "thrust_example_3");
    }
};
static example_grapple_init static_example;

// intercept grapple execution policy functions for profiling
template<typename Array>
void initialize(grapple_system& exec, Array& keys)
{
    using example::detail::initialize;

    exec.start(THRUST_EXAMPLE_INIT);
    initialize(exec.policy(), keys);
    exec.stop();
}

template<typename Array>
void thrust_example_1(grapple_system& exec, Array& keys)
{
    using example::detail::thrust_example_1;

    exec.start(THRUST_EXAMPLE_1);
    thrust_example_1(exec.policy(), keys);
    exec.stop();
}

template<typename Array>
void thrust_example_2(grapple_system& exec, Array& keys)
{
    using example::detail::thrust_example_2;

    exec.start(THRUST_EXAMPLE_2);
    thrust_example_2(exec.policy(), keys);
    exec.stop();
}

template<typename Array>
void thrust_example_3(grapple_system& exec, Array& keys)
{
    using example::detail::thrust_example_3;

    exec.start(THRUST_EXAMPLE_3);
    thrust_example_3(exec.policy(), keys);
    exec.stop();
}
} // end namespace grapple

