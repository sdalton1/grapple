// include base header file containing function definitions
#include "complex_test.h"
// intercept policy based functions for profiling with grapple
#include "complex_grapple_defs.h"

int main(void)
{
    size_t N = 1<<18;
    thrust::device_vector<float> keys(N);

    // call tests normally
    {
        std::cout << "Executing examples...";
        thrust_example_1(keys);
        thrust_example_2(keys);
        thrust_example_3(keys);
        std::cout << "complete!" << std::endl;
    }

    std::cout << std::endl;

    // call tests with grapple profiling
    {
        grapple::grapple_system exec;

        std::cout << "Executing examples with grapple...";
        thrust_example_1(exec, keys);
        thrust_example_2(exec, keys);
        thrust_example_3(exec, keys);
        std::cout << "complete!" << std::endl;

        // grapple performance printed automatically during destructor
    }

    return 0;
}


