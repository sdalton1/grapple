#include <thrust/device_vector.h>
#include <thrust/extrema.h>
#include <thrust/host_vector.h>
#include <thrust/sort.h>
#include <thrust/reduce.h>
#include <thrust/inner_product.h>
#include <thrust/functional.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/random.h>

#include <iostream>
#include <iterator>

#include <grapple/grapple.h>

template<typename DerivedPolicy, typename Array>
thrust::tuple<int,int> compute_mode(const thrust::execution_policy<DerivedPolicy>& exec,
                                    const Array& d_data)
{
    Array d_data_copy(d_data);

    // sort data to bring equal elements together
    thrust::sort(exec, d_data_copy.begin(), d_data_copy.end());

    // print the sorted data
    std::cout << "sorted data" << std::endl;
    thrust::copy(d_data_copy.begin(), d_data_copy.end(), std::ostream_iterator<int>(std::cout, " "));
    std::cout << std::endl;

    // count number of unique keys
    size_t num_unique = thrust::inner_product(exec,
                        d_data_copy.begin(), d_data_copy.end() - 1,
                        d_data_copy.begin() + 1,
                        0,
                        thrust::plus<int>(),
                        thrust::not_equal_to<int>()) + 1;

    // count multiplicity of each key
    thrust::device_vector<int> d_output_keys(num_unique);
    thrust::device_vector<int> d_output_counts(num_unique);
    thrust::reduce_by_key(exec,
                          d_data_copy.begin(), d_data_copy.end(),
                          thrust::constant_iterator<int>(1),
                          d_output_keys.begin(),
                          d_output_counts.begin());

    // print the counts
    std::cout << "values" << std::endl;
    thrust::copy(d_output_keys.begin(), d_output_keys.end(), std::ostream_iterator<int>(std::cout, " "));
    std::cout << std::endl;

    // print the counts
    std::cout << "counts" << std::endl;
    thrust::copy(d_output_counts.begin(), d_output_counts.end(), std::ostream_iterator<int>(std::cout, " "));
    std::cout << std::endl;

    // find the index of the maximum count
    thrust::device_vector<int>::iterator mode_iter;
    mode_iter = thrust::max_element(exec, d_output_counts.begin(), d_output_counts.end());

    int mode = d_output_keys[mode_iter - d_output_counts.begin()];
    int occurrences = *mode_iter;

    return thrust::tie(mode, occurrences);
}

template<typename Array>
thrust::tuple<int,int> compute_mode(const Array& d_data)
{
    using thrust::system::detail::generic::select_system;
    typename thrust::iterator_system<typename Array::iterator>::type system;
    return compute_mode(select_system(system), d_data);
}

int main(void)
{
    const size_t N = 30;
    const size_t M = 10;
    thrust::default_random_engine rng;
    thrust::uniform_int_distribution<int> dist(0, M - 1);

    // generate random data on the host
    thrust::host_vector<int> h_data(N);
    for(size_t i = 0; i < N; i++)
        h_data[i] = dist(rng);

    // transfer data to device
    thrust::device_vector<int> d_data(h_data);

    // print the initial data
    std::cout << "initial data" << std::endl;
    thrust::copy(d_data.begin(), d_data.end(), std::ostream_iterator<int>(std::cout, " "));
    std::cout << std::endl << std::endl;

    {
      int mode, occurrences;
      thrust::tie(mode, occurrences) = compute_mode(d_data);
      std::cout << "Modal value " << mode << " occurs " << occurrences << " times " << std::endl;
    }

    std::cout << std::endl;

    {
      int mode, occurrences;
      grapple_system grapple;
      thrust::tie(mode, occurrences) = compute_mode(grapple, d_data);
      std::cout << "Modal value " << mode << " occurs " << occurrences << " times " << std::endl;
    }

    return 0;
}

