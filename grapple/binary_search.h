#include <thrust/detail/config.h>
#include <thrust/binary_search.h>

namespace grapple
{

template<typename ForwardIterator, typename LessThanComparable>
ForwardIterator lower_bound(grapple_system &exec,
                            ForwardIterator first,
                            ForwardIterator last,
                            const LessThanComparable &value)
{
    exec.start(THRUST_LOWER_BOUND);
    ForwardIterator ret = thrust::lower_bound(exec.policy(get_system(first)), first, last, value);
    exec.stop();

    return ret;
}

template<typename ForwardIterator, typename T, typename StrictWeakOrdering>
ForwardIterator lower_bound(grapple_system &exec,
                            ForwardIterator first,
                            ForwardIterator last,
                            const T &value,
                            StrictWeakOrdering comp)
{
    exec.start(THRUST_LOWER_BOUND);
    ForwardIterator ret = thrust::lower_bound(exec.policy(get_system(first)), first, last, value, comp);
    exec.stop();

    return ret;
}

template<typename ForwardIterator, typename LessThanComparable>
ForwardIterator upper_bound(grapple_system &exec,
                            ForwardIterator first,
                            ForwardIterator last,
                            const LessThanComparable &value)
{
    exec.start(THRUST_UPPER_BOUND);
    ForwardIterator ret = thrust::upper_bound(exec.policy(get_system(first)), first, last, value);
    exec.stop();

    return ret;
}

template<typename ForwardIterator, typename T, typename StrictWeakOrdering>
ForwardIterator upper_bound(grapple_system &exec,
                            ForwardIterator first,
                            ForwardIterator last,
                            const T &value,
                            StrictWeakOrdering comp)
{
    exec.start(THRUST_UPPER_BOUND);
    ForwardIterator ret = thrust::upper_bound(exec.policy(get_system(first)), first, last, value, comp);
    exec.stop();

    return ret;
}

template <typename ForwardIterator, typename LessThanComparable>
bool binary_search(grapple_system &exec,
                   ForwardIterator first,
                   ForwardIterator last,
                   const LessThanComparable& value)
{
    exec.start(THRUST_BINARY_SEARCH);
    bool ret = thrust::binary_search(exec.policy(get_system(first)), first, last, value);
    exec.stop();

    return ret;
}

template <typename ForwardIterator, typename T, typename StrictWeakOrdering>
bool binary_search(grapple_system &exec,
                   ForwardIterator first,
                   ForwardIterator last,
                   const T& value,
                   StrictWeakOrdering comp)
{
    exec.start(THRUST_BINARY_SEARCH);
    bool ret = thrust::binary_search(exec.policy(get_system(first)), first, last, value, comp);
    exec.stop();

    return ret;
}

template <typename ForwardIterator, typename LessThanComparable>
thrust::pair<ForwardIterator, ForwardIterator>
equal_range(grapple_system &exec,
            ForwardIterator first,
            ForwardIterator last,
            const LessThanComparable& value)
{
    exec.start(THRUST_EQUAL_RANGE);
    thrust::pair<ForwardIterator, ForwardIterator> ret = thrust::equal_range(exec.policy(get_system(first)), first, last, value);
    exec.stop();

    return ret;
}

template <typename ForwardIterator, typename T, typename StrictWeakOrdering>
thrust::pair<ForwardIterator, ForwardIterator>
equal_range(grapple_system &exec,
            ForwardIterator first,
            ForwardIterator last,
            const T& value,
            StrictWeakOrdering comp)
{
    exec.start(THRUST_EQUAL_RANGE);
    thrust::pair<ForwardIterator, ForwardIterator> ret = thrust::equal_range(exec.policy(get_system(first)), first, last, value, comp);
    exec.stop();

    return ret;
}


template <typename ForwardIterator, typename InputIterator, typename OutputIterator>
OutputIterator lower_bound(grapple_system &exec,
                           ForwardIterator first,
                           ForwardIterator last,
                           InputIterator values_first,
                           InputIterator values_last,
                           OutputIterator result)
{
    exec.start(THRUST_LOWER_BOUND);
    OutputIterator ret = thrust::lower_bound(exec.policy(get_system(first,values_first,result)), first, last, values_first, values_last, result);
    exec.stop();

    return ret;
}

template <typename ForwardIterator, typename InputIterator, typename OutputIterator, typename StrictWeakOrdering>
OutputIterator lower_bound(grapple_system &exec,
                           ForwardIterator first,
                           ForwardIterator last,
                           InputIterator values_first,
                           InputIterator values_last,
                           OutputIterator result,
                           StrictWeakOrdering comp)
{
    exec.start(THRUST_LOWER_BOUND);
    OutputIterator ret = thrust::lower_bound(exec.policy(get_system(first,values_first,result)), first, last, values_first, values_last, result, comp);
    exec.stop();

    return ret;
}

template <typename ForwardIterator, typename InputIterator, typename OutputIterator>
OutputIterator upper_bound(grapple_system &exec,
                           ForwardIterator first,
                           ForwardIterator last,
                           InputIterator values_first,
                           InputIterator values_last,
                           OutputIterator result)
{
    exec.start(THRUST_UPPER_BOUND);
    OutputIterator ret = thrust::upper_bound(exec.policy(get_system(first,values_first,result)), first, last, values_first, values_last, result);
    exec.stop();

    return ret;
}

template <typename ForwardIterator, typename InputIterator, typename OutputIterator, typename StrictWeakOrdering>
OutputIterator upper_bound(grapple_system &exec,
                           ForwardIterator first,
                           ForwardIterator last,
                           InputIterator values_first,
                           InputIterator values_last,
                           OutputIterator result,
                           StrictWeakOrdering comp)
{
    exec.start(THRUST_UPPER_BOUND);
    OutputIterator ret = thrust::upper_bound(exec.policy(get_system(first,values_first,result)), first, last, values_first, values_last, result, comp);
    exec.stop();

    return ret;
}

template <typename ForwardIterator, typename InputIterator, typename OutputIterator>
OutputIterator binary_search(grapple_system &exec,
                             ForwardIterator first,
                             ForwardIterator last,
                             InputIterator values_first,
                             InputIterator values_last,
                             OutputIterator result)
{
    exec.start(THRUST_BINARY_SEARCH);
    OutputIterator ret = thrust::binary_search(exec.policy(get_system(first,values_first,result)), first, last, values_first, values_last, result);
    exec.stop();

    return ret;
}

template <typename ForwardIterator, typename InputIterator, typename OutputIterator, typename StrictWeakOrdering>
OutputIterator binary_search(grapple_system &exec,
                             ForwardIterator first,
                             ForwardIterator last,
                             InputIterator values_first,
                             InputIterator values_last,
                             OutputIterator result,
                             StrictWeakOrdering comp)
{
    exec.start(THRUST_BINARY_SEARCH);
    OutputIterator ret = thrust::binary_search(exec.policy(get_system(first,values_first,result)), first, last, values_first, values_last, result, comp);
    exec.stop();

    return ret;
}

}

