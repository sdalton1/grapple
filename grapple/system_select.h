template<typename Iterator>
typename thrust::detail::disable_if<
thrust::system::detail::generic::select_system1_exists<
typename thrust::iterator_system<Iterator>::type>::value,
         typename thrust::iterator_system<Iterator>::type &
         >::type
         get_system(Iterator)
{
    using thrust::system::detail::generic::select_system;

    typedef typename thrust::iterator_system<Iterator>::type System;

    System system;

    return select_system(system);
}

template<typename Iterator1, typename Iterator2>
typename thrust::detail::enable_if_defined<
thrust::detail::minimum_system<typename thrust::iterator_system<Iterator1>::type,
       typename thrust::iterator_system<Iterator2>::type>
       >::type
       get_system(Iterator1, Iterator2)
{
    using thrust::system::detail::generic::select_system;

    typedef typename thrust::iterator_system<Iterator1>::type System1;
    typedef typename thrust::iterator_system<Iterator2>::type System2;

    System1 system1;
    System2 system2;

    return select_system(system1,system2);
}

template<typename Iterator1, typename Iterator2, typename Iterator3>
typename thrust::detail::lazy_disable_if<
thrust::system::detail::generic::select_system3_exists<
typename thrust::iterator_system<Iterator1>::type,
         typename thrust::iterator_system<Iterator2>::type,
         typename thrust::iterator_system<Iterator3>::type>::value,
         thrust::detail::minimum_system<typename thrust::iterator_system<Iterator1>::type,
         typename thrust::iterator_system<Iterator2>::type,
         typename thrust::iterator_system<Iterator3>::type>
         >::type
         get_system(Iterator1, Iterator2, Iterator3)
{
    using thrust::system::detail::generic::select_system;

    typedef typename thrust::iterator_system<Iterator1>::type System1;
    typedef typename thrust::iterator_system<Iterator2>::type System2;
    typedef typename thrust::iterator_system<Iterator3>::type System3;

    System1 system1;
    System2 system2;
    System3 system3;

    return select_system(system1,system2,system3);
}

template<typename Iterator1, typename Iterator2, typename Iterator3, typename Iterator4>
typename thrust::detail::lazy_disable_if<
thrust::system::detail::generic::select_system4_exists<
typename thrust::iterator_system<Iterator1>::type,
         typename thrust::iterator_system<Iterator2>::type,
         typename thrust::iterator_system<Iterator3>::type,
         typename thrust::iterator_system<Iterator4>::type>::value,
         thrust::detail::minimum_system<typename thrust::iterator_system<Iterator1>::type,
         typename thrust::iterator_system<Iterator2>::type,
         typename thrust::iterator_system<Iterator3>::type,
         typename thrust::iterator_system<Iterator4>::type>
         >::type
         get_system(Iterator1, Iterator2, Iterator3, Iterator4)
{
    using thrust::system::detail::generic::select_system;

    typedef typename thrust::iterator_system<Iterator1>::type System1;
    typedef typename thrust::iterator_system<Iterator2>::type System2;
    typedef typename thrust::iterator_system<Iterator3>::type System3;
    typedef typename thrust::iterator_system<Iterator4>::type System4;

    System1 system1;
    System2 system2;
    System3 system3;
    System4 system4;

    return select_system(system1,system2,system3,system4);
}
