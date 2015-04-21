#pragma once

#include <map>
#include <string>

enum
{
    THRUST_ADJACENT_DIFFERENCE = 1<<20,

    THRUST_LOWER_BOUND,
    THRUST_UPPER_BOUND,
    THRUST_BINARY_SEARCH,
    THRUST_EQUAL_RANGE,

    THRUST_COPY,
    THRUST_COPY_N,
    THRUST_COPY_IF,

    THRUST_COUNT,
    THRUST_COUNT_IF,

    THRUST_EQUAL,

    THRUST_MIN_ELEMENT,
    THRUST_MAX_ELEMENT,
    THRUST_MINMAX_ELEMENT,

    THRUST_FILL,
    THRUST_FILL_N,

    THRUST_FOR_EACH,
    THRUST_FOR_EACH_N,

    THRUST_GATHER,
    THRUST_GATHER_IF,

    THRUST_GENERATE,
    THRUST_GENERATE_N,

    THRUST_INNER_PRODUCT,

    THRUST_MERGE,
    THRUST_MERGE_BY_KEY,

    THRUST_MISMATCH,

    THRUST_PARTITION,
    THRUST_PARTITION_COPY,
    THRUST_STABLE_PARTITION,
    THRUST_STABLE_PARTITION_COPY,
    THRUST_PARTITION_POINT,
    THRUST_IS_PARTITIONED,

    THRUST_REDUCE,
    THRUST_REDUCE_BY_KEY,

    THRUST_REMOVE,
    THRUST_REMOVE_COPY,
    THRUST_REMOVE_IF,
    THRUST_REMOVE_COPY_IF,

    THRUST_REPLACE,
    THRUST_REPLACE_COPY,
    THRUST_REPLACE_IF,
    THRUST_REPLACE_COPY_IF,

    THRUST_REVERSE,
    THRUST_REVERSE_COPY,

    THRUST_INCLUSIVE_SCAN,
    THRUST_EXCLUSIVE_SCAN,
    THRUST_INCLUSIVE_SCAN_BY_KEY,
    THRUST_EXCLUSIVE_SCAN_BY_KEY,

    THRUST_SCATTER,
    THRUST_SCATTER_IF,

    THRUST_SEQUENCE,

    THRUST_SET_DIFFERENCE,
    THRUST_SET_INTERSECTION,
    THRUST_SET_SYMMETRIC_DIFFERENCE,
    THRUST_SET_UNION,
    THRUST_SET_DIFFERENCE_BY_KEY,
    THRUST_SET_INTERSECTION_BY_KEY,
    THRUST_SET_SYMMETRIC_DIFFERENCE_BY_KEY,
    THRUST_SET_UNION_BY_KEY,

    THRUST_SORT,
    THRUST_SORT_BY_KEY,
    THRUST_STABLE_SORT,
    THRUST_STABLE_SORT_BY_KEY,
    THRUST_IS_SORTED,
    THRUST_IS_SORTED_UNTIL,

    THRUST_TABULATE,

    THRUST_TRANSFORM,
    THRUST_TRANSFORM_IF,

    THRUST_TRANSFORM_REDUCE,

    THRUST_TRANSFORM_INCLUSIVE_SCAN,
    THRUST_TRANSFORM_EXCLUSIVE_SCAN,

    THRUST_UNINITIALIZED_COPY,
    THRUST_UNINITIALIZED_COPY_N,

    THRUST_UNINITIALIZED_FILL,
    THRUST_UNINITIALIZED_FILL_N,

    THRUST_UNIQUE,
    THRUST_UNIQUE_COPY,
    THRUST_UNIQUE_BY_KEY,
    THRUST_UNIQUE_BY_KEY_COPY,

    THRUST_LAST_KEY,
};

namespace grapple
{

class grapple_thrust_mapper
{
    std::map<std::string,int> create_map(void)
    {
        std::map<std::string,int> m;

        m["adjacent_difference"]      = THRUST_ADJACENT_DIFFERENCE;

        m["lower_bound"]              = THRUST_LOWER_BOUND;
        m["upper_bound"]              = THRUST_UPPER_BOUND;
        m["binary_search"]            = THRUST_BINARY_SEARCH;
        m["equal_range"]              = THRUST_EQUAL_RANGE;

        m["copy"]                     = THRUST_COPY;
        m["copy_n"]                   = THRUST_COPY_N;
        m["copy_if"]                  = THRUST_COPY_IF;

        m["count"]                    = THRUST_COUNT;
        m["count_if"]                 = THRUST_COUNT_IF;

        m["equal"]                    = THRUST_EQUAL;

        m["min_element"]              = THRUST_MIN_ELEMENT;
        m["max_element"]              = THRUST_MAX_ELEMENT;
        m["minmax_element"]           = THRUST_MINMAX_ELEMENT;

        m["fill"]                     = THRUST_FILL;
        m["fill_n"]                   = THRUST_FILL_N;

        m["for_each"]                 = THRUST_FOR_EACH;
        m["for_each_n"]               = THRUST_FOR_EACH_N;

        m["gather"]                   = THRUST_GATHER;
        m["gather_if"]                = THRUST_GATHER_IF;

        m["generate"]                 = THRUST_GENERATE;
        m["generate_n"]               = THRUST_GENERATE_N;

        m["inner_product"]            = THRUST_INNER_PRODUCT;

        m["merge"]                    = THRUST_MERGE;
        m["merge_by_key"]             = THRUST_MERGE_BY_KEY;

        m["mismatch"]                 = THRUST_MISMATCH;

        m["partition"]                = THRUST_PARTITION;
        m["partition_copy"]           = THRUST_PARTITION_COPY;
        m["stable_partition"]         = THRUST_STABLE_PARTITION;
        m["stable_partition_copy"]    = THRUST_STABLE_PARTITION_COPY;
        m["partition_point"]          = THRUST_PARTITION_POINT;
        m["is_partitioned"]           = THRUST_IS_PARTITIONED;

        m["reduce"]                   = THRUST_REDUCE;
        m["reduce_by_key"]            = THRUST_REDUCE_BY_KEY;

        m["remove"]                   = THRUST_REMOVE;
        m["remove_copy"]              = THRUST_REMOVE_COPY;
        m["remove_if"]                = THRUST_REMOVE_IF;
        m["remove_copy_if"]           = THRUST_REMOVE_COPY_IF;

        m["replace"]                  = THRUST_REPLACE;
        m["replace_if"]               = THRUST_REPLACE_IF;
        m["replace_copy"]             = THRUST_REPLACE_COPY;
        m["replace_copy_if"]          = THRUST_REPLACE_COPY_IF;

        m["reverse"]                  = THRUST_REVERSE;
        m["reverse_copy"]             = THRUST_REVERSE_COPY;

        m["inclusive_scan"]           = THRUST_INCLUSIVE_SCAN;
        m["exclusive_scan"]           = THRUST_EXCLUSIVE_SCAN;
        m["inclusive_scan_by_key"]    = THRUST_INCLUSIVE_SCAN_BY_KEY;
        m["exclusive_scan_by_key"]    = THRUST_EXCLUSIVE_SCAN_BY_KEY;

        m["scatter"]                  = THRUST_SCATTER;
        m["scatter_if"]               = THRUST_SCATTER_IF;

        m["sequence"]                 = THRUST_SEQUENCE;

        m["set_difference"]           = THRUST_SET_DIFFERENCE;
        m["set_intersection"]         = THRUST_SET_INTERSECTION;
        m["set_symmetric_difference"] = THRUST_SET_SYMMETRIC_DIFFERENCE;
        m["set_union"]                = THRUST_SET_UNION;
        m["set_difference_by_key"]    = THRUST_SET_DIFFERENCE_BY_KEY;
        m["set_intersection_by_key"]  = THRUST_SET_INTERSECTION_BY_KEY;
        m["set_symmetric_difference_by_key"]  = THRUST_SET_SYMMETRIC_DIFFERENCE_BY_KEY;
        m["set_union_by_key"]                 = THRUST_SET_UNION_BY_KEY;

        m["sort"]                     = THRUST_SORT;
        m["sort_by_key"]              = THRUST_SORT_BY_KEY;
        m["stable_sort"]              = THRUST_STABLE_SORT;
        m["stable_sort_by_key"]       = THRUST_STABLE_SORT_BY_KEY;
        m["is_sorted"]                = THRUST_IS_SORTED;
        m["is_sorted_until"]          = THRUST_IS_SORTED_UNTIL;

        m["tabulate"]                 = THRUST_TABULATE;

        m["transform"]                = THRUST_TRANSFORM;
        m["transform_if"]             = THRUST_TRANSFORM_IF;

        m["transform_reduce"]         = THRUST_TRANSFORM_REDUCE;

        m["transform_inclusive_scan"] = THRUST_TRANSFORM_INCLUSIVE_SCAN;
        m["transform_exclusive_scan"] = THRUST_TRANSFORM_EXCLUSIVE_SCAN;

        m["uninitialized_copy"]       = THRUST_UNINITIALIZED_COPY;
        m["uninitialized_copy_n"]     = THRUST_UNINITIALIZED_COPY_N;

        m["uninitialized_fill"]       = THRUST_UNINITIALIZED_FILL;
        m["uninitialized_fill_n"]     = THRUST_UNINITIALIZED_FILL_N;

        m["unique"]                   = THRUST_UNIQUE;
        m["unique_copy"]              = THRUST_UNIQUE_COPY;
        m["unique_by_key"]            = THRUST_UNIQUE_BY_KEY;
        m["unique_by_key_copy"]       = THRUST_UNIQUE_BY_KEY_COPY;

        return m;
    }

    std::map<int,std::string> create_reverse_map(void)
    {
        std::map<int,std::string> m;

        for(std::map<const std::string,int>::const_iterator iter = thrustMap.begin(); iter != thrustMap.end(); ++iter)
            m.insert(std::pair<int,std::string>(iter->second, iter->first));

        return m;
    }

public:

    grapple_thrust_mapper(void)
    {
        thrustMap = create_map();
        thrustReverseMap = create_reverse_map();
    }

    void insert(int index, const std::string& name)
    {
        thrustReverseMap.insert(std::pair<int,std::string>(index, name));
    }

    std::string find(int index)
    {
        return thrustReverseMap.find(index)->second;
    }

private:

    std::map<std::string,int> thrustMap;
    std::map<int,std::string> thrustReverseMap;
};
static grapple_thrust_mapper grapple_thrust_map;

enum grapple_type
{
    GRAPPLE_CPP,
    GRAPPLE_OMP,
    GRAPPLE_TBB,
    GRAPPLE_CUDA,
    GRAPPLE_D2H,
    GRAPPLE_H2D,
};

class grapple_system_mapper
{
    std::map<int,std::string> create_map(void)
    {
        std::map<int,std::string> m;

        m[GRAPPLE_CPP]     = "cpp ";
        m[GRAPPLE_TBB]     = "tbb ";
        m[GRAPPLE_OMP]     = "omp ";
        m[GRAPPLE_CUDA]    = "cuda";
        m[GRAPPLE_D2H]     = "d->h";
        m[GRAPPLE_H2D]     = "h->d";

        return m;
    }

public:

    grapple_system_mapper(void)
    {
        systemMap = create_map();
    }

    std::string find(int index)
    {
        return systemMap.find(index)->second;
    }

private:

    std::map<int,std::string> systemMap;
};
static grapple_system_mapper grapple_system_map;

void insert(int index, const std::string& name)
{
    grapple_thrust_map.insert(index, name);
}

} // end namespace grapple

