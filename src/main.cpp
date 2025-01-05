#include "markshare.hpp"

#include <algorithm>
#include <bitset>
#include <cassert>
#include <climits> // For CHAR_BIT
#include <cstddef>
#include <iostream>
#include <numeric>
#include <omp.h>

size_t highestSetBit(size_t value) {
    if (value == 0) return -1; // No bits are set
#if defined(__SIZEOF_SIZE_T__) && __SIZEOF_SIZE_T__ == 8
    return (sizeof(size_t) * CHAR_BIT - 1) - __builtin_clzll(value);
#elif defined(__SIZEOF_SIZE_T__) && __SIZEOF_SIZE_T__ == 4
    return (sizeof(size_t) * CHAR_BIT - 1) - __builtin_clz(value);
#else
#error Unsupported size_t size
#endif
}

void print_bits(size_t value) {
    if (value == 0) {
        std::cout << "0"; // Special case: value is 0
        return;
    }

    // Determine the position of the highest set bit
    size_t msb = 0;
    for (size_t i = sizeof(size_t) * 8; i > 0; --i) {
        if (value & (1ULL << (i - 1))) {
            msb = i - 1;
            break;
        }
    }

    // Print bits from the highest set bit down to 0
    for (size_t i = msb + 1; i > 0; --i) {
        std::cout << ((value & (1ULL << (i - 1))) ? '1' : '0');
    }
}

std::vector<size_t> generate_subsets(const std::vector<size_t>& weights) {
    size_t n = weights.size();
    size_t total_subsets = 1ULL << n; /* Total subsets is 2^n. */

    printf("Generating %ld possible subsets.\n", total_subsets);
    std::vector<size_t> subsets(total_subsets, 0);

    for (size_t pass = 0; pass < n; ++pass) {
        const size_t weight = weights[pass];
        /* Step size corresponds to 2^pass (position of the bit). */
        size_t step = 1ULL << pass;

        #pragma omp parallel for
        for (size_t i = 0; i < total_subsets; i += step * 2) {
            /* Add `number` to all subsets where the `pass`-th bit is set. */
            for (size_t j = 0; j < step; ++j) {
                subsets[i + step + j] += weight;
            }
        }
    }

    return subsets;
}

// Function to sort an array and obtain sorted indices
std::vector<size_t> sort_indices(const std::vector<size_t>& arr, bool ascending) {
    size_t n = arr.size();

    // Create indices list from 0 to n-1 using std::iota
    std::vector<size_t> indices(n);
    std::iota(indices.begin(), indices.end(), 0);

    // Sort indices based on corresponding values in the array
    if (ascending)
    {
        std::sort(indices.begin(), indices.end(), [&arr](size_t i1, size_t i2) {
            return arr[i1] < arr[i2];
        });
    }
    else
    {
        std::sort(indices.begin(), indices.end(), [&arr](size_t i1, size_t i2) {
            return arr[i1] > arr[i2];
        });
    }

    return indices;
}

size_t print_subset_and_compute_sum(const std::vector<size_t>& numbers, size_t index) {
    std::cout << "Subset for index " << index << " (binary " << std::bitset<64>(index) << "): ";
    size_t sum = 0;

    bool hasElements = false;
    for (size_t i = 0; i < numbers.size(); ++i) {
        // Check if the i-th bit in the index is set
        if (index & (1ULL << i)) {
            if (hasElements) {
                std::cout << ", ";
            }
            std::cout << numbers[i];
            sum += numbers[i];
            hasElements = true;
        }
    }

    if (!hasElements) {
        std::cout << "Empty";
    }

    std::cout << std::endl;
    return sum;
}

void print_two_list_solution(size_t index_list1, size_t index_list2, const std::vector<size_t>& list1, const std::vector<size_t>& list2)
{
    auto sum1 = print_subset_and_compute_sum(list1, index_list1);
    auto sum2 = print_subset_and_compute_sum(list2, index_list2);

    std::cout << "The sum is " << sum1 << " + " << sum2 << " = " << sum1 + sum2 << std::endl;
}

int main()
{
    const int seed = 2025;
    /* Generate/read instance. For now, random instances. */
    MarkShareFeas instance(6, seed);

    /* Solve the instance using one of the available algorithms. */

    /* Create the one dimensional subset sum problem. */
    const size_t one_dim_rhs = instance.b()[0];
    printf("One dim rhs: %ld\n", one_dim_rhs);

    const size_t split_index = instance.n() / 2;
    std::vector<size_t> one_dim_subsetsum(instance.A().begin(), instance.A().begin() + instance.n());

    std::vector<size_t> list1(one_dim_subsetsum.begin(), one_dim_subsetsum.begin() + split_index);
    std::vector<size_t> list2(one_dim_subsetsum.begin() + split_index, one_dim_subsetsum.end());

    assert(list1.size() + list2.size() == instance.n());

    /* We produce the possible subsets for list1 and list2. Each subset corresponds to an entry in the vector. The elements used are given by the binary representation of the accessed element. */
    auto subsets1 = generate_subsets(list1);
    auto subsets2 = generate_subsets(list2);

    printf("Sorting the lists!\n");

    /* Sort the subsets, subsets1 ascending, subsets2 descending. */
    auto asc_indicies1 = sort_indices(subsets1, true);
    auto desc_indicies2 = sort_indices(subsets2, false);

    /* Iterate the lists (one from the front and one from the back an generate solutions to the subset sum problem). */
    size_t pos_list1 = 0;
    size_t pos_list2 = 0;

    while (pos_list1 < subsets1.size() && pos_list2 < subsets2.size())
    {
        const size_t elem_asc = subsets1[asc_indicies1[pos_list1]];
        const size_t elem_desc = subsets2[desc_indicies2[pos_list2]];

        const size_t elem_sum = elem_asc + elem_desc;

        if (elem_sum == one_dim_rhs)
        {
            printf("Found solution! \n");

            print_two_list_solution(asc_indicies1[pos_list1], desc_indicies2[pos_list2], list1, list2);
            return 0;
        }
        else if (elem_sum < one_dim_rhs)
            ++pos_list1;
        else
            ++pos_list2;
    }

    return 0;
}