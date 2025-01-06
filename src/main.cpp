#include "markshare.hpp"

#include <algorithm>
#include <bitset>
#include <cassert>
#include <chrono>
#include <climits> // For CHAR_BIT
#include <cstddef>
#include <execution>
#include <iostream>
#include <numeric>
#include <queue>
#include <utility>

#include <omp.h>

typedef std::chrono::high_resolution_clock::time_point TimeVar;

#define duration(a) std::chrono::duration_cast<std::chrono::nanoseconds>(a).count()
#define timeNow() std::chrono::high_resolution_clock::now()

template <typename F, typename... Args>
double funcTime(F func, Args &&...args)
{
    TimeVar t1 = timeNow();
    func(std::forward<Args>(args)...);
    return duration(timeNow() - t1);
}

size_t highestSetBit(size_t value)
{
    if (value == 0)
        return -1; // No bits are set
#if defined(__SIZEOF_SIZE_T__) && __SIZEOF_SIZE_T__ == 8
    return (sizeof(size_t) * CHAR_BIT - 1) - __builtin_clzll(value);
#elif defined(__SIZEOF_SIZE_T__) && __SIZEOF_SIZE_T__ == 4
    return (sizeof(size_t) * CHAR_BIT - 1) - __builtin_clz(value);
#else
#error Unsupported size_t size
#endif
}

void print_bits(size_t value)
{
    if (value == 0)
    {
        std::cout << "0"; // Special case: value is 0
        return;
    }

    // Determine the position of the highest set bit
    size_t msb = 0;
    for (size_t i = sizeof(size_t) * 8; i > 0; --i)
    {
        if (value & (1ULL << (i - 1)))
        {
            msb = i - 1;
            break;
        }
    }

    // Print bits from the highest set bit down to 0
    for (size_t i = msb + 1; i > 0; --i)
    {
        std::cout << ((value & (1ULL << (i - 1))) ? '1' : '0');
    }
}

std::vector<size_t> generate_subsets(const std::vector<size_t> &weights)
{
    size_t n = weights.size();
    size_t total_subsets = 1ULL << n; /* Total subsets is 2^n. */

    printf("Generating %ld possible subsets.\n", total_subsets);
    std::vector<size_t> subsets(total_subsets, 0);

    for (size_t pass = 0; pass < n; ++pass)
    {
        const size_t weight = weights[pass];
        /* Step size corresponds to 2^pass (position of the bit). */
        size_t step = 1ULL << pass;

#pragma omp parallel for
        for (size_t i = 0; i < total_subsets; i += step * 2)
        {
            /* Add `number` to all subsets where the `pass`-th bit is set. */
            for (size_t j = 0; j < step; ++j)
            {
                subsets[i + step + j] += weight;
            }
        }
    }

    return subsets;
}

// Function to sort an array and obtain sorted indices
std::vector<size_t> sort_indices(const std::vector<size_t> &arr, bool ascending)
{
    size_t n = arr.size();

    // Create indices list from 0 to n-1 using std::iota
    std::vector<size_t> indices(n);
    std::iota(indices.begin(), indices.end(), 0);

    // Sort indices based on corresponding values in the array
    if (ascending)
    {
        std::sort(std::execution::par_unseq, indices.begin(), indices.end(), [&arr](size_t i1, size_t i2)
                  { return arr[i1] < arr[i2]; });
    }
    else
    {
        std::sort(std::execution::par_unseq, indices.begin(), indices.end(), [&arr](size_t i1, size_t i2)
                  { return arr[i1] > arr[i2]; });
    }

    return indices;
}

size_t print_subset_and_compute_sum(const std::vector<size_t> &numbers, size_t index)
{
    std::cout << "Subset for index " << index << " (binary " << std::bitset<64>(index) << "): ";
    size_t sum = 0;

    bool hasElements = false;
    for (size_t i = 0; i < numbers.size(); ++i)
    {
        // Check if the i-th bit in the index is set
        if (index & (1ULL << i))
        {
            if (hasElements)
            {
                std::cout << ", ";
            }
            std::cout << numbers[i];
            sum += numbers[i];
            hasElements = true;
        }
    }

    if (!hasElements)
    {
        std::cout << "Empty";
    }

    std::cout << std::endl;
    return sum;
}

void print_two_list_solution(size_t index_list1, size_t index_list2, const std::vector<size_t> &list1, const std::vector<size_t> &list2)
{
    auto sum1 = print_subset_and_compute_sum(list1, index_list1);
    auto sum2 = print_subset_and_compute_sum(list2, index_list2);

    std::cout << "The sum is " << sum1 << " + " << sum2 << " = " << sum1 + sum2 << std::endl;
}

bool two_list_algorithm(const std::vector<size_t> &subset_sum_1d, size_t rhs_subset_sum_1d)
{
    const size_t split_index = subset_sum_1d.size() / 2;

    std::vector<size_t> list1(subset_sum_1d.begin(), subset_sum_1d.begin() + split_index);
    std::vector<size_t> list2(subset_sum_1d.begin() + split_index, subset_sum_1d.end());

    assert(list1.size() + list2.size() == subset_sum_1d.size());

    /* We produce the possible subsets for list1 and list2. Each subset corresponds to an entry in the vector. The elements used are given by the binary representation of the accessed element. */
    auto subsets1 = generate_subsets(list1);
    auto subsets2 = generate_subsets(list2);

    printf("Sorting the lists!\n");

    /* Sort the subsets, subsets1 ascending, subsets2 descending. */
    auto asc_indicies1 = sort_indices(subsets1, true);
    auto desc_indicies2 = sort_indices(subsets2, false);

    printf("Finding solutions\n");
    /* Iterate the lists (one from the front and one from the back an generate solutions to the subset sum problem). */
    size_t pos_list1 = 0;
    size_t pos_list2 = 0;

    std::vector<std::pair<size_t, size_t>> solutions;

    while (pos_list1 < subsets1.size() && pos_list2 < subsets2.size())
    {
        const size_t elem_asc = subsets1[asc_indicies1[pos_list1]];
        const size_t elem_desc = subsets2[desc_indicies2[pos_list2]];

        const size_t elem_sum = elem_asc + elem_desc;

        if (elem_sum == rhs_subset_sum_1d)
        {
            printf("Found solution! \n");

            // check_solution()
            // solutions.push_back({elem_asc, elem_desc});
            // ++pos_list1;
            print_two_list_solution(asc_indicies1[pos_list1], desc_indicies2[pos_list2], list1, list2);
            return true;
        }
        else if (elem_sum < rhs_subset_sum_1d)
            ++pos_list1;
        else
            ++pos_list2;
    }

    return false;
}

void print_four_list_solution(size_t index_list1, size_t index_list2, size_t index_list3, size_t index_list4, const std::vector<size_t> &list1, const std::vector<size_t> &list2, const std::vector<size_t> &list3, const std::vector<size_t> &list4)
{
    auto sum1 = print_subset_and_compute_sum(list1, index_list1);
    auto sum2 = print_subset_and_compute_sum(list2, index_list2);
    auto sum3 = print_subset_and_compute_sum(list3, index_list3);
    auto sum4 = print_subset_and_compute_sum(list4, index_list4);

    std::cout << "The sum is " << sum1 << " + " << sum2 << " + " << sum3 << " + " << sum4 << " = " << sum1 + sum2 + sum3 + sum4 << std::endl;
}

bool shroeppel_shamir(const std::vector<size_t> &subset_sum_1d, size_t rhs_subset_sum_1d)
{
    const size_t split_index1 = subset_sum_1d.size() / 4;
    const size_t split_index2 = subset_sum_1d.size() / 2;
    const size_t split_index3 = 3 * subset_sum_1d.size() / 4;

    /* Get 4 sublists. */
    std::vector<size_t> list1(subset_sum_1d.begin(), subset_sum_1d.begin() + split_index1);
    std::vector<size_t> list2(subset_sum_1d.begin() + split_index1, subset_sum_1d.begin() + split_index2);
    std::vector<size_t> list3(subset_sum_1d.begin() + split_index2, subset_sum_1d.begin() + split_index3);
    std::vector<size_t> list4(subset_sum_1d.begin() + split_index3, subset_sum_1d.end());
    assert(list1.size() + list2.size() + list3.size() + list4.size() == subset_sum_1d.size());

    auto subsets1 = generate_subsets(list1);
    auto subsets2 = generate_subsets(list2);
    auto subsets3 = generate_subsets(list3);
    auto subsets4 = generate_subsets(list4);

    printf("Sorting the sublists\n");

    /* Sort subsets2 ascending, subsets4 descending. */
    auto asc_indicies_subsets2 = sort_indices(subsets2, true);
    auto desc_indicies_subsets4 = sort_indices(subsets4, false);

    printf("Building the queues\n");

    /* Create the priority queues q1 consisting of pairs {(i, 0) | i \in subsets1} and q2 consisting of {(i, 0) | i \in subsets3}. The priority/score for a pair (i, j)
     * is given subsets1[i] + subsets2[j] if the pair is in q1 and subsets3[i] + subsets4[j] if the pair is in q2. */

    /* Compare returns true if the first argument comes BEFORE the second argument. */
    auto min_cmp1 = [&](std::pair<size_t, size_t> a1, std::pair<size_t, size_t> a2) -> bool
    {
        return subsets1[a1.first] + subsets2[asc_indicies_subsets2[a1.second]] < subsets1[a2.first] + subsets2[asc_indicies_subsets2[a2.second]];
    };

    auto min_cmp2 = [&](std::pair<size_t, size_t> a1, std::pair<size_t, size_t> a2) -> bool
    {
        return subsets3[a1.first] + subsets4[desc_indicies_subsets4[a1.second]] < subsets3[a2.first] + subsets4[desc_indicies_subsets4[a2.second]];
    };

    std::priority_queue<std::pair<size_t, size_t>, std::vector<std::pair<size_t, size_t>>, decltype(min_cmp1)> q1(min_cmp1);
    std::priority_queue<std::pair<size_t, size_t>, std::vector<std::pair<size_t, size_t>>, decltype(min_cmp2)> q2(min_cmp2);

    size_t pos_subsets2 = 0;
    size_t pos_subsets4 = 0;

    // TODO: the initial insert can likely be improved by simple sorting.

    for (size_t i = 0; i < subsets1.size(); ++i)
    {
        /* If already the sum of these 2 elements is greater than the right hand side we can skip them. Subsequent combinations (e.g. with higher pos_subset2)
         * will only be even larger. */
        if (subsets1[i] + subsets2[asc_indicies_subsets2[pos_subsets2]] <= rhs_subset_sum_1d)
            q1.emplace(i, pos_subsets2);
    }

    for (size_t i = 0; i < subsets3.size(); ++i)
        q2.emplace(i, pos_subsets4);

    printf("Running the search loop\n");

    while (!q1.empty() && !q2.empty())
    {
        auto pair1 = q1.top();
        auto pair2 = q2.top();

        /* score_pair1 is the currently lowest score in {subsets1, subsets2} we are still considering */
        const size_t score_pair1 = subsets1[pair1.first] + subsets2[asc_indicies_subsets2[pair1.second]];
        /* score_pair2 is the currently highest score in {subsets3, subsets4} we are still considering */
        const size_t score_pair2 = subsets3[pair2.first] + subsets4[desc_indicies_subsets4[pair2.second]];

        const size_t score = score_pair1 + score_pair2;

        // printf("Score %ld = %ld + %ld + %ld + %ld\n", score, subsets1[pair1.first], subsets2[asc_indicies_subsets2[pair1.second]], subsets3[pair2.first], subsets4[desc_indicies_subsets4[pair2.second]]);

        if (score == rhs_subset_sum_1d)
        {
            /* We found a solution. Construct it, print it, and return. */
            printf("Found solution from SS-Algorithm! %ld == %ld\n", score, rhs_subset_sum_1d);
            print_four_list_solution(pair1.first, asc_indicies_subsets2[pair1.second], pair2.first, desc_indicies_subsets4[pair2.second], list1, list2, list3, list4);
            return true;
        }
        else if (score < rhs_subset_sum_1d)
        {
            q1.pop();
            ++pos_subsets2;

            while (pos_subsets2 + 1 < subsets2.size() && (subsets2[asc_indicies_subsets2[pos_subsets2]] == subsets2[asc_indicies_subsets2[pair1.second]] || (subsets1[pair1.first] + subsets2[asc_indicies_subsets2[pos_subsets2]] + score_pair2) < rhs_subset_sum_1d))
                ++pos_subsets2;

            /* Again, the element in q1 can only increase. So ignore elements that are already too big. */
            if (pos_subsets2 < subsets2.size() && subsets1[pair1.first] + subsets2[asc_indicies_subsets2[pos_subsets2]] <= rhs_subset_sum_1d)
                q1.emplace(pair1.first, pos_subsets2);
        }
        else if (score > rhs_subset_sum_1d)
        {
            q2.pop();
            ++pos_subsets4;

            /* Skip all entries in subsets4 until we find a smaller one. */
            while (pos_subsets4 + 1 < subsets4.size() && (subsets4[desc_indicies_subsets4[pos_subsets4]] == subsets4[desc_indicies_subsets4[pair2.second]] || (score_pair1 + subsets3[pair2.first] + subsets4[desc_indicies_subsets4[pos_subsets4]]) > rhs_subset_sum_1d))
                ++pos_subsets4;

            if (pos_subsets4 < subsets4.size() && subsets3[pair2.first] + subsets4[desc_indicies_subsets4[pos_subsets4]] <= rhs_subset_sum_1d)
                q2.emplace(pair2.first, pos_subsets4);
        }
    }
    return false;
}

int main()
{
    const int seed = 2026;
    /* Generate/read instance. For now, random instances. */
    MarkShareFeas instance(11, seed);

    /* Solve the instance using one of the available algorithms. */

    /* Create the one dimensional subset sum problem. */
    const size_t rhs_subset_sum_1d = instance.b()[0];
    const std::vector<size_t> subset_sum_1d(instance.A().begin(), instance.A().begin() + instance.n());
    printf("One dim rhs: %ld\n", rhs_subset_sum_1d);

    // two_list_algorithm(subset_sum_1d, rhs_subset_sum_1d);

    /* Shroeppel-Shamir */
    shroeppel_shamir(subset_sum_1d, rhs_subset_sum_1d);

    return 0;
}