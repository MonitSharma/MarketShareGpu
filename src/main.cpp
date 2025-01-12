#include "markshare.hpp"
#include "profiler.hpp"

#include "argparse.hpp"

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
#include <string>

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

std::vector<size_t> extract_subset(const std::vector<size_t> &numbers, size_t index, size_t offset = 0)
{
    std::vector<size_t> indices;

    for (size_t i = 0; i < numbers.size(); ++i)
    {
        // Check if the i-th bit in the index is set
        if (index & (1ULL << i))
        {
            indices.push_back(i + offset);
        }
    }
    return indices;
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

std::vector<size_t> extract_4_list_solution(size_t index_list1, size_t index_list2, size_t index_list3, size_t index_list4, const std::vector<size_t> &list1, const std::vector<size_t> &list2, const std::vector<size_t> &list3, const std::vector<size_t> &list4)
{
    std::vector<size_t> indicies1 = extract_subset(list1, index_list1, 0);
    std::vector<size_t> indicies2 = extract_subset(list2, index_list2, list1.size());
    std::vector<size_t> indicies3 = extract_subset(list3, index_list3, list1.size() + list2.size());
    std::vector<size_t> indicies4 = extract_subset(list4, index_list4, list1.size() + list2.size() + list3.size());
    indicies1.insert(indicies1.end(), indicies2.begin(), indicies2.end());
    indicies1.insert(indicies1.end(), indicies3.begin(), indicies3.end());
    indicies1.insert(indicies1.end(), indicies4.begin(), indicies4.end());

    return indicies1;
}

void print_four_list_solution(size_t index_list1, size_t index_list2, size_t index_list3, size_t index_list4, const std::vector<size_t> &list1, const std::vector<size_t> &list2, const std::vector<size_t> &list3, const std::vector<size_t> &list4)
{
    auto sum1 = print_subset_and_compute_sum(list1, index_list1);
    auto sum2 = print_subset_and_compute_sum(list2, index_list2);
    auto sum3 = print_subset_and_compute_sum(list3, index_list3);
    auto sum4 = print_subset_and_compute_sum(list4, index_list4);

    std::cout << "The sum is " << sum1 << " + " << sum2 << " + " << sum3 << " + " << sum4 << " = " << sum1 + sum2 + sum3 + sum4 << std::endl;
}

bool shroeppel_shamir(const std::vector<size_t> &subset_sum_1d, size_t rhs_subset_sum_1d, const MarkShareFeas &ms_inst)
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

    /* Sort subsets2 ascending, subsets4 descending. */
    auto asc_indicies_subsets2 = sort_indices(subsets2, true);
    auto desc_indicies_subsets4 = sort_indices(subsets4, false);

    /* Create the priority queues q1 consisting of pairs {(i, 0) | i \in subsets1} and q2 consisting of {(i, 0) | i \in subsets3}. The priority/score for a pair (i, j)
     * is given subsets1[i] + subsets2[j] if the pair is in q1 and subsets3[i] + subsets4[j] if the pair is in q2. */

    /* Compare returns true if the first argument comes BEFORE the second argument. Since however the priority queue outputs the largest element first,
     * we have to flip the > signs. */
    auto min_cmp = [&](std::pair<size_t, size_t> a1, std::pair<size_t, size_t> a2) -> bool
    {
        return subsets1[a1.first] + subsets2[asc_indicies_subsets2[a1.second]] > subsets1[a2.first] + subsets2[asc_indicies_subsets2[a2.second]];
    };

    auto max_cmp = [&](std::pair<size_t, size_t> a1, std::pair<size_t, size_t> a2) -> bool
    {
        return subsets3[a1.first] + subsets4[desc_indicies_subsets4[a1.second]] < subsets3[a2.first] + subsets4[desc_indicies_subsets4[a2.second]];
    };

    std::priority_queue<std::pair<size_t, size_t>, std::vector<std::pair<size_t, size_t>>, decltype(min_cmp)> q1(min_cmp);
    std::priority_queue<std::pair<size_t, size_t>, std::vector<std::pair<size_t, size_t>>, decltype(max_cmp)> q2(max_cmp);

    // TODO: the initial insert can likely be improved by simple sorting.

    for (size_t i = 0; i < subsets1.size(); ++i)
    {
        /* If already the sum of these 2 elements is greater than the right hand side we can skip them. Subsequent combinations (e.g. with higher pos_subset2)
         * will only be even larger. */
        if (subsets1[i] + subsets2[asc_indicies_subsets2[0]] <= rhs_subset_sum_1d)
            q1.emplace(i, 0);
    }

    for (size_t i = 0; i < subsets3.size(); ++i)
        q2.emplace(i, 0);

    printf("Running the search loop\n");

    ScopedProfiler("List traversal");
    while (!q1.empty() && !q2.empty())
    {
        auto pair1 = q1.top();
        auto pair2 = q2.top();

        size_t pos_subsets2 = pair1.second;
        size_t pos_subsets4 = pair2.second;

        /* score_pair1 is the currently lowest score in {subsets1, subsets2} we are still considering */
        const size_t score_pair1 = subsets1[pair1.first] + subsets2[asc_indicies_subsets2[pair1.second]];
        /* score_pair2 is the currently highest score in {subsets3, subsets4} we are still considering */
        const size_t score_pair2 = subsets3[pair2.first] + subsets4[desc_indicies_subsets4[pair2.second]];

        const size_t score = score_pair1 + score_pair2;

        if (score == rhs_subset_sum_1d)
        {
            ScopedProfiler("Solution validation");
            const auto pos2_val = subsets2[asc_indicies_subsets2[pos_subsets2]];
            const auto pos4_val = subsets4[desc_indicies_subsets4[pos_subsets4]];

            std::vector<size_t> same_value_subset2;
            std::vector<size_t> same_value_subset4;
            /* We've found a solution. Extract all other solutions for this pair of subsets1/subsets3 elements by quadratically combining all subsets2/subsets4 combinations. Then, drop both entries. */

            while (pos_subsets2 < subsets2.size() && pos2_val == subsets2[asc_indicies_subsets2[pos_subsets2]])
            {
                same_value_subset2.push_back(pos_subsets2);
                ++pos_subsets2;
            }
            assert(pos_subsets2 == subsets2.size() || subsets2[asc_indicies_subsets2[pos_subsets2]] > pos2_val);

            while (pos_subsets4 < subsets4.size() && pos4_val == subsets4[desc_indicies_subsets4[pos_subsets4]])
            {
                same_value_subset4.push_back(pos_subsets4);
                ++pos_subsets4;
            }
            assert(pos_subsets4 == subsets4.size() || subsets4[desc_indicies_subsets4[pos_subsets4]] < pos4_val);

            for (size_t pos_subset2 : same_value_subset2)
            {
                for (size_t pos_subset4 : same_value_subset4)
                {
                    const std::vector<size_t> sol_1d = extract_4_list_solution(pair1.first, asc_indicies_subsets2[pos_subset2], pair2.first, desc_indicies_subsets4[pos_subset4], list1, list2, list3, list4);

                    /* We found a solution. Construct it, print it, and return. */
                    if (ms_inst.is_solution_feasible(sol_1d))
                    {
                        printf("Found market share solution from SS-Algorithm! %ld == %ld\n", score, rhs_subset_sum_1d);

                        print_four_list_solution(pair1.first, asc_indicies_subsets2[pos_subset2], pair2.first, desc_indicies_subsets4[pos_subset4], list1, list2, list3, list4);
                        return true;
                    }
                }
            }

            q1.pop();
            if (pos_subsets2 < subsets2.size())
                q1.emplace(pair1.first, pos_subsets2);

            q2.pop();
            if (pos_subsets4 < subsets4.size())
                q2.emplace(pair2.first, pos_subsets4);
        }
        else if (score < rhs_subset_sum_1d)
        {
            q1.pop();
            ++pos_subsets2;

            while (pos_subsets2 + 1 < subsets2.size() && (subsets2[asc_indicies_subsets2[pos_subsets2]] == subsets2[asc_indicies_subsets2[pair1.second]] || (subsets1[pair1.first] + subsets2[asc_indicies_subsets2[pos_subsets2]] + score_pair2) < rhs_subset_sum_1d))
                ++pos_subsets2;

            /* Again, the element in q1 can only increase (or stay equal). So ignore elements that are already too big. */
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

int main(int argc, char *argv[])
{
    argparse::ArgumentParser program("markshare");

    size_t n_iter = 1;
    size_t seed = 2025;
    size_t m = 5;

    program.add_argument("-m", "--m")
        .store_into(m)
        .help("Number of rows of the markshare problem.")
        .required();

    program.add_argument("-s", "--seed")
        .store_into(seed)
        .help("Random seed for instance generation.")
        .default_value(2025);

    program.add_argument("-i", "--iter")
        .store_into(n_iter)
        .help("Number of problems to solve. Seed for problem of iteration i (starting from 0) is seed + i.")
        .default_value(1);

    try
    {
        program.parse_args(argc, argv);
    }
    catch (const std::exception &err)
    {
        std::cerr << err.what() << std::endl;
        std::cerr << program;
        std::exit(1);
    }

    for (size_t i_iter = 0; i_iter < n_iter; ++i_iter)
    {
        const size_t seed_iter = seed + i_iter;
        printf("Running markshare: m=%ld, seed=%ld, iter=%ld\n", m, seed_iter, i_iter);

        /* Generate/read instance. For now, random instances. */
        MarkShareFeas instance(m, seed_iter);

        const std::string filename = "markshare_m_" + std::to_string(6) + "_seed_" + std::to_string(seed_iter) + ".prb";
        instance.write_as_prb(filename);

        // MarkShareFeas instance(4, 10 * 3, {72, 30, 67, 47, 91, 83, 67, 35, 11, 35, 35, 73, 84, 46, 37, 44, 73, 33, 29, 82, 55, 1, 65, 21, 89, 54, 81, 67, 68, 43, 49, 11, 62, 38, 58, 5, 98, 20, 79, 89, 14, 14, 43, 57, 53, 51, 65, 66, 71, 19, 0, 11, 31, 39, 66, 95, 27, 35, 10, 80, 3, 3, 72, 49, 48, 46, 43, 48, 73, 42, 25, 10, 34, 64, 46, 37, 1, 10, 18, 38, 5, 18, 58, 52, 30, 82, 76, 33, 65, 56, 69, 75, 79, 93, 21, 59, 27, 29, 32, 57, 78, 37, 13, 65, 96, 0, 18, 24, 21, 90, 88, 49, 55, 0, 30, 27, 99, 48, 66, 79}, {809, 678, 592, 762});
        /* Solution is [1 0 1 0 1 0 0 0 1 0 0 1 0 1 0 1 0 1 0 1 1 1 0 1 1 0 1 0 0 1] */

        /* Solve the instance using one of the available algorithms. */

        /* Create the one dimensional subset sum problem. */
        const size_t rhs_subset_sum_1d = instance.b()[0];
        const std::vector<size_t> subset_sum_1d(instance.A().begin(), instance.A().begin() + instance.n());

        // two_list_algorithm(subset_sum_1d, rhs_subset_sum_1d);

        /* Shroeppel-Shamir */
        if (shroeppel_shamir(subset_sum_1d, rhs_subset_sum_1d, instance))
        {
            printf("Actually found something!\n");
            break;
        }
        else
            printf("Instance was infeasible .. \n");
    }

    ScopedProfiler::report();

    return 0;
}