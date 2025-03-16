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
#include <fstream>
#include <iostream>
#include <numeric>
#include <queue>
#include <utility>
#include <string>
#include <unordered_map>

#include <omp.h>

#ifdef WITH_GPU
#include "cuda_kernels.cuh"
#endif

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

template <typename T>
void print_vector(const std::vector<T> vec)
{
    std::cout << "[";
    for (auto &e : vec)
        std::cout << " " << e;
    std::cout << "]\n";
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

size_t countSetBits(size_t num)
{
    return __builtin_popcount(num);
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

template <typename T>
std::vector<T> apply_permutation(
    const std::vector<T> &vec,
    const std::vector<std::size_t> &p)
{
    std::vector<T> sorted_vec(vec.size());
    std::transform(p.begin(), p.end(), sorted_vec.begin(),
                   [&](std::size_t i)
                   { return vec[i]; });
    return sorted_vec;
}

std::pair<std::vector<size_t>, std::vector<std::vector<size_t>>> generate_subsets(const std::vector<size_t> &weights)
{
    size_t n = weights.size();
    size_t total_subsets = 1ULL << n; /* Total subsets is 2^n. */

    printf("Generating %ld possible subsets.\n", total_subsets);
    std::vector<size_t> set_weights(total_subsets, 0);
    std::vector<std::vector<size_t>> sets(total_subsets);

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
                set_weights[i + step + j] += weight;
                sets[i + step + j].push_back(pass);
            }
        }
    }

    return {set_weights, sets};
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

std::vector<size_t> extract_subset(const std::vector<size_t> &numbers, size_t index)
{
    std::vector<size_t> indices;

    size_t position = 0;
    while (index > 0)
    {
        if (index & 1)
            indices.push_back(numbers[position]);
        index >>= 1;
        ++position;
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
    auto [set1_weights, set1_subsets] = generate_subsets(list1);
    auto [set2_weights, set2_subsets] = generate_subsets(list2);

    printf("Sorting the lists!\n");

    /* Sort the subsets, set1_weights ascending, set2_weights descending. */
    auto asc_indices1 = sort_indices(set1_weights, true);
    auto desc_indices2 = sort_indices(set2_weights, false);

    printf("Finding solutions\n");
    /* Iterate the lists (one from the front and one from the back an generate solutions to the subset sum problem). */
    size_t pos_list1 = 0;
    size_t pos_list2 = 0;

    std::vector<std::pair<size_t, size_t>> solutions;

    while (pos_list1 < set1_weights.size() && pos_list2 < set2_weights.size())
    {
        const size_t elem_asc = set1_weights[asc_indices1[pos_list1]];
        const size_t elem_desc = set2_weights[desc_indices2[pos_list2]];

        const size_t elem_sum = elem_asc + elem_desc;

        if (elem_sum == rhs_subset_sum_1d)
        {
            printf("Found solution! \n");

            // check_solution()
            // solutions.push_back({elem_asc, elem_desc});
            // ++pos_list1;
            print_two_list_solution(asc_indices1[pos_list1], desc_indices2[pos_list2], list1, list2);
            return true;
        }
        else if (elem_sum < rhs_subset_sum_1d)
            ++pos_list1;
        else
            ++pos_list2;
    }

    return false;
}

void concat_vectors(std::vector<size_t> &concat_vec, size_t &concat_len, const std::vector<const std::vector<size_t> *> &vectors, const std::vector<size_t> offsets)
{
    assert(vectors.size() == offsets.size());

    concat_len = 0;
    for (const auto &vec : vectors)
    {
        concat_len += vec->size();
    }

    assert(concat_len <= concat_vec.size());

    size_t pos = 0;
    for (size_t ivec = 0; ivec < vectors.size(); ++ivec)
    {
        const auto &vec = *vectors[ivec];
        const auto offset = offsets[ivec];

        for (size_t num : vec)
        {
            concat_vec[pos] = (num + offset);
            ++pos;
        }
    }

    assert(pos == concat_len);
}

void print_four_list_solution(size_t index_list1, size_t index_list2, size_t index_list3, size_t index_list4, const std::vector<size_t> &list1, const std::vector<size_t> &list2, const std::vector<size_t> &list3, const std::vector<size_t> &list4)
{
    auto sum1 = print_subset_and_compute_sum(list1, index_list1);
    auto sum2 = print_subset_and_compute_sum(list2, index_list2);
    auto sum3 = print_subset_and_compute_sum(list3, index_list3);
    auto sum4 = print_subset_and_compute_sum(list4, index_list4);

    std::cout << "The sum is " << sum1 << " + " << sum2 << " + " << sum3 << " + " << sum4 << " = " << sum1 + sum2 + sum3 + sum4 << std::endl;
}

void append_solution_to_file(std::ofstream &sol_file, const std::vector<size_t> &numbers, size_t index)
{
    for (size_t i = 0; i < numbers.size(); ++i)
    {
        // Check if the i-th bit in the index is set
        if (index & (1ULL << i))
            sol_file << 1;
        else
            sol_file << 0;
    }
}

void write_four_list_solution_to_file(size_t index_list1, size_t index_list2, size_t index_list3, size_t index_list4, const std::vector<size_t> &list1, const std::vector<size_t> &list2, const std::vector<size_t> &list3, const std::vector<size_t> &list4, const std::string &instance_name)
{
    const std::string sol_name = instance_name + ".sol";
    printf("Writing solution to %s\n", sol_name.c_str());
    std::ofstream sol_file(sol_name);

    append_solution_to_file(sol_file, list1, index_list1);
    append_solution_to_file(sol_file, list2, index_list2);
    append_solution_to_file(sol_file, list3, index_list3);
    append_solution_to_file(sol_file, list4, index_list4);

    sol_file << std::endl;
}

std::pair<bool, std::pair<size_t, size_t>> evaluate_solutions_cpu(const MarkShareFeas &ms_inst, const std::vector<bool> &feas_q1, const std::vector<bool> &feas_q2, const std::vector<size_t> &scores_q1, const std::vector<size_t> &scores_q2, size_t n_q1, size_t n_q2)
{
    bool done = false;
    std::pair<size_t, size_t> solution_indices;

#pragma omp parallel shared(done)
#pragma omp for
    for (size_t iq1 = 0; iq1 < n_q1; ++iq1)
    {
        if (!feas_q1[iq1])
            continue;

        for (size_t iq2 = 0; iq2 < n_q2; ++iq2)
        {
            if (!feas_q2[iq2])
                continue;

            if (done)
            {
#pragma omp cancel for
            }

            if (!ms_inst.check_sum_feas(scores_q1.data() + iq1 * ms_inst.m(), scores_q2.data() + iq2 * ms_inst.m()))
                continue;

/* Found a feasible solution! */
#pragma omp critical
            {
                if (!done)
                {
                    done = true;
                    solution_indices = {iq1, iq2};
                }
            }
#pragma omp flush(done)
#pragma omp cancel for
        }
    }

    return {done, solution_indices};
}

// Custom hash function to encode the vector as an __int128_t
__int128_t encode_vector(const size_t *vec, size_t len)
{
    constexpr size_t base = 10000; // 4 digits per value
    __int128_t hash_value = 0;

    for (size_t i_vec = 0; i_vec < len; ++i_vec)
        hash_value = hash_value * base + vec[i_vec];

    return hash_value;
}

std::pair<bool, std::pair<size_t, size_t>> evaluate_solutions_cpu_hashing(const MarkShareFeas &ms_inst, const std::vector<size_t> &scores_q1, const std::vector<size_t> &scores_q2, size_t n_q1, size_t n_q2)
{
    const size_t len_vec = ms_inst.m();
    std::unordered_map<__int128_t, size_t> hashTable;
    std::vector<size_t> needed(len_vec);
    bool done = false;
    std::pair<size_t, size_t> solution_indices = {n_q1, n_q2};

    auto profiler = std::make_unique<ScopedProfiler>("Eval CPU: Hash table setup  ");
    for (size_t i_q1 = 0; i_q1 < n_q1; ++i_q1)
    {
        for (size_t j = 0; j < len_vec; ++j)
        {
            needed[j] = ms_inst.b()[j] - scores_q1[i_q1 * len_vec + j];
        }

        __int128_t needed_key = encode_vector(needed.data(), len_vec);
        hashTable[needed_key] = i_q1;
    }

    profiler = std::make_unique<ScopedProfiler>("Eval CPU: Hash table search ");
#pragma omp parallel shared(done)
#pragma omp for
    for (size_t i_q2 = 0; i_q2 < n_q2; ++i_q2)
    {
        if (done)
        {
#pragma omp cancel for
        }

        __int128_t given_key = encode_vector(scores_q2.data() + i_q2 * len_vec, len_vec);

        if (hashTable.find(given_key) != hashTable.end())
        {
            /* Found a feasible solution! */
#pragma omp critical
            {
                if (!done)
                {
                    const size_t i_q1 = hashTable[given_key];
                    done = true;
                    solution_indices = {i_q1, i_q2};
                }
            }
#pragma omp flush(done)
#pragma omp cancel for
        }
    }

    profiler.reset();

    return {done, solution_indices};
}

void compute_scores_cpu(const MarkShareFeas &ms_inst, std::vector<size_t> &scores, const std::vector<std::vector<size_t>> &subsets, size_t offset)
{
#pragma omp parallel for
    for (size_t i = 0; i < subsets.size(); ++i)
    {
        ms_inst.compute_value(subsets[i], offset, scores.data() + i * ms_inst.m());
    }
}

void combine_scores_cpu(const std::vector<size_t> &set1_scores, const std::vector<size_t> &set2_scores, size_t m_rows, const std::vector<PairsTuple> &tuples, std::vector<size_t> &scores_pairs)
{
#pragma omp parallel for
    for (size_t i_tuple = 0; i_tuple < tuples.size(); ++i_tuple)
    {
        auto first = tuples[i_tuple].pairs_first;
        auto pair_second_beg = tuples[i_tuple].pairs_second_beg;
        auto pair_second_end = pair_second_beg + tuples[i_tuple].pairs_n_second;
        auto pair_offset = tuples[i_tuple].pairs_offset;

        for (size_t second = pair_second_beg; second < pair_second_end; ++second)
        {
            for (size_t i_row = 0; i_row < m_rows; ++i_row)
                scores_pairs[pair_offset * m_rows + i_row] = set1_scores[first * m_rows + i_row] + set2_scores[second * m_rows + i_row];
            ++pair_offset;
        }
    }
}

void print_info_line(const GpuData &gpu_data, size_t i_iter, double time, size_t score1, size_t score2, size_t n_q1, size_t n_q2)
{
    bool print = false;
    if (i_iter < 1000000)
        print = true;
    else if (i_iter < 100 && i_iter % 10 == 0)
        print = true;
    else if (i_iter % 100 == 0)
        print = true;

    if (print)
    {
        const double n_gb = gpu_data.get_gb_allocated();
        printf("%5ld %8.2fs [%.6f GB]: %6ld + %6ld; %ld x %ld = %ld possible solutions\n", i_iter, time, n_gb, score1, score2, n_q1, n_q2, n_q1 * n_q2);
    }
}

bool verify_solution(const std::pair<size_t, size_t> &solution, const std::vector<PairsTuple> &same_score_q1, const std::vector<PairsTuple> &same_score_q2,
                     const std::vector<std::vector<size_t>> &set1_subsets, const std::vector<std::vector<size_t>> &set2_subsets_sorted_asc, const std::vector<std::vector<size_t>> &set3_subsets, const std::vector<std::vector<size_t>> &set4_subsets_sorted_desc, const std::vector<size_t> &subset_sum_1d, const std::vector<size_t> &offsets, const MarkShareFeas &ms_inst)
{
    /* Print and verify the solution! */
    /* Get the correct pairs. */
    size_t pos_q1 = 0;
    size_t pos_q2 = 0;

    while (solution.first >= same_score_q1[pos_q1].pairs_offset + same_score_q1[pos_q1].pairs_n_second)
        ++pos_q1;
    while (solution.second >= same_score_q2[pos_q2].pairs_offset + same_score_q2[pos_q2].pairs_n_second)
        ++pos_q2;

    assert(solution.first - same_score_q1[pos_q1].pairs_offset < same_score_q1[pos_q1].pairs_n_second);
    assert(solution.second - same_score_q2[pos_q2].pairs_offset < same_score_q2[pos_q2].pairs_n_second);

    const size_t pair_q1_second = same_score_q1[pos_q1].pairs_second_beg + solution.first - same_score_q1[pos_q1].pairs_offset;
    const size_t pair_q2_second = same_score_q2[pos_q2].pairs_second_beg + solution.second - same_score_q2[pos_q2].pairs_offset;
    std::pair<size_t, size_t> pair_q1 = {same_score_q1[pos_q1].pairs_first, pair_q1_second};
    std::pair<size_t, size_t> pair_q2 = {same_score_q2[pos_q2].pairs_first, pair_q2_second};

    std::vector<size_t> solution_1d(subset_sum_1d.size());

    const std::vector<const std::vector<size_t> *> vectors = {&set1_subsets[pair_q1.first], &set2_subsets_sorted_asc[pair_q1.second], &set3_subsets[pair_q2.first], &set4_subsets_sorted_desc[pair_q2.second]};

    size_t len;
    concat_vectors(solution_1d, len, vectors, offsets);

    /* We found a solution. Construct it, print it, and return. */
    if (!ms_inst.is_solution_feasible(solution_1d, len))
    {
        printf("Error, solution is not feasible!\n");
        return false;
    }

    return true;
}

bool shroeppel_shamir(const std::vector<size_t> &subset_sum_1d, size_t rhs_subset_sum_1d, const MarkShareFeas &ms_inst, bool run_on_gpu, const std::string &instance_name)
{
    const size_t split_index1 = subset_sum_1d.size() / 4;
    const size_t split_index2 = subset_sum_1d.size() / 2;
    const size_t split_index3 = 3 * subset_sum_1d.size() / 4;
    printf("Splitting sets into [0, %ld]; [%ld, %ld]; [%ld, %ld]; [%ld, %ld]\n", split_index1 - 1, split_index1, split_index2 - 1, split_index2, split_index3 - 1, split_index3, subset_sum_1d.size());

    std::cout << "Running with " << omp_get_max_threads() << " threads" << std::endl;
    auto profiler = std::make_unique<ScopedProfiler>("Setup time                  ");
    auto profilerTotal = std::make_unique<ScopedProfiler>("Solution time               ");

    /* Get 4 sublists. */
    std::vector<size_t> list1(subset_sum_1d.begin(), subset_sum_1d.begin() + split_index1);
    std::vector<size_t> list2(subset_sum_1d.begin() + split_index1, subset_sum_1d.begin() + split_index2);
    std::vector<size_t> list3(subset_sum_1d.begin() + split_index2, subset_sum_1d.begin() + split_index3);
    std::vector<size_t> list4(subset_sum_1d.begin() + split_index3, subset_sum_1d.end());
    assert(list1.size() + list2.size() + list3.size() + list4.size() == subset_sum_1d.size());

    const std::vector<size_t> offsets = {0, list1.size(), list1.size() + list2.size(), list1.size() + list2.size() + list3.size()};
    const std::vector<size_t> offsetsQ1 = {0, list1.size()};
    const std::vector<size_t> offsetsQ2 = {list1.size() + list2.size(), list1.size() + list2.size() + list3.size()};

    // TODO: generate subsets and extract sets can possibly be combined.
    auto [set1_weights, set1_subsets] = generate_subsets(list1);
    auto [set2_weights, set2_subsets] = generate_subsets(list2);
    auto [set3_weights, set3_subsets] = generate_subsets(list3);
    auto [set4_weights, set4_subsets] = generate_subsets(list4);

    /* Sort set2_weights ascending, set4_weights descending. */
    auto asc_indices_set2_weights = sort_indices(set2_weights, true);
    auto desc_indices_set4_weights = sort_indices(set4_weights, false);

    const auto set2_weights_sorted_asc = apply_permutation(set2_weights, asc_indices_set2_weights);
    const auto set2_subsets_sorted_asc = apply_permutation(set2_subsets, asc_indices_set2_weights);

    const auto set4_weights_sorted_desc = apply_permutation(set4_weights, desc_indices_set4_weights);
    const auto set4_subsets_sorted_desc = apply_permutation(set4_subsets, desc_indices_set4_weights);

    std::vector<size_t> set1_scores(set1_subsets.size() * ms_inst.m());
    std::vector<size_t> set2_scores_sorted_asc(set2_subsets.size() * ms_inst.m());
    std::vector<size_t> set3_scores(set3_subsets.size() * ms_inst.m());
    std::vector<size_t> set4_scores_sorted_desc(set4_subsets.size() * ms_inst.m());

    compute_scores_cpu(ms_inst, set1_scores, set1_subsets, offsets[0]);
    compute_scores_cpu(ms_inst, set2_scores_sorted_asc, set2_subsets_sorted_asc, offsets[1]);
    compute_scores_cpu(ms_inst, set3_scores, set3_subsets, offsets[2]);
    compute_scores_cpu(ms_inst, set4_scores_sorted_desc, set4_subsets_sorted_desc, offsets[3]);

    GpuData gpu_data(ms_inst, set1_scores, set2_scores_sorted_asc, set3_scores, set4_scores_sorted_desc);

    /* Create the priority queues q1 consisting of pairs {(i, 0) | i \in set1_weights} and q2 consisting of {(i, 0) | i \in set3_weights}. The priority/score for a pair (i, j)
     * is given set1_weights[i] + set2_weights[j] if the pair is in q1 and set3_weights[i] + set4_weights[j] if the pair is in q2. */

    /* Compare returns true if the first argument comes BEFORE the second argument. Since however the priority queue outputs the largest element first,
     * we have to flip the > signs. */
    auto min_cmp = [&](std::pair<size_t, size_t> a1, std::pair<size_t, size_t> a2) -> bool
    {
        return set1_weights[a1.first] + set2_weights_sorted_asc[a1.second] > set1_weights[a2.first] + set2_weights_sorted_asc[a2.second];
    };

    auto max_cmp = [&](std::pair<size_t, size_t> a1, std::pair<size_t, size_t> a2) -> bool
    {
        return set3_weights[a1.first] + set4_weights_sorted_desc[a1.second] < set3_weights[a2.first] + set4_weights_sorted_desc[a2.second];
    };

    /* Vectors used to count the number of elements extracted from q1/q2 with the same solution value. */
    /* Each tuple {a, b, c, d} will describe the range of pairs <a, b> ... <a, b + c - 1>; d denotes the offset of the pairs within a list of all pairs. */
    std::vector<PairsTuple> same_score_q1;
    same_score_q1.reserve(100000);
    std::vector<PairsTuple> same_score_q2;
    same_score_q2.reserve(100000);

    std::vector<std::pair<size_t, size_t>> heap1;
    heap1.reserve(set1_weights.size());
    std::vector<std::pair<size_t, size_t>> heap2;
    heap2.reserve(set3_weights.size());

    // std::priority_queue<std::pair<size_t, size_t>, std::vector<std::pair<size_t, size_t>>, decltype(min_cmp)> q1(min_cmp);
    // std::priority_queue<std::pair<size_t, size_t>, std::vector<std::pair<size_t, size_t>>, decltype(max_cmp)> q2(max_cmp);
    // TODO: the initial insert can likely be improved by simple sorting.

    for (size_t i = 0; i < set1_weights.size(); ++i)
    {
        /* If already the sum of these 2 elements is greater than the right hand side we can skip them. Subsequent combinations (e.g. with higher pos_subset2)
         * will only be even larger. */
        if (set1_weights[i] + set2_weights_sorted_asc[0] <= rhs_subset_sum_1d)
            heap1.emplace_back(i, 0);
    }

    for (size_t i = 0; i < set3_weights.size(); ++i)
        heap2.emplace_back(i, 0);

    std::make_heap(heap1.begin(), heap1.end(), min_cmp);
    std::make_heap(heap2.begin(), heap2.end(), max_cmp);

    printf("Running the search loop\n\n");

    profiler = std::make_unique<ScopedProfiler>("List traversal              ");

    size_t i_iter_checking = 0;
    while (!heap1.empty() && !heap2.empty())
    {
        const auto pair1 = heap1.front();
        const auto pair2 = heap2.front();

        /* score_pair1 is the currently lowest score in {set1_weights, set2_weights} we are still considering */
        const size_t score_pair1 = set1_weights[pair1.first] + set2_weights_sorted_asc[pair1.second];
        /* score_pair2 is the currently highest score in {set3_weights, set4_weights} we are still considering */
        const size_t score_pair2 = set3_weights[pair2.first] + set4_weights_sorted_desc[pair2.second];

        const size_t score = score_pair1 + score_pair2;

        if (score == rhs_subset_sum_1d)
        {
            ++i_iter_checking;
            /* Clear vectors but keep their old capacity. */
            same_score_q1.clear();
            same_score_q2.clear();
            size_t n_q1 = 0;
            size_t n_q2 = 0;

            auto profiler_inside_loop = std::make_unique<ScopedProfiler>("Candidate extraction        ");

            /* For each element a in q1 with score(a) == score_pair1, collect all solutions. */
#pragma omp parallel sections num_threads(2)
            {
#pragma omp section
                {
                    while (!heap1.empty() && set1_weights[heap1.front().first] + set2_weights_sorted_asc[heap1.front().second] == score_pair1)
                    {
                        const auto pair1_same_score = heap1.front();
                        std::pop_heap(heap1.begin(), heap1.end(), min_cmp);
                        heap1.pop_back();

                        const size_t pos_set2_weights_beg = pair1_same_score.second;
                        size_t pos_set2_weights_end = pos_set2_weights_beg;

                        /* Iterate the second elements. */
                        const auto pos2_val = set2_weights_sorted_asc[pos_set2_weights_end];

                        while (pos_set2_weights_end < set2_weights.size() && pos2_val == set2_weights_sorted_asc[pos_set2_weights_end])
                            ++pos_set2_weights_end;

                        size_t n_pairs = pos_set2_weights_end - pos_set2_weights_beg;
                        same_score_q1.emplace_back(pair1_same_score.first, pos_set2_weights_beg, n_pairs, n_q1);
                        n_q1 += n_pairs;

                        if (pos_set2_weights_end < set2_weights.size())
                        {
                            assert(score_pair1 < set1_weights[pair1_same_score.first] + set2_weights_sorted_asc[pos_set2_weights_end]);
                            heap1.emplace_back(pair1_same_score.first, pos_set2_weights_end);
                            std::push_heap(heap1.begin(), heap1.end(), min_cmp);
                        }
                    }
                }
#pragma omp section
                {
                    /* For each element a in q2 with score(a) == score_pair2, collect all solutions. */
                    while (!heap2.empty() && set3_weights[heap2.front().first] + set4_weights_sorted_desc[heap2.front().second] == score_pair2)
                    {
                        const auto pair2_same_score = heap2.front();
                        std::pop_heap(heap2.begin(), heap2.end(), max_cmp);
                        heap2.pop_back();

                        const size_t pos_set4_weights_beg = pair2_same_score.second;
                        size_t pos_set4_weights_end = pos_set4_weights_beg;

                        /* Iterate the second elements. */
                        const auto pos4_val = set4_weights_sorted_desc[pos_set4_weights_end];

                        while (pos_set4_weights_end < set4_weights.size() && pos4_val == set4_weights_sorted_desc[pos_set4_weights_end])
                            ++pos_set4_weights_end;

                        const size_t n_pairs = pos_set4_weights_end - pos_set4_weights_beg;

                        same_score_q2.emplace_back(pair2_same_score.first, pos_set4_weights_beg, n_pairs, n_q2);
                        n_q2 += n_pairs;

                        if (pos_set4_weights_end < set4_weights.size())
                        {
                            assert(score_pair2 > set3_weights[pair2_same_score.first] + set4_weights_sorted_desc[pos_set4_weights_end]);
                            heap2.emplace_back(pair2_same_score.first, pos_set4_weights_end);
                            std::push_heap(heap2.begin(), heap2.end(), max_cmp);
                        }
                    }
                }
            }
            profiler_inside_loop = std::make_unique<ScopedProfiler>("Combine scores              ");

            print_info_line(gpu_data, i_iter_checking, profilerTotal->elapsed(), score_pair1, score_pair2, n_q1, n_q2);

            bool found = false;
            std::pair<size_t, size_t> solution;

            if (run_on_gpu)
            {
                combine_and_encode_tuples_gpu(gpu_data, same_score_q1, same_score_q2, n_q1, n_q2);

                profiler_inside_loop = std::make_unique<ScopedProfiler>("Evaluate solutions GPU      ");
                const std::vector<size_t> hashes = find_equal_hashes(gpu_data);

                if (!hashes.empty())
                {
                    /* Retrieve the actual solution. We have to copy encode our arrays once more and look for the hash afterwards. */
                    combine_and_encode_tuples_gpu(gpu_data, same_score_q1, same_score_q2, n_q1, n_q2);

                    const std::vector<std::pair<size_t, size_t>> candidates = find_hash_positions_gpu(gpu_data, hashes, n_q1, n_q2);

                    /* Check all potential solutions. */
                    for (const auto &solution_cand : candidates)
                    {
                        const bool feasible = verify_solution(solution_cand, same_score_q1, same_score_q2, set1_subsets, set2_subsets_sorted_asc, set3_subsets, set4_subsets_sorted_desc, subset_sum_1d, offsets, ms_inst);

                        if (feasible)
                        {
                            solution = solution_cand;
                            found = true;
                            break;
                        }
                    }
                }
                profiler_inside_loop.reset();
            }
            else
            {
                /* Precompute the partial scores. */
                std::vector<size_t> buffered_scores_q1(ms_inst.m() * n_q1);
                std::vector<size_t> buffered_scores_q2(ms_inst.m() * n_q2);

                combine_scores_cpu(set1_scores, set2_scores_sorted_asc, ms_inst.m(), same_score_q1, buffered_scores_q1);
                combine_scores_cpu(set3_scores, set4_scores_sorted_desc, ms_inst.m(), same_score_q2, buffered_scores_q2);

                profiler_inside_loop = std::make_unique<ScopedProfiler>("Evaluate solutions CPU      ");

                auto [done, solution_indices] = evaluate_solutions_cpu_hashing(ms_inst, buffered_scores_q1, buffered_scores_q2, n_q1, n_q2);

                found = done;
                solution = solution_indices;
                profiler_inside_loop.reset();
            }

            if (!found)
                continue;

            /* Print and verify the solution! */
            /* Get the correct pairs. */
            size_t pos_q1 = 0;
            size_t pos_q2 = 0;

            while (solution.first >= same_score_q1[pos_q1].pairs_offset + same_score_q1[pos_q1].pairs_n_second)
                ++pos_q1;
            while (solution.second >= same_score_q2[pos_q2].pairs_offset + same_score_q2[pos_q2].pairs_n_second)
                ++pos_q2;

            assert(solution.first - same_score_q1[pos_q1].pairs_offset < same_score_q1[pos_q1].pairs_n_second);
            assert(solution.second - same_score_q2[pos_q2].pairs_offset < same_score_q2[pos_q2].pairs_n_second);

            const size_t pair_q1_second = same_score_q1[pos_q1].pairs_second_beg + solution.first - same_score_q1[pos_q1].pairs_offset;
            const size_t pair_q2_second = same_score_q2[pos_q2].pairs_second_beg + solution.second - same_score_q2[pos_q2].pairs_offset;
            std::pair<size_t, size_t> pair_q1 = {same_score_q1[pos_q1].pairs_first, pair_q1_second};
            std::pair<size_t, size_t> pair_q2 = {same_score_q2[pos_q2].pairs_first, pair_q2_second};

            std::vector<size_t> solution_1d(subset_sum_1d.size());

            assert(set1_weights[pair_q1.first] + set2_weights_sorted_asc[pair_q1.second] + set3_weights[pair_q2.first] + set4_weights_sorted_desc[pair_q2.second] == rhs_subset_sum_1d);

            const std::vector<const std::vector<size_t> *> vectors = {&set1_subsets[pair_q1.first], &set2_subsets_sorted_asc[pair_q1.second], &set3_subsets[pair_q2.first], &set4_subsets_sorted_desc[pair_q2.second]};

            size_t len;
            concat_vectors(solution_1d, len, vectors, offsets);

            /* We found a solution. Construct it, print it, and return. */
            if (!ms_inst.is_solution_feasible(solution_1d, len))
            {
                printf("Error, solution is not feasible!\n");
                exit(1);
            }

            printf("Found market share solution from SS-Algorithm!\n");
            print_four_list_solution(pair_q1.first, asc_indices_set2_weights[pair_q1.second], pair_q2.first, desc_indices_set4_weights[pair_q2.second], list1, list2, list3, list4);

            if (!instance_name.empty())
                write_four_list_solution_to_file(pair_q1.first, asc_indices_set2_weights[pair_q1.second], pair_q2.first, desc_indices_set4_weights[pair_q2.second], list1, list2, list3, list4, instance_name);
            return true;
        }
        else if (score < rhs_subset_sum_1d)
        {
            size_t pos_set2_weights = pair1.second;

            std::pop_heap(heap1.begin(), heap1.end(), min_cmp);
            heap1.pop_back();

            ++pos_set2_weights;

            while (pos_set2_weights + 1 < set2_weights.size() && (set2_weights_sorted_asc[pos_set2_weights] == set2_weights_sorted_asc[pair1.second] || (set1_weights[pair1.first] + set2_weights_sorted_asc[pos_set2_weights] + score_pair2) < rhs_subset_sum_1d))
                ++pos_set2_weights;

            /* Again, the element in q1 can only increase (or stay equal). So ignore elements that are already too big. */
            if (pos_set2_weights < set2_weights.size() && set1_weights[pair1.first] + set2_weights_sorted_asc[pos_set2_weights] <= rhs_subset_sum_1d)
            {
                heap1.emplace_back(pair1.first, pos_set2_weights);
                std::push_heap(heap1.begin(), heap1.end(), min_cmp);
            }
        }
        else if (score > rhs_subset_sum_1d)
        {
            size_t pos_set4_weights = pair2.second;

            std::pop_heap(heap2.begin(), heap2.end(), max_cmp);
            heap2.pop_back();

            ++pos_set4_weights;

            /* Skip all entries in set4_weights until we find a smaller one. */
            while (pos_set4_weights + 1 < set4_weights.size() && (set4_weights_sorted_desc[pos_set4_weights] == set4_weights_sorted_desc[pair2.second] || (score_pair1 + set3_weights[pair2.first] + set4_weights_sorted_desc[pos_set4_weights]) > rhs_subset_sum_1d))
                ++pos_set4_weights;

            if (pos_set4_weights < set4_weights.size() && set3_weights[pair2.first] + set4_weights_sorted_desc[pos_set4_weights] <= rhs_subset_sum_1d)
            {
                heap2.emplace_back(pair2.first, pos_set4_weights);
                std::push_heap(heap2.begin(), heap2.end(), max_cmp);
            }
        }
    }

    profiler.reset();

    return false;
}

std::string get_filename_without_extension(const std::string &filePath)
{
    // Find the last path separator '/'
    size_t lastSlash = filePath.find_last_of('/');
    size_t start = (lastSlash == std::string::npos) ? 0 : lastSlash + 1;

    // Find the last dot after the last slash
    size_t lastDot = filePath.find_last_of('.');
    size_t end = (lastDot == std::string::npos || lastDot < start) ? filePath.length() : lastDot;

    // Extract the filename without extension
    return filePath.substr(start, end - start);
}

int main(int argc, char *argv[])
{
    argparse::ArgumentParser program("markshare");

    std::string path = "";
    size_t n_iter = 1;
    size_t seed = 2025;
    size_t m = 5;
    size_t n = 0;

    program.add_argument("-m", "--m")
        .store_into(m)
        .help("Number of rows of the markshare problem.");

    program.add_argument("-n", "--n")
        .store_into(n)
        .help("Number of columns of the markshare problem. Set to (m - 1) * 10 if not given. ")
        .default_value(0);

    program.add_argument("-s", "--seed")
        .store_into(seed)
        .help("Random seed for instance generation.")
        .default_value(2025);

    program.add_argument("-i", "--iter")
        .store_into(n_iter)
        .help("Number of problems to solve. Seed for problem of iteration i (starting from 0) is seed + i.")
        .default_value(1);

    program.add_argument("--gpu")
        .help("Run validation on GPU")
        .flag();

    program.add_argument("-f", "--file")
        .store_into(path)
        .help("Supply instance path to read instance from. Overrides '-m', '-n', and '-i'");

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

    /* Adjust n. */
    if (n == 0)
        n = (m - 1) * 10;

    for (size_t i_iter = 0; i_iter < n_iter; ++i_iter)
    {
        std::string instance_name{};
        const size_t seed_iter = seed + i_iter;

        MarkShareFeas instance;
        /* Generate/read instance. For now, random instances. */
        if (!path.empty())
        {
            instance_name = get_filename_without_extension(path);
            printf("Reading instance from file %s; instance_name %s\n", path.c_str(), instance_name.c_str());
            instance = MarkShareFeas(path);
        }
        else
        {
            instance = MarkShareFeas(m, n, seed_iter);
            instance_name = "markshare_m_" + std::to_string(m) + "_n_" + std::to_string(n) + "_seed_" + std::to_string(seed_iter);
            instance.write_as_prb(instance_name + ".prb");
        }
        printf("Running markshare: m=%ld, n=%ld, seed=%ld, iter=%ld, nthread=%d\n", instance.m(), instance.n(), seed_iter, i_iter, omp_get_max_threads());
        instance.print();

        // MarkShareFeas instance(4, 10 * 3, {72, 30, 67, 47, 91, 83, 67, 35, 11, 35, 35, 73, 84, 46, 37, 44, 73, 33, 29, 82, 55, 1, 65, 21, 89, 54, 81, 67, 68, 43, 49, 11, 62, 38, 58, 5, 98, 20, 79, 89, 14, 14, 43, 57, 53, 51, 65, 66, 71, 19, 0, 11, 31, 39, 66, 95, 27, 35, 10, 80, 3, 3, 72, 49, 48, 46, 43, 48, 73, 42, 25, 10, 34, 64, 46, 37, 1, 10, 18, 38, 5, 18, 58, 52, 30, 82, 76, 33, 65, 56, 69, 75, 79, 93, 21, 59, 27, 29, 32, 57, 78, 37, 13, 65, 96, 0, 18, 24, 21, 90, 88, 49, 55, 0, 30, 27, 99, 48, 66, 79}, {809, 678, 592, 762});
        /* Solution is [1 0 1 0 1 0 0 0 1 0 0 1 0 1 0 1 0 1 0 1 1 1 0 1 1 0 1 0 0 1] */

        /* Solve the instance using one of the available algorithms. */

        /* Create the one dimensional subset sum problem. */
        const size_t rhs_subset_sum_1d = instance.b()[0];
        const std::vector<size_t> subset_sum_1d(instance.A().begin(), instance.A().begin() + instance.n());

        // two_list_algorithm(subset_sum_1d, rhs_subset_sum_1d);

        /* Shroeppel-Shamir */
        if (shroeppel_shamir(subset_sum_1d, rhs_subset_sum_1d, instance, program["--gpu"] == true, instance_name))
            printf("Found feasible solution!\n");
        else
            printf("Instance was infeasible .. \n");
    }

    ScopedProfiler::report();

    return 0;
}