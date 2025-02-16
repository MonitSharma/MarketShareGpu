#include "cuda_kernels.cuh"

#include "profiler.hpp"

#include <cuda_runtime.h>

#include <thrust/host_vector.h>
#include <thrust/binary_search.h>
#include <thrust/device_vector.h>
#include <thrust/sequence.h>
#include <thrust/sort.h>

#include <iostream>

template <typename T>
T *copy_to_device(const std::vector<T> &host_vec)
{
    T *device_ptr;
    cudaMalloc(&device_ptr, host_vec.size() * sizeof(T));
    cudaMemcpy(device_ptr, host_vec.data(), host_vec.size() * sizeof(T), cudaMemcpyHostToDevice);
    return device_ptr;
}

static void copy_subsets(const std::vector<std::vector<size_t>>& set_subsets, size_t** set_beg_gpu, size_t** sets_gpu)
{
    size_t subsets_size_total = 0;
    std::vector<size_t> set_beg;
    set_beg.reserve(set_subsets.size() + 1);

    for (const auto & subset : set_subsets)
    {
        set_beg.push_back(subsets_size_total);
        subsets_size_total += subset.size();
    }
    set_beg.push_back(subsets_size_total);

    std::vector<size_t> sets_flattened;
    sets_flattened.reserve(subsets_size_total);

    for (const auto& subset : set_subsets)
    {
        sets_flattened.insert(sets_flattened.end(), subset.begin(), subset.end());
    }

    assert(sets_flattened.size() == subsets_size_total);

    *set_beg_gpu = copy_to_device(set_beg);
    *sets_gpu = copy_to_device(sets_flattened);
}

GpuData::GpuData(const MarkShareFeas &ms_inst, const std::vector<std::vector<size_t>>& set1_subsets, const std::vector<std::vector<size_t>>& set2_subsets, const std::vector<std::vector<size_t>>& set3_subsets, const std::vector<std::vector<size_t>>& set4_subsets, const std::vector<size_t>& asc_indicies_set2_weights, const std::vector<size_t>& desc_indicies_set4_weights) : m_rows(ms_inst.m()), n_cols(ms_inst.n())
{
    this->matrix = copy_to_device(ms_inst.A());
    this->rhs = copy_to_device(ms_inst.b());

    copy_subsets(set1_subsets, &this->set1_subsets, &this->set1_subsets_beg);
    copy_subsets(set2_subsets, &this->set2_subsets, &this->set2_subsets_beg);
    copy_subsets(set3_subsets, &this->set3_subsets, &this->set3_subsets_beg);
    copy_subsets(set4_subsets, &this->set4_subsets, &this->set4_subsets_beg);

    this->asc_indicies_set2_weights = copy_to_device(asc_indicies_set2_weights);
    this->desc_indicies_set4_weights = copy_to_device(desc_indicies_set4_weights);
}

GpuData::~GpuData()
{
    cudaFree(matrix);
    cudaFree(rhs);
}

__global__ void check_sums(const size_t *val1, const size_t *val2, const size_t *rhs, size_t *solution, size_t n_val1, size_t n_val2, size_t m_rhs)
{
    int i1 = blockIdx.x * blockDim.x + threadIdx.x; // Thread for index i1
    int i2 = blockIdx.y * blockDim.y + threadIdx.y; // Thread for index i2

    if (i1 < n_val1 && i2 < n_val2)
    {
        bool feas = true;

        for (int j = 1; j < m_rhs; ++j)
        {
            const size_t sum = val1[i1 * m_rhs + j] + val2[i2 * m_rhs + j];
            if (sum != rhs[j])
            {
                feas = false;
                break;
            }
        }

        if (feas)
        {
            *solution = i1 * n_val2 + i2;
            return;
        }
    }
}

__global__ void binarySearchKeys(const __int128_t *sorted_keys, size_t num_sorted_keys,
                                 const __int128_t *query_keys, size_t *results, size_t num_query_keys)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_query_keys)
        return;

    __int128_t query_key = query_keys[idx];
    int64_t left = 0, right = num_sorted_keys - 1;

    // Binary search
    while (left <= right)
    {
        int64_t mid = left + (right - left) / 2;
        if (sorted_keys[mid] == query_key)
        {
            results[idx] = mid;
            return;
        }
        else if (sorted_keys[mid] < query_key)
        {
            left = mid + 1;
        }
        else
        {
            right = mid - 1;
        }
    }
}

__global__ void encodeVectors(const size_t *vectors, size_t num_vectors, size_t vector_size, __int128_t *encoded_keys)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_vectors)
        return;

    __int128_t key = 0;
    size_t base = 10000; // Maximum value per vector element
    for (size_t i = 0; i < vector_size; i++)
    {
        key = key * base + vectors[idx * vector_size + i];
    }
    encoded_keys[idx] = key;
}

std::pair<bool, std::pair<size_t, size_t>> evaluate_solutions_gpu(const GpuData &gpu_data, const std::vector<size_t> &scores_q1, const std::vector<size_t> &scores_q2, size_t n_q1, size_t n_q2)
{
    size_t result;
    size_t* d_rhs = gpu_data.rhs;
    size_t m = gpu_data.m_rows;

    size_t *d_scores_q1 = copy_to_device(scores_q1);
    size_t *d_scores_q2 = copy_to_device(scores_q2);

    size_t *d_solution;
    size_t sol_invalid = scores_q1.size() * scores_q1.size() + 1;
    cudaMalloc(&d_solution, sizeof(size_t));
    cudaMemcpy(d_solution, &sol_invalid, sizeof(size_t), cudaMemcpyHostToDevice);

    // Define grid and block sizes
    dim3 blockDim(32, 32);                                                                    // Threads per block
    dim3 gridDim((n_q1 + blockDim.x - 1) / blockDim.x, (n_q2 + blockDim.y - 1) / blockDim.y); // Blocks per grid

    // Launch kernel
    check_sums<<<gridDim, blockDim>>>(d_scores_q1, d_scores_q2, d_rhs, d_solution, n_q1, n_q2, m);

    // Copy result back to host
    cudaMemcpy(&result, d_solution, sizeof(size_t), cudaMemcpyDeviceToHost);

    cudaFree(d_scores_q1);
    cudaFree(d_scores_q2);
    cudaFree(d_rhs);
    cudaFree(d_solution);

    cudaDeviceSynchronize();
    if (result != sol_invalid)
    {
        printf("GPU found solution!\n");
        size_t i_q1 = result / n_q2;
        size_t i_q2 = result % n_q2;

        return {true, {i_q1, i_q2}};
    }
    else
    {
        return {false, {n_q1, n_q2}};
    }
}

__global__ void compute_required(const size_t* __restrict__ rhs, const size_t* __restrict__ scores_q1, size_t* __restrict__ required, size_t m, size_t n_q1)
{
    // Get the index of the current thread in the grid
    size_t i_q1 = blockIdx.x * blockDim.x + threadIdx.x;  // Corresponds to n_q1
    size_t i_m = threadIdx.y;                            // Corresponds to m

    // Check bounds
    if (i_q1 < n_q1 && i_m < m) {
        // Calculate the index
        size_t idx = i_q1 * m + i_m;
        // Perform the computation
        required[idx] = rhs[i_m] - scores_q1[idx];
    }
}

void compute_scores_gpu(const GpuData& gpu_data, std::vector<size_t>& scores, const std::vector<std::pair<size_t, size_t>>& pairs, const std::vector<std::vector<size_t>>& pair_first_subsets,
    const std::vector<std::vector<size_t>>& pair_second_subsets, const std::vector<size_t> pair_second_map, const std::vector<size_t> offsets)
{
    // size_t* d_scores = copy_to_device(scores);
    // size_t* d_pair = copy_to_device(pairs);
    // copy_to_device
    // copy_to_device

}

std::pair<bool, std::pair<size_t, size_t>> evaluate_solutions_gpu_hashing(const GpuData& gpu_data, const std::vector<size_t> &scores_q1, const std::vector<size_t> &scores_q2, size_t n_q1, size_t n_q2)
{
    size_t* d_rhs = gpu_data.rhs;
    size_t m_rows = gpu_data.m_rows;

    auto profiler = std::make_unique<ScopedProfiler>("GPU hash setup");
    thrust::device_vector<size_t> d_required(scores_q1.size());
    thrust::device_vector<size_t> d_scores_q1(scores_q1);

    profiler = std::make_unique<ScopedProfiler>("GPU compute required");

    // Configure grid and block sizes
    dim3 blockDim(128, m_rows);  // 128 threads for i_q1, and each thread handles one value of m
    dim3 gridDim((n_q1 + blockDim.x - 1) / blockDim.x);
    compute_required<<<gridDim, blockDim>>>(d_rhs, thrust::raw_pointer_cast(d_scores_q1.data()), thrust::raw_pointer_cast(d_required.data()), m_rows, n_q1);

    profiler = std::make_unique<ScopedProfiler>("GPU data setup");

    thrust::device_vector<size_t> d_scores_q2(scores_q2);

    // THE ALGORITHM!
    // Allocate encoded key arrays on the GPU
    thrust::device_vector<__int128_t> d_keys1(n_q1);
    thrust::device_vector<__int128_t> d_keys2(n_q2);

    thrust::device_vector<size_t> d_indices(n_q1);
    thrust::sequence(d_indices.begin(), d_indices.end());

    profiler = std::make_unique<ScopedProfiler>("GPU encode");

    // Encode vectors into keys
    encodeVectors<<<(n_q1 + 255) / 256, 256>>>(thrust::raw_pointer_cast(d_required.data()), n_q1, m_rows, thrust::raw_pointer_cast(d_keys1.data()));
    encodeVectors<<<(n_q2 + 255) / 256, 256>>>(thrust::raw_pointer_cast(d_scores_q2.data()), n_q2, m_rows, thrust::raw_pointer_cast(d_keys2.data()));

    profiler = std::make_unique<ScopedProfiler>("GPU sort");

    // Sort the keys from l1
    thrust::sort_by_key(d_keys1.begin(), d_keys1.end(), d_indices.begin());

    profiler = std::make_unique<ScopedProfiler>("GPU search");

    thrust::device_vector<bool> d_result(n_q2);
    thrust::binary_search(thrust::device, d_keys1.begin(), d_keys1.end(), d_keys2.begin(), d_keys2.end(), d_result.begin());

    profiler = std::make_unique<ScopedProfiler>("Check results");
    thrust::host_vector<bool> result = d_result;

    for (size_t i_q2 = 0; i_q2 < n_q2; ++i_q2)
    {
        if (!result[i_q2])
            continue;

        thrust::host_vector<size_t> indices = d_indices;
        /* Retrieve i_q1. */
        auto iter = thrust::find(d_keys1.begin(), d_keys1.end(), d_keys2[i_q2]);
        profiler.reset();

        return {true, {indices[thrust::distance(d_keys1.begin(), iter)], i_q2}};
    }
    profiler.reset();

    return {false, {n_q1, n_q2}};
}