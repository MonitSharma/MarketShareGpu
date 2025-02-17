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

GpuData::GpuData(const MarkShareFeas &ms_inst, const std::vector<size_t> &set1_scores, const std::vector<size_t> &set2_scores, const std::vector<size_t> &set3_scores, const std::vector<size_t> &set4_scores) : m_rows(ms_inst.m()), n_cols(ms_inst.n())
{
    this->matrix = copy_to_device(ms_inst.A());
    this->rhs = copy_to_device(ms_inst.b());

    this->set1_scores = copy_to_device(set1_scores);
    this->set2_scores = copy_to_device(set2_scores);
    this->set3_scores = copy_to_device(set3_scores);
    this->set4_scores = copy_to_device(set4_scores);
}

GpuData::~GpuData()
{
    cudaFree(set1_scores);
    cudaFree(set2_scores);
    cudaFree(set3_scores);
    cudaFree(set4_scores);

    cudaFree(matrix);
    cudaFree(rhs);

    cudaFree(scores_buffer1);
    cudaFree(scores_buffer2);
    cudaFree(pairs_buffer1);
    cudaFree(pairs_buffer2);
}

void GpuData::init_scores_buffer(size_t n_pairs, bool first_buffer)
{
    size_t len_required = n_pairs * m_rows;

    if ((first_buffer && len_required > len_scores_buffer1) || (!first_buffer && len_required > len_scores_buffer2))
    {
        size_t new_len = static_cast<size_t>(len_required * 1.4 + 1);
        assert(new_len > len_required);

        if (first_buffer)
        {
            if (len_pairs_buffer1 > 0)
                cudaFree(scores_buffer1);
            cudaMalloc(&scores_buffer1, new_len * sizeof(size_t));
            len_scores_buffer1 = new_len;
        }
        else
        {
            if (len_pairs_buffer2 > 0)
                cudaFree(scores_buffer2);
            cudaMalloc(&scores_buffer2, new_len * sizeof(size_t));
            len_scores_buffer2 = new_len;
        }
    }

    assert(!((first_buffer && len_required > len_scores_buffer1) || (!first_buffer && len_required > len_scores_buffer2)));
}

void GpuData::init_keys_buffer(size_t n_keys, bool first_buffer)
{
    size_t len_required = n_keys;

    if ((first_buffer && len_required > len_keys_buffer1) || (!first_buffer && len_required > len_keys_buffer2))
    {
        size_t new_len = static_cast<size_t>(len_required * 1.4 + 1);
        assert(new_len > len_required);

        if (first_buffer)
        {
            /* Also allocate the indices array here! */
            if (len_keys_buffer1 > 0)
            {
                cudaFree(keys_buffer1);
                cudaFree(indices_keys_buffer1);
            }
            cudaMalloc(&keys_buffer1, new_len * sizeof(__int128_t));
            cudaMalloc(&indices_keys_buffer1, new_len * sizeof(size_t));
            len_keys_buffer1 = new_len;
        }
        else
        {
            if (len_keys_buffer2 > 0)
            {
                cudaFree(keys_buffer2);
                cudaFree(result);
            }
            cudaMalloc(&keys_buffer2, new_len * sizeof(__int128_t));
            cudaMalloc(&result, new_len * sizeof(size_t));
            len_keys_buffer2 = new_len;
        }
    }

    assert(!((first_buffer && len_required > len_keys_buffer1) || (!first_buffer && len_required > len_keys_buffer2)));
}

void GpuData::copy_pairs(const std::vector<std::pair<size_t, size_t>> &pairs, bool first_buffer)
{
    size_t len_required = pairs.size() * 2;

    if ((first_buffer && len_required > len_pairs_buffer1) || (!first_buffer && len_required > len_pairs_buffer2))
    {
        size_t new_len = static_cast<size_t>(len_required * 1.4 + 1);
        assert(new_len > len_required);

        if (first_buffer)
        {
            if (len_pairs_buffer1 > 0)
                cudaFree(pairs_buffer1);

            cudaMalloc(&pairs_buffer1, new_len * sizeof(size_t));
            len_pairs_buffer1 = new_len;
        }
        else
        {
            if (len_pairs_buffer2 > 0)
                cudaFree(pairs_buffer2);

            cudaMalloc(&pairs_buffer2, new_len * sizeof(size_t));
            len_pairs_buffer2 = new_len;
        }
    }

    assert(!((first_buffer && len_required > len_pairs_buffer1) || (!first_buffer && len_required > len_pairs_buffer2)));
    cudaMemcpy(first_buffer ? pairs_buffer1 : pairs_buffer2, pairs.data(), pairs.size() * 2 * sizeof(size_t), cudaMemcpyHostToDevice);
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

__global__ void compute_required(const size_t *__restrict__ rhs, size_t *__restrict__ scores, size_t m, size_t n_scores)
{
    // Get the index of the current thread in the grid
    size_t i_idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t i_score = i_idx / m;
    size_t i_m = i_idx % m;

    // Check bounds
    if (i_score < n_scores && i_m < m)
    {
        // Calculate the index
        size_t idx = i_score * m + i_m;
        // Perform the computation
        scores[idx] = rhs[i_m] - scores[idx];
    }
}

__global__ void combine_scores_kernel(const size_t *__restrict__ scores1, const size_t *__restrict__ scores2, const size_t *__restrict__ pairs, size_t *__restrict__ result, size_t m, size_t n_pairs)
{
    // Get the index of the current thread in the grid
    size_t i_idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t i_pair = i_idx / m;
    size_t i_m = i_idx % m;

    // Check bounds
    if (i_pair < n_pairs && i_m < m)
    {
        size_t idx1 = pairs[2 * i_pair];
        size_t idx2 = pairs[2 * i_pair + 1];

        // Perform the computation
        result[i_pair * m + i_m] = scores1[idx1 * m + i_m] + scores2[idx2 * m + i_m];
    }
}

void combine_scores_gpu(GpuData &gpu_data, const std::vector<std::pair<size_t, size_t>> &pairs, bool first_buffer)
{
    gpu_data.copy_pairs(pairs, first_buffer);
    gpu_data.init_scores_buffer(pairs.size(), first_buffer);

    int block_dim = 128;
    int n_blocks = (pairs.size() * gpu_data.m_rows + block_dim - 1) / block_dim;
    if (first_buffer)
    {
        combine_scores_kernel<<<n_blocks, block_dim>>>(gpu_data.set1_scores, gpu_data.set2_scores, gpu_data.pairs_buffer1, gpu_data.scores_buffer1, gpu_data.m_rows, pairs.size());
    }
    else
    {
        combine_scores_kernel<<<n_blocks, block_dim>>>(gpu_data.set3_scores, gpu_data.set4_scores, gpu_data.pairs_buffer2, gpu_data.scores_buffer2, gpu_data.m_rows, pairs.size());
    }
}

std::pair<bool, std::pair<size_t, size_t>> evaluate_solutions_gpu_hashing(GpuData &gpu_data, size_t n_q1, size_t n_q2)
{
    size_t m_rows = gpu_data.m_rows;

    assert(m_rows > 0);

    auto profiler = std::make_unique<ScopedProfiler>("GPU hash setup");

    profiler = std::make_unique<ScopedProfiler>("GPU compute required");

    // Configure grid and block sizes
    int block_dim = 128;
    int n_blocks = (n_q1 * gpu_data.m_rows + block_dim - 1) / block_dim;

    compute_required<<<n_blocks, block_dim>>>(gpu_data.rhs, gpu_data.scores_buffer1, m_rows, n_q1);

    profiler = std::make_unique<ScopedProfiler>("GPU data setup");

    // THE ALGORITHM!
    // Allocate encoded key arrays on the GPU
    gpu_data.init_keys_buffer(n_q1, true);
    gpu_data.init_keys_buffer(n_q2, false);

    thrust::sequence(thrust::device, gpu_data.indices_keys_buffer1, gpu_data.indices_keys_buffer1 + n_q1);

    profiler = std::make_unique<ScopedProfiler>("GPU encode");

    // Encode vectors into keys
    encodeVectors<<<(n_q1 + 255) / 256, 256>>>(gpu_data.scores_buffer1, n_q1, m_rows, gpu_data.keys_buffer1);
    encodeVectors<<<(n_q2 + 255) / 256, 256>>>(gpu_data.scores_buffer2, n_q2, m_rows, gpu_data.keys_buffer2);

    profiler = std::make_unique<ScopedProfiler>("GPU sort");

    // Sort the keys from l1
    thrust::sort_by_key(thrust::device, gpu_data.keys_buffer1, gpu_data.keys_buffer1 + n_q1, gpu_data.indices_keys_buffer1);

    profiler = std::make_unique<ScopedProfiler>("GPU search");

    thrust::binary_search(thrust::device, gpu_data.keys_buffer1, gpu_data.keys_buffer1 + n_q1, gpu_data.keys_buffer2, gpu_data.keys_buffer2 + n_q2, gpu_data.result);

    profiler = std::make_unique<ScopedProfiler>("Check results");

    std::vector<size_t> result(n_q2);
    cudaMemcpy(result.data(), gpu_data.result, n_q2 * sizeof(size_t), cudaMemcpyDeviceToHost);

    for (size_t i_q2 = 0; i_q2 < n_q2; ++i_q2)
    {
        if (!result[i_q2])
            continue;

        __int128_t val = 0;
        cudaMemcpy(&val, gpu_data.keys_buffer2 + i_q2, sizeof(__int128_t), cudaMemcpyDeviceToHost);

        /* Retrieve i_q1. */
        auto iter = thrust::find(thrust::device, gpu_data.keys_buffer1, gpu_data.keys_buffer1 + n_q1, val);
        profiler.reset();

        size_t pos_i_q1 = thrust::distance(gpu_data.keys_buffer1, iter);
        size_t i_q1 = 0;
        cudaMemcpy(&i_q1, gpu_data.indices_keys_buffer1 + pos_i_q1, sizeof(size_t), cudaMemcpyDeviceToHost);

        return {true, {i_q1, i_q2}};
    }
    profiler.reset();

    return {false, {n_q1, n_q2}};
}