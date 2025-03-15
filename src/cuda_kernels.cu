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
T *copy_to_device(const std::vector<T> &host_vec, int64_t &n_bytes_alloc_total)
{
    T *device_ptr;
    const auto n_bytes_alloc = host_vec.size() * sizeof(T);
    cudaMalloc(&device_ptr, n_bytes_alloc);
    cudaMemcpy(device_ptr, host_vec.data(), n_bytes_alloc, cudaMemcpyHostToDevice);

    n_bytes_alloc_total += n_bytes_alloc;
    return device_ptr;
}

GpuData::GpuData(const MarkShareFeas &ms_inst, const std::vector<size_t> &set1_scores, const std::vector<size_t> &set2_scores, const std::vector<size_t> &set3_scores, const std::vector<size_t> &set4_scores) : m_rows(ms_inst.m()), n_cols(ms_inst.n())
{
    this->matrix = copy_to_device(ms_inst.A(), n_bytes_alloc);
    this->rhs = copy_to_device(ms_inst.b(), n_bytes_alloc);

    this->set1_scores = copy_to_device(set1_scores, n_bytes_alloc);
    this->set2_scores = copy_to_device(set2_scores, n_bytes_alloc);
    this->set3_scores = copy_to_device(set3_scores, n_bytes_alloc);
    this->set4_scores = copy_to_device(set4_scores, n_bytes_alloc);
}

GpuData::~GpuData()
{
    cudaFree(set1_scores);
    cudaFree(set2_scores);
    cudaFree(set3_scores);
    cudaFree(set4_scores);

    cudaFree(matrix);
    cudaFree(rhs);

    cudaFree(required_buffer);

    cudaFree(search_buffer);
    cudaFree(results_search_buffer);
}

void GpuData::copy_pairs_required(const std::vector<std::pair<size_t, size_t>> &pairs)
{
    size_t len_needed = pairs.size();

    if (len_needed > len_required_buffer)
    {
        size_t new_len = static_cast<size_t>(len_needed * 1.4 + 1);
        assert(new_len > len_needed);

        cudaFree(required_buffer);

        n_bytes_alloc += sizeof(__int128_t) * (new_len - len_required_buffer);
        cudaMalloc(&required_buffer, new_len * sizeof(__int128_t));

        len_required_buffer = new_len;
    }

    size_t *pairs_required = (size_t *)(required_buffer);
    cudaMemcpy(pairs_required, pairs.data(), pairs.size() * 2 * sizeof(size_t), cudaMemcpyHostToDevice);
    n_required = len_needed;
}

void GpuData::copy_pairs_search(const std::vector<std::pair<size_t, size_t>> &pairs)
{
    size_t len_needed = pairs.size();

    if (len_needed > len_search_buffer)
    {
        size_t new_len = static_cast<size_t>(len_needed * 1.4 + 1);
        assert(new_len > len_needed);

        cudaFree(search_buffer);
        cudaFree(results_search_buffer);

        n_bytes_alloc += (sizeof(size_t) + sizeof(__int128_t)) * (new_len - len_search_buffer);
        cudaMalloc(&search_buffer, new_len * sizeof(__int128_t));
        cudaMalloc(&results_search_buffer, new_len * sizeof(size_t));

        len_search_buffer = new_len;
    }

    size_t *pairs_serach = (size_t *)(search_buffer);
    cudaMemcpy(pairs_serach, pairs.data(), pairs.size() * 2 * sizeof(size_t), cudaMemcpyHostToDevice);
    n_search = len_needed;
}

template <bool ENCODE_REQUIRED>
__global__ void combine_and_encode_kernel(const size_t *__restrict__ scores1, const size_t *__restrict__ scores2, const size_t *__restrict__ rhs, size_t *__restrict__ pairs, size_t n_pairs, size_t m_rows)
{
    const int i_pair = blockIdx.x * blockDim.x + threadIdx.x;
    constexpr size_t BASE = 10000; /* Maximum value per vector element */

    if (i_pair >= n_pairs)
        return;

    __int128_t key = 0;
    const size_t idx1 = pairs[2 * i_pair];
    const size_t idx2 = pairs[2 * i_pair + 1];

    for (size_t i_row = 0; i_row < m_rows; ++i_row)
    {

        /* Compute the pair's score of this row and add it (encoded) to key. */
        size_t row_score = scores1[idx1 * m_rows + i_row] + scores2[idx2 * m_rows + i_row];

        if (ENCODE_REQUIRED)
            row_score = rhs[i_row] - row_score;

        /* FMA. */
        key = key * BASE + row_score;
    }

    /* Offload key to the original pair position. */
    *(__int128_t *)(pairs + 2 * i_pair) = key;
}

void combine_and_encode_gpu(GpuData &gpu_data, const std::vector<std::pair<size_t, size_t>> &pairs1, const std::vector<std::pair<size_t, size_t>> &pairs2)
{
    const size_t m_rows = gpu_data.m_rows;
    const size_t n_p1 = pairs1.size();
    const size_t n_p2 = pairs2.size();

    /* The shorter array will be encoded as required and will be sorted. */
    const bool encode_first_as_required = (n_p1 < n_p2);

    auto profiler = std::make_unique<ScopedProfiler>("Eval GPU: combine + encode  ");

    gpu_data.copy_pairs_required(encode_first_as_required ? pairs1 : pairs2);
    gpu_data.copy_pairs_search(encode_first_as_required ? pairs2 : pairs1);

    /* Each pair is treated by one single thread. */
    constexpr int block_dim = 128;
    int n_blocks_1 = (n_p1 + block_dim - 1) / block_dim;
    int n_blocks_2 = (n_p2 + block_dim - 1) / block_dim;

    if (encode_first_as_required)
    {
        combine_and_encode_kernel<true><<<n_blocks_1, block_dim>>>(gpu_data.set1_scores, gpu_data.set2_scores, gpu_data.rhs, (size_t *)gpu_data.required_buffer, n_p1, m_rows);
        combine_and_encode_kernel<false><<<n_blocks_2, block_dim>>>(gpu_data.set3_scores, gpu_data.set4_scores, gpu_data.rhs, (size_t *)gpu_data.search_buffer, n_p2, m_rows);
    }
    else
    {
        combine_and_encode_kernel<false><<<n_blocks_1, block_dim>>>(gpu_data.set1_scores, gpu_data.set2_scores, gpu_data.rhs, (size_t *)gpu_data.search_buffer, n_p1, m_rows);
        combine_and_encode_kernel<true><<<n_blocks_2, block_dim>>>(gpu_data.set3_scores, gpu_data.set4_scores, gpu_data.rhs, (size_t *)gpu_data.required_buffer, n_p2, m_rows);
    }

    profiler.reset();
}

std::pair<bool, __int128_t> find_equal_hash(GpuData &gpu_data)
{
    /* The shorter array will be encoded as required and will be sorted. */
    const size_t n_required = gpu_data.n_required;
    const size_t n_search = gpu_data.n_search;

    __int128_t *required = gpu_data.required_buffer;
    __int128_t *search = gpu_data.search_buffer;
    size_t *result = gpu_data.results_search_buffer;

    /* Compute hashes of required vectors. */
    auto profiler = std::make_unique<ScopedProfiler>("Eval GPU: sort required     ");

    /* Sort the array of required keys. */
    thrust::sort(thrust::device, required, required + n_required);

    profiler = std::make_unique<ScopedProfiler>("Eval GPU: binary search     ");

    thrust::binary_search(thrust::device, required, required + n_required, search, search + n_search, result);

    profiler = std::make_unique<ScopedProfiler>("Eval GPU: check results     ");

    thrust::device_ptr<size_t> result_ptr(result);
    auto iter = thrust::find(thrust::device, result_ptr, result_ptr + n_search, true);

    if (iter != result_ptr + n_search)
    {
        printf("Found!\n");

        /* Get the position of the found element and copy back its (unsorted) search value. */
        size_t i_search = thrust::distance(result_ptr, iter);

        __int128_t val = 0;
        cudaMemcpy(&val, search + i_search, sizeof(__int128_t), cudaMemcpyDeviceToHost);

        return {true, val};
    }
    profiler.reset();

    return {false, 0};
}

std::pair<size_t, size_t> find_hash_positions_gpu(GpuData &gpu_data, __int128_t hash, size_t n_p1, size_t n_p2)
{
    const bool encode_first_as_required = (n_p1 < n_p2);

    __int128_t *required = gpu_data.required_buffer;
    __int128_t *search = gpu_data.search_buffer;

    const size_t n_required = gpu_data.n_required;
    const size_t n_search = gpu_data.n_search;

    auto iter_req = thrust::find(thrust::device, required, required + n_required, hash);
    auto iter_search = thrust::find(thrust::device, search, search + n_search, hash);

    assert(iter_req != required + n_required);
    assert(iter_search != search + n_search);

    auto pos_req = thrust::distance(required, iter_req);
    auto pos_search = thrust::distance(search, iter_search);

    if (encode_first_as_required)
        return {pos_req, pos_search};
    else
        return {pos_search, pos_req};
}
