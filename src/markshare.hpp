#pragma once

#include <cassert>
#include <fstream>
#include <iostream>
#include <random>
#include <vector>
#include <fstream>
#include <string>

/* A market share feasibility instance. */
class MarkShareFeas
{
public:
    MarkShareFeas() = default;
    ~MarkShareFeas() = default;
    MarkShareFeas(MarkShareFeas &&) noexcept = default;
    MarkShareFeas &operator=(MarkShareFeas &&) = default;
    MarkShareFeas &operator=(const MarkShareFeas &) = default;
    MarkShareFeas(const MarkShareFeas &) = default;

    MarkShareFeas(size_t m, size_t n, std::vector<size_t> &&A, std::vector<size_t> &&b) : matrix(std::move(A)), rhs(std::move(b)), m_rows(m), n_cols(n)
    {
        assert(matrix.size() == m * n);
        assert(rhs.size() == m);
    }

    /* Generates a market share feasibility instance (or a multi-dimensional subset sum problem)
     *   Ax = b
     * with A \in \Z^{mxn}, a_ij \in \{0,..,99\}, n, m \in \N, b_i = |_ 1/2 \sum^n_j=1 a_ij _|
     *
     * See
     * Gérard Cornuéjols, Milind Dawande, (1999) A Class of Hard Small 0-1 Programs. INFORMS Journal on Computing 11(2):205-210, https://doi.org/10.1287/ijoc.11.2.205.
     */
    MarkShareFeas(size_t m, size_t n, size_t seed = 0) : m_rows{m}, n_cols{n}
    {
        matrix.reserve(m_rows * n_cols);
        rhs.reserve(m_rows);

        constexpr size_t lower = 0;
        constexpr size_t upper = 99;

        std::mt19937 generator(seed);

        /* Generate the instance truly random if no seed was given. */
        if (seed == 0)
        {
            std::random_device rd;
            generator.seed(rd());
        }

        std::uniform_int_distribution<size_t> distribution(lower, upper);

        /* Fill matrix A and right hand side b. */
        for (size_t m = 0; m < m_rows; ++m)
        {
            size_t row_sum = 0;

            for (size_t n = 0; n < n_cols; ++n)
            {
                size_t next = distribution(generator);
                matrix.push_back(next);
                row_sum += next;
            }

            rhs.push_back(std::floor(0.5 * row_sum));
        }
    }

    /* Generates a market share feasibility instance with n = 10 * (m - 1). */
    MarkShareFeas(size_t m, size_t seed = 0) : MarkShareFeas(m, 10 * (m - 1), seed) {};

    /* Read a MarkShareInstance from file. The format supported is
     * +++++++++++++++++++++++++++++++++++++++
     * m n
     * a11 a12 a13 ... a1n b1
     * a21 a22 ...     a2n b2
     *  .               .  .
     *  .               .  .
     *  .               .  .
     * am1 am2 am3     amn bm
     * +++++++++++++++++++++++++++++++++++++++
     */
    MarkShareFeas(const std::string instance_path)
    {
        std::ifstream file(instance_path);
        if (!file.is_open())
        {
            throw std::runtime_error("Could not open file");
        }

        /* Read the first line to get m and n. */
        file >> m_rows >> n_cols;

        matrix.reserve(m_rows * n_cols);
        rhs.reserve(m_rows);

        /* Read the matrix rows and right hand side. */
        for (size_t i_row = 0; i_row < m_rows; ++i_row)
        {
            for (size_t j_col = 0; j_col < n_cols; ++j_col)
            {
                size_t value;
                file >> value;
                matrix.push_back(value);
            }

            size_t rhs_val;
            file >> rhs_val;

            rhs.push_back(rhs_val);
        }
    };

    const std::vector<size_t> &A() const
    {
        return matrix;
    }

    const std::vector<size_t> &b() const
    {
        return rhs;
    }

    size_t m() const
    {
        return m_rows;
    }

    size_t n() const
    {
        return n_cols;
    }

    bool check_sum_feas(const size_t *values1, const size_t *values2) const
    {
        assert(values1[0] + values2[0] == rhs[0]);

        for (size_t row = 1; row < m_rows; ++row)
        {
            if (values1[row] + values2[row] != rhs[row])
                return false;
        }
        return true;
    }

    void compute_value(const std::vector<size_t> &indices, size_t offset, size_t *values) const
    {
        for (size_t row = 0; row < m_rows; ++row)
        {
            size_t val = 0;
            for (size_t i_idx = 0; i_idx < indices.size(); ++i_idx)
            {
                size_t j_col = indices[i_idx] + offset;
                val += matrix[row * n_cols + j_col];
            }

            values[row] = val;
        }
    }

    bool is_solution_feasible(const std::vector<size_t> &indices, size_t len_indices) const
    {
        for (size_t row = 0; row < m_rows; ++row)
        {
            size_t val = 0;
            for (size_t i_index = 0; i_index < len_indices; ++i_index)
            {
                size_t i = indices[i_index];
                val += matrix[row * n_cols + i];
                if (val > rhs[row])
                {
                    assert(row != 0);
                    return false;
                }
            }

            if (val != rhs[row])
            {
                assert(row != 0);
                return false;
            }
        }
        return true;
    }

    void print() const
    {
        std::cout << "[\n";
        /* Write the matrix. */
        for (size_t row = 0; row < m_rows; ++row)
        {
            std::cout << "[";
            for (size_t col = 0; col < n_cols; ++col)
            {
                std::cout << " " << matrix[row * n_cols + col];
            }
            std::cout << "\t" << rhs[row] << " ]\n";
        }
        std::cout << "]\n";
    }

    void write_as_prb(const std::string &name) const
    {
        std::cout << "Storing mark share instance as " << name << "\n";
        std::ofstream file(name);

        file << "[\n"
             << "]\n";
        file << "\n";

        file << "[\n";
        /* Write the matrix. */
        for (size_t row = 0; row < m_rows; ++row)
        {
            file << "[";
            for (size_t col = 0; col < n_cols; ++col)
            {
                file << " " << matrix[row * n_cols + col];
            }
            file << " ]\n";
        }
        file << "]\n";

        file << "[ ]\n";
        file << "\n";
        file << "[ ]\n";
        file << "\n";

        file << "[";
        /* Write the right hand side. */
        for (size_t row = 0; row < m_rows; ++row)
        {
            file << " " << rhs[row];
        }
        file << " ]\n";

        file << "\n";

        file << "[";
        /* Write 1s for each column. */
        for (size_t col = 0; col < n_cols; ++col)
        {
            file << " 1";
        }
        file << "]\n";

        file << "\n";
        file << "0";
        file << "\n";
    }

private:
    std::vector<size_t> matrix;
    std::vector<size_t> rhs;
    size_t m_rows;
    size_t n_cols;
};