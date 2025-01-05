#pragma once

#include <vector>
#include <random>

/* A market share feasibility instance. */
class MarkShareFeas
{
public:
    /* Generates a market share feasibility instance (or a multi-dimensional subset sum problem)
     *   Ax = b
     * with A \in \Z^{mxn}, a_ij \in \{0,..,99\}, n = 10 * (m - 1), m \in \N, b_i = |_ 1/2 \sum^n_j=1 a_ij _|
     *
     * See
     * Gérard Cornuéjols, Milind Dawande, (1999) A Class of Hard Small 0-1 Programs. INFORMS Journal on Computing 11(2):205-210, https://doi.org/10.1287/ijoc.11.2.205.
     */
    MarkShareFeas(size_t m, size_t seed = 0) : m_rows{m}, n_cols{10 * (m_rows - 1)}
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
    };

    const std::vector<size_t> &A()
    {
        return matrix;
    }

    const std::vector<size_t> &b()
    {
        return rhs;
    }

    size_t m()
    {
        return m_rows;
    }

    size_t n()
    {
        return n_cols;
    }

private:
    std::vector<size_t> matrix;
    std::vector<size_t> rhs;
    const size_t m_rows;
    const size_t n_cols;
};