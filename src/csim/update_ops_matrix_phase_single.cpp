
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "constant.hpp"
#include "update_ops.hpp"
#include "utility.hpp"
#ifdef _OPENMP
#include <omp.h>
#endif

#ifdef _USE_SIMD
#ifdef _MSC_VER
#include <intrin.h>
#else
#include <x86intrin.h>
#endif
#endif

void single_qubit_phase_gate(
    UINT target_qubit_index, CTYPE phase, CTYPE* state, ITYPE dim) {
#ifdef _USE_SIMD
#ifdef _OPENMP
    OMPutil omputil = get_omputil();
    omputil->set_qulacs_num_threads(dim, 12);
#endif
    single_qubit_phase_gate_parallel_simd(
        target_qubit_index, phase, state, dim);
#ifdef _OPENMP
    omputil->reset_qulacs_num_threads();
#endif
#else
#ifdef _OPENMP
    OMPutil omputil = get_omputil();
    omputil->set_qulacs_num_threads(dim, 12);
#endif
    single_qubit_phase_gate_parallel_unroll(
        target_qubit_index, phase, state, dim);
#ifdef _OPENMP
    omputil->reset_qulacs_num_threads();
#endif
#endif
}

void single_qubit_phase_gate_parallel_unroll(
    UINT target_qubit_index, CTYPE phase, CTYPE* state, ITYPE dim) {
    // target tmask
    const ITYPE mask = 1ULL << target_qubit_index;
    const ITYPE low_mask = mask - 1;
    const ITYPE high_mask = ~low_mask;

    // loop varaibles
    const ITYPE loop_dim = dim / 2;
    if (target_qubit_index == 0) {
        ITYPE state_index;
#ifdef _OPENMP
#pragma omp parallel for
#endif
        for (state_index = 1; state_index < dim; state_index += 2) {
            state[state_index] *= phase;
        }
    } else {
        ITYPE state_index;
#ifdef _OPENMP
#pragma omp parallel for
#endif
        for (state_index = 0; state_index < loop_dim; state_index += 2) {
            ITYPE basis = (state_index & low_mask) +
                          ((state_index & high_mask) << 1) + mask;
            state[basis] *= phase;
            state[basis + 1] *= phase;
        }
    }
}

#ifdef _USE_SIMD
void single_qubit_phase_gate_parallel_simd(
    UINT target_qubit_index, CTYPE phase, CTYPE* state, ITYPE dim) {
    // target tmask
    const ITYPE mask = 1ULL << target_qubit_index;
    const ITYPE low_mask = mask - 1;
    const ITYPE high_mask = ~low_mask;

    // loop varaibles
    const ITYPE loop_dim = dim / 2;
    if (target_qubit_index == 0) {
        ITYPE state_index;
#ifdef _OPENMP
#pragma omp parallel for
#endif
        for (state_index = 1; state_index < dim; state_index += 2) {
            state[state_index] *= phase;
        }
    } else {
        ITYPE state_index;
        __m256d mv0 = _mm256_set_pd(
            -_cimag(phase), _creal(phase), -_cimag(phase), _creal(phase));
        __m256d mv1 = _mm256_set_pd(
            _creal(phase), _cimag(phase), _creal(phase), _cimag(phase));
#ifdef _OPENMP
#pragma omp parallel for
#endif
        for (state_index = 0; state_index < loop_dim; state_index += 2) {
            ITYPE basis = (state_index & low_mask) +
                          ((state_index & high_mask) << 1) + mask;
            double* ptr = (double*)(state + basis);
            __m256d data = _mm256_loadu_pd(ptr);
            __m256d data0 = _mm256_mul_pd(data, mv0);
            __m256d data1 = _mm256_mul_pd(data, mv1);
            data = _mm256_hadd_pd(data0, data1);
            _mm256_storeu_pd(ptr, data);
        }
    }
}
#endif
