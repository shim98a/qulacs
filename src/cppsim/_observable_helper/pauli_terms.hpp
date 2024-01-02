#pragma once

#include <cmath>
#include <stdexcept>
#include <vector>

#include <robin_hood.h>

namespace observable_helper {

using pauli_id_type = uint8_t;          // pauli_id (we only use 2bits)

/**
 * `single_pauli_type` is a type to represent Pauli operators like "X 1" and "Y 3".
 * We use `uint32_t` to represent Pauli operators.
 * The two most siginicant bits are used for Pauli ids (01 for 'X', 10 for 'Y' and 11 for 'Z').
 * The other 30 bits are used for qubit indices. So `single_pauli_type` variables
 * can represent up to 2^{30} qubits. Examples of `single_pauli_type` are
 * - 0b 01 000...00001 ("X 1")
 * - 0b 11 000...01111 ("Z 15")
 */
using single_pauli_type = uint32_t;
using index_type = single_pauli_type;   // qubit index type (we only use 30 bits)

using float_type = double;              // type used for Pauli terms' coefficients

// a value smaller than `eq_tolerance` is considered as zero.
constexpr float_type eq_tolerance = 1e-8;

class HermitianPauliTerms final {
private:
    struct _HashPaulis {
        size_t operator()(const std::vector<single_pauli_type>& paulis) const {
            size_t seed = robin_hood::hash_int(4711);
            for(single_pauli_type v: paulis){
                // almost the same hash function used in PyQUBO (https://github.com/recruit-communications/pyqubo)
                seed ^= robin_hood::hash_int(v) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
            }
            return robin_hood::hash_int(seed);
        }
    };

public:
    /**
     * hash map containing pairs of (product of Pauli operators, coefficient of the product).
     * `std::vector<single_pauli_type>` is the type for the products of Pauli operators.
     * `float_type` is the type for the coefficients.
     * In hash maps we need a hash function. `_HashPaulis::operator()` is the hash function
     * that takes a Pauli product (`std::vector<single_pauli_type>`) as its argument,
     * and it returns a hash value of the product.
     */
    using terms_map = robin_hood::unordered_map<
                                std::vector<single_pauli_type>, // product of Pauli operators like [X1, Y2, ...]
                                float_type,                     // coefficient of a Pauli term
                                _HashPaulis                     // struct that defines the hash function
                            >;
    /**
     * pair of std::vector<single_pauli_type> and float_type,
     * which represents "Pauli ids and coefficient".
    */
    using term_pair = terms_map::value_type;

    explicit HermitianPauliTerms() : _m_constant(0.0) {}

    HermitianPauliTerms(const HermitianPauliTerms&) = delete;

    HermitianPauliTerms(HermitianPauliTerms&&) = delete;

    HermitianPauliTerms& operator=(const HermitianPauliTerms&) = delete;

    HermitianPauliTerms& operator=(HermitianPauliTerms&&) = delete;

    inline const terms_map& get_terms() const noexcept {
        return _m_terms;
    } 
    
    float_type get_coeff(const std::vector<single_pauli_type>& paulis) {
        if(!_m_terms.contains(paulis)) {
            throw std::runtime_error("No term corresponding to the input argument found");
        }
        float_type coeff = _m_terms[paulis];
        return coeff;
    }

    inline float_type get_constant() const noexcept {
        return _m_constant;
    }

    inline size_t get_num_terms() const noexcept {
        return _m_terms.size() + ((std::abs(_m_constant) < eq_tolerance) ? 0 : 1);
    }

    size_t get_size_in_bytes() const {
        size_t size = sizeof(_m_constant) + sizeof(_m_terms);
        size += _m_terms.size() * (sizeof(std::vector<single_pauli_type>) + sizeof(float_type));
        for(const term_pair& term : _m_terms) {
            size += sizeof(single_pauli_type) * term.first.size();
        }
        return size;
    }

    void add_term(const std::vector<single_pauli_type>& paulis, float_type coeff) {
        if(std::abs(_m_terms[paulis] += coeff) < eq_tolerance) {
            _m_terms.erase(paulis);
        }
    }

    inline void add_constant(float_type constant) noexcept {
        _m_constant += constant;
    }

    inline void clear() {
        _m_terms.clear();
        _m_constant = 0.0;
    }

private:
    // hash map containing products of Pauli operators and corresponding coefficients.
    terms_map _m_terms;
    // constant value of the Hamiltonian.
    float_type _m_constant;
};

} // namespace observable_helper
