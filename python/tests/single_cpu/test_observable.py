import numpy as np
import pytest

from openfermion import jordan_wigner, InteractionOperator, QubitOperator
from openfermionpyscf import generate_molecular_hamiltonian

from qulacs import GeneralQuantumOperator, Observable
from qulacs.observable import create_observable_from_openfermion_op


class TestObservable:
    def test_get_matrix_from_observable(self) -> None:
        n_qubits = 3
        obs = Observable(n_qubits)
        obs.add_operator(0.5, "Z 2")
        obs.add_operator(1.0, "X 0 X 1 X 2")
        obs.add_operator(1.0, "Y 1")
        ans = np.array(
            [
                [0.5, 0, -1j, 0, 0, 0, 0, 1],
                [0, 0.5, 0, -1j, 0, 0, 1, 0],
                [1j, 0, 0.5, 0, 0, 1, 0, 0],
                [0, 1j, 0, 0.5, 1, 0, 0, 0],
                [0, 0, 0, 1, -0.5, 0, -1j, 0],
                [0, 0, 1, 0, 0, -0.5, 0, -1j],
                [0, 1, 0, 0, 1j, 0, -0.5, 0],
                [1, 0, 0, 0, 0, 1j, 0, -0.5],
            ],
            dtype=np.complex128,
        )
        assert np.linalg.norm(ans - obs.get_matrix().todense()) <= 1e-6  # type: ignore

    def test_get_matrix_from_general_quantum_operator(self) -> None:
        n_qubits = 3
        obs = GeneralQuantumOperator(n_qubits)
        obs.add_operator(0.5j, "Z 2")
        obs.add_operator(1.0, "X 0 X 1 X 2")
        obs.add_operator(1.0, "Y 1")
        ans = np.array(
            [
                [0.5j, 0, -1j, 0, 0, 0, 0, 1],
                [0, 0.5j, 0, -1j, 0, 0, 1, 0],
                [1j, 0, 0.5j, 0, 0, 1, 0, 0],
                [0, 1j, 0, 0.5j, 1, 0, 0, 0],
                [0, 0, 0, 1, -0.5j, 0, -1j, 0],
                [0, 0, 1, 0, 0, -0.5j, 0, -1j],
                [0, 1, 0, 0, 1j, 0, -0.5j, 0],
                [1, 0, 0, 0, 0, 1j, 0, -0.5j],
            ],
            dtype=np.complex128,
        )
        assert np.linalg.norm(ans - obs.get_matrix().todense()) <= 1e-6  # type: ignore


class TestCreateObservable:
    _PAULI_ID_TO_STR = {1: "X", 2: "Y", 3: "Z"}

    def _make_paulis_tuple(
        self, index_list: list[int], pauli_id_list: list[int]
    ) -> tuple[tuple[int, str]]:
        """For example `_make_paulis_tuple([0, 3, 5], [3, 1, 2])` returns ((0, 'Z'), (3, 'X'), (5, 'Y')).
        The input lists are assumed to be sorted in the index order.
        """
        assert len(index_list) == len(pauli_id_list)
        return tuple(
            (index, self._PAULI_ID_TO_STR[pauli_id])
            for index, pauli_id in zip(index_list, pauli_id_list)
        )

    def _compare_observable_and_openfermion_operator(
        self, observable: Observable, qubit_operator_openfermion: QubitOperator
    ) -> None:
        terms_openfermion = qubit_operator_openfermion.terms

        assert (
            observable.get_term_count() == len(terms_openfermion) - 1
        ), "check terms count"

        for i in range(observable.get_term_count()):
            paulis_tuple = self._make_paulis_tuple(
                observable.get_term(i).get_index_list(),
                observable.get_term(i).get_pauli_id_list(),
            )
            coeff = observable.get_term(i).get_coef()

            assert paulis_tuple in terms_openfermion, "check pauli term key"
            assert (
                terms_openfermion[paulis_tuple].real == coeff.real
            ), "check coefficient"

    @pytest.mark.parametrize("is_comlex_number", [True, False])
    def test_create_observable_from_openfermion_op(
        self, is_comlex_number: bool
    ) -> None:
        """For `openfermion.InteractionOperator`,
        `create_observable_from_openfermion_op` works in the same way as
        `openfermion.jordan_wigner` except that in
        `create_observable_from_openfermion_op` the constant value term and
        the imaginary parts of the pauli term coefficients are explicitly ignored.
        So we check if the operators created by `openfermion.jordan_wigner` and
        `create_observable_from_openfermion_op` are the same expect for the
        constant term and the imaginary parts of the pauli terms coefficients.
        """
        n_qubits = 8
        one_body = np.random.randn(n_qubits, n_qubits).astype(float)
        two_body = np.random.randn(n_qubits, n_qubits, n_qubits, n_qubits).astype(float)
        if is_comlex_number:
            one_body = one_body.astype(complex) + 1j * np.random.randn(
                n_qubits, n_qubits
            ).astype(float)
            two_body = two_body.astype(complex) + 1j * np.random.randn(
                n_qubits, n_qubits, n_qubits, n_qubits
            ).astype(float)

        # random InteractionOperator
        interaction_op = InteractionOperator(0.0, one_body, two_body)

        openfermion_qubit_op = jordan_wigner(interaction_op)
        observable = create_observable_from_openfermion_op(interaction_op)

        self._compare_observable_and_openfermion_operator(
            observable, openfermion_qubit_op
        )

    def test_create_observable_from_openfermion_H2O(self) -> None:
        basis = "sto-3g"  # basis set
        multiplicity = 1  # spin multiplicity
        geometry = [
            ("O", (0, 0, 0.117)),
            ("H", (0, 0.755, -0.471)),
            ("H", (0, -0.755, -0.471)),
        ]  # xyz coordinates for atoms

        interaction_op = generate_molecular_hamiltonian(geometry, basis, multiplicity)

        observable = create_observable_from_openfermion_op(interaction_op)
        qubit_operator_openfermion = jordan_wigner(interaction_op)

        self._compare_observable_and_openfermion_operator(
            observable, qubit_operator_openfermion
        )

    @pytest.mark.parametrize(
        "operator,error_msg",
        [
            (
                QubitOperator(),
                "Argument must be an instance of `openfermion.InteractionOperator`.",
            ),
            (
                InteractionOperator(0, np.zeros(1, dtype=int), np.zeros(1, dtype=int)),
                "Type of tensor values must be 'np.float64' or 'np.complex128'.",
            ),
        ],
    )
    def test_create_observable_from_openfermion_error(
        self, operator, error_msg
    ) -> None:
        with pytest.raises(TypeError) as err:
            create_observable_from_openfermion_op(operator)
        assert str(err.value) == error_msg
