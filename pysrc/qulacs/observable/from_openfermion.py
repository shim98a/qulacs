import numpy as np
from openfermion import InteractionOperator

import qulacs_core
from qulacs_core.observable import (
    create_observable_from_real_electron_integrals,
    create_observable_from_complex_electron_integrals,
)


def create_observable_from_openfermion_op(
    operator: InteractionOperator,
) -> qulacs_core.Observable:
    if not isinstance(operator, InteractionOperator):
        raise TypeError(
            "Argument must be an instance of `openfermion.InteractionOperator`."
        )
    if operator.one_body_tensor.dtype == np.float64:
        return create_observable_from_real_electron_integrals(
            operator.one_body_tensor, operator.two_body_tensor, operator.n_qubits
        )
    elif operator.one_body_tensor.dtype == np.complex128:
        return create_observable_from_complex_electron_integrals(
            operator.one_body_tensor, operator.two_body_tensor, operator.n_qubits
        )
    else:
        raise TypeError(
            "Type of tensor values must be 'np.float64' or 'np.complex128'."
        )
