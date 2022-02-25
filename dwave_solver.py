from typing import Optional

import numpy as np
from dwave.system import DWaveSampler, EmbeddingComposite


class DwaveSolver:
    DEFAULT_DWAVE_PARAMS = {
        'num_reads': 100
    }

    def __init__(self, token: str):
        self.token = token
        self.sampler = EmbeddingComposite(DWaveSampler(token=self.token))

    def sample(self, qubo: np.ndarray, dwave_parameters: Optional[dict] = None):

        dwave_parameters = dwave_parameters or {}
        for key, value in self.DEFAULT_DWAVE_PARAMS.items():
            if key not in dwave_parameters:
                dwave_parameters[key] = value

        sample_set = self.sampler.sample_qubo(qubo, **dwave_parameters)
        pandas_data = sample_set.to_pandas_dataframe()
        energy = min(pandas_data['energy'])

        configuration = pandas_data[pandas_data['energy'] == min(pandas_data['energy'])].drop(
            columns=['chain_break_fraction', 'energy', 'num_occurrences']
        ).iloc[0].values

        return configuration, energy
