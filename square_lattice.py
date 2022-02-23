from __future__ import annotations
import numpy as np


class SquareLattice:
    def __init__(self, size=20):
        self.size = size
        self.hamiltonian = None
        self.extra_energy = 0
        self.j1 = 0
        self.j2 = 0
        self.field = 0

    def nearest_neighbours(self, i, j):
        if i != 0 and j != 0 and i != self.size - 1 and j != self.size - 1:
            return [[i, j - 1], [i - 1, j], [i, j + 1], [i + 1, j]]
        elif i == 0 and j != 0 and j != self.size - 1:
            return [[i, j - 1], [i, j + 1], [i + 1, j]]
        elif i != 0 and j == 0 and i != self.size - 1:
            return [[i - 1, j], [i, j + 1], [i + 1, j]]
        elif i == self.size - 1 and j != 0 and j != self.size - 1:
            return [[i, j - 1], [i - 1, j], [i, j + 1]]
        elif j == self.size - 1 and i != 0 and i != self.size - 1:
            return [[i - 1, j], [i, j - 1], [i + 1, j]]
        elif i == 0 and j == 0:
            return [[1, 0], [0, 1]]
        elif i == 0 and j == self.size - 1:
            return [[0, j - 1], [1, j]]
        elif j == 0 and i == self.size - 1:
            return [[i - 1, 0], [i, 1]]
        else:
            return [[i - 1, j], [i, j - 1]]

    def diagonal_neighbours(self, i, j):
        if i != 0 and j != 0 and i != self.size - 1 and j != self.size - 1:
            return [[i - 1, j - 1], [i - 1, j + 1], [i + 1, j + 1], [i + 1, j - 1]]
        elif i == 0 and j != 0 and j != self.size - 1:
            return [[i + 1, j - 1], [i + 1, j + 1]]
        elif i != 0 and j == 0 and i != self.size - 1:
            return [[i - 1, j + 1], [i + 1, j + 1]]
        elif i == self.size - 1 and j != 0 and j != self.size - 1:
            return [[i - 1, j - 1], [i - 1, j + 1]]
        elif j == self.size - 1 and i != 0 and i != self.size - 1:
            return [[i - 1, j - 1], [i + 1, j - 1]]
        elif i == 0 and j == 0:
            return [[1, 1]]
        elif i == 0 and j == self.size - 1:
            return [[1, j - 1]]
        elif j == 0 and i == self.size - 1:
            return [[i - 1, 1]]
        else:
            return [[i - 1, j - 1]]

    def get_index(self, two_dim_index):
        two_dim_index = np.transpose(two_dim_index)
        result = np.array(two_dim_index[0] + self.size * two_dim_index[1])
        return result.flatten()

    def create_zero_field_hamiltonian(self, j2_j1_rate, j1=1):
        """
        :param j1:
        :param j2_j1_rate: J2/J1 rate
        :return:
        """
        j2 = j1 * j2_j1_rate
        hamiltonian = np.zeros((self.size ** 2, self.size ** 2))
        for k in range(self.size):
            for m in range(self.size):
                i = self.get_index([k, m])
                for j in self.get_index(self.nearest_neighbours(k, m)):
                    hamiltonian[i, j] -= 1 / 2 * j1
                for j in self.get_index(self.diagonal_neighbours(k, m)):
                    hamiltonian[i, j] += 1 / 2 * j2
        self.extra_energy = sum(sum(hamiltonian))
        diag_ham = np.zeros(self.size ** 2)
        for p in range(len(diag_ham)):
            diag_ham[p] = - 2 * (sum(hamiltonian[:, p]) + sum(hamiltonian[p, :]))
        self.hamiltonian = 4 * hamiltonian + np.diag(diag_ham)
        self.j1 = j1
        self.j2 = j2
        return self.hamiltonian

    def add_field(self, field):  # this function should be the last one
        self.field = field
        self.extra_energy += field * self.size ** 2
        self.hamiltonian -= 2 * field * np.identity(self.size ** 2)

    def create_full_hamiltonian(self, j2_j1_rate, field, j1=1):
        self.create_zero_field_hamiltonian(j2_j1_rate, j1)
        self.add_field(field)
        return self.hamiltonian
