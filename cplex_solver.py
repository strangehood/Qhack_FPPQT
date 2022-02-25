import numpy as np
from docplex.mp.model import Model
from docplex.mp.context import Context
from docplex.mp.solution import SolveSolution

from typing import Optional, List


class CPLEXSolver:
    def __init__(self, show_timer=False):
        self.show_timer = show_timer

    def sample(self, qubo: np.ndarray, show_timer=None, TimeLimit=None, starting_point: Optional[List] = None,
               log_output=False):
        show_timer = show_timer or self.show_timer
        model = Model()
        x_vars = [model.binary_var(f"x_{i}") for i in range(qubo.shape[0])]
        expr = 0
        rows, cols = qubo.nonzero()
        for row, col in zip(rows, cols):
            value = qubo[row, col]
            if row == col:
                expr += value * x_vars[row]
            else:
                expr += value * x_vars[row] * x_vars[col]
        model.minimize(expr)
        context = Context.make_default_context()
        if TimeLimit is not None:
            context.cplex_parameters.timelimit = TimeLimit
        context.solver.log_output = log_output
        if starting_point is not None:
            mip_start_sol = SolveSolution(model)
            for var, value in zip(x_vars, starting_point):
                mip_start_sol.add_var_value(var, value)
            model.add_mip_start(mip_start_sol)
        results = model.solve(agent='local', context=context)
        energy = results.get_objective_value()
        x_sol = np.array(results.get_values(x_vars), dtype=int)
        return x_sol, energy
