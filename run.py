import glob
import pandas as pd
from src.io_tsp import ProblemInstance
from src.TSP_solver import TSPSolver, available_improvers, available_solvers
import numpy as np


def use_solver_to_compute_solution(solver, improve, index, results, name, verbose, show_plots):
    solver.bind(improve)
    solver.compute_solution(return_value=False, verbose=verbose)

    if verbose:
        print(f"the total length for the solution found is {solver.found_length}",
              f"while the optimal length is {solver.problem_instance.best_sol}",
              f"the gap is {solver.gap}%",
              f"the solution is found in {solver.duration} seconds", sep="\n")

    index.append((name, solver.name_method))
    results.append([solver.found_length, solver.problem_instance.best_sol, solver.gap, solver.duration])

    if show_plots:
        solver.plot_solution()


def run(show_plots=False, verbose=False):
    # problems = glob.glob('./problems/*.tsp')
    problems = ["./problems/eil76.tsp"]
    solvers_names = available_solvers.keys()
    improvers_names = available_improvers.keys()
    results = []
    index = []
    for problem_path in problems:
        prob_instance = ProblemInstance(problem_path)
        if verbose:
            prob_instance.print_info()
        if show_plots:
            prob_instance.plot_data()

        for solver_name in solvers_names:
            for improve in improvers_names:
                solver = TSPSolver(solver_name, prob_instance)
                use_solver_to_compute_solution(solver, improve, index, results, problem_path, verbose, show_plots)
                for improve2 in [j for j in improvers_names if j not in [improve]]:
                    use_solver_to_compute_solution(solver, improve2, index, results, problem_path, verbose, show_plots)

                    for improve3 in [j for j in improvers_names if j not in [improve, improve2]]:
                        use_solver_to_compute_solution(solver, improve3, index, results, problem_path, verbose,
                                                       show_plots)
                        solver.pop()

                    solver.pop()

        if prob_instance.exist_opt and show_plots:
            solver = TSPSolver("optimal", prob_instance)
            solver.solved = True
            solver.solution = np.concatenate([prob_instance.optimal_tour, [prob_instance.optimal_tour[0]]])
            solver.plot_solution()

    index = pd.MultiIndex.from_tuples(index, names=['problem', 'method'])

    return pd.DataFrame(results, index=index, columns=["tour length", "optimal solution", "gap", "time to solve"])


if __name__ == '__main__':
    df = run(show_plots=False, verbose=True)
    df.to_csv("./results.csv")
