import numpy as np
import tsplib95
import matplotlib.pyplot as plt
import utils

# Головна функція для запуску обчислювального експерименту
def main():
    number_of_starts = 20
    interactive_plot = False
    initial_plot_pause = 6
    tsp_problem_file_name = 'a280.tsp'
    tsp_problem_solution_file_name = 'a280.opt.tour'

    tsp_problem = tsplib95.load(str(utils.input_data_folder / tsp_problem_file_name))

    cities = tsp_problem.node_coords
    city_coords = np.array([*cities.values()])

    if tsp_problem_solution_file_name:
        tsp_problem_solution = tsplib95.load(str(utils.input_data_folder / tsp_problem_solution_file_name))

        tsp_problem_solution_tour = np.array([cities[city] for city in tsp_problem_solution.tours[0]])
        tsp_problem_solution_tour_distance = utils.calculate_distance(tsp_problem_solution_tour)

        plt.figure('Рішення')
        utils.plot_route(
            tsp_problem_solution_tour, f"Рішення, Дистанція: {tsp_problem_solution_tour_distance:.2f}",
            block = False, pause = initial_plot_pause)

    iteration_figure = plt.figure('Ітерація')
    best_route, best_distance, worst_route, worst_distance, results = utils.hill_climbing_multi_start(
        city_coords, number_of_starts, interactive_plot=interactive_plot, first_pause = initial_plot_pause)
    plt.close(iteration_figure)

    plt.figure('Hill Climbing')
    utils.plot_route(best_route, f"Ітерація: {number_of_starts or 'остання'}, Дистанція: {best_distance:.2f}")

    # Різниця між найкращим та гіршим рішеннями
    difference = utils.calculate_route_distance_difference(best_route, worst_route)
    print(f"Різниця між найкращим та гіршим рішеннями: {difference:.2f}")

    # Побудова гістограми рішень
    utils.plot_histogram(results)

    # Графік збіжності цільової функції
    utils.plot_convergence(results)


# Запуск програми
if __name__ == "__main__":
    main()
