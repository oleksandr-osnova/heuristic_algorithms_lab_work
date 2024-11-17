import numpy as np
import tsplib95
import matplotlib.pyplot as plt
import utils

# Головна функція для запуску обчислювального експерименту
def main():
    number_of_starts = 20
    tsp_problem = tsplib95.load(str(utils.input_data_folder / 'a280.tsp'))
    tsp_problem_solution = tsplib95.load(str(utils.input_data_folder / 'a280.opt.tour'))

    cities = tsp_problem.node_coords
    city_coords = np.array([*cities.values()])

    tsp_problem_solution_tour = np.array([cities[city] for city in tsp_problem_solution.tours[0]])
    tsp_problem_solution_tour_distance = utils.calculate_distance(tsp_problem_solution_tour)

    plt.figure('Рішення')
    utils.plot_route(tsp_problem_solution_tour, f"Рішення, Дистанція: {tsp_problem_solution_tour_distance:.2f}", block = False, pause = 6)

    best_route, best_distance, worst_route, worst_distance, results = utils.hill_climbing_multi_start(city_coords, number_of_starts, interactive_plot=False)

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
