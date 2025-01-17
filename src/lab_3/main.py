import numpy as np
import matplotlib.pyplot as plt
import tsplib95
import utils

def main ():
    NUM_ANTS = 76  # Кількість мурах
    NUM_ITERATIONS = 150  # Кількість ітерацій
    ALPHA = 1  # Вплив феромонів
    BETA = 3  # Вплив відстаней
    EVAPORATION_RATE = 0.4  # Швидкість випаровування феромонів
    PHEROMONE_INTENSITY = 50  # Інтенсивність феромонів

    tsp_problem_file_name = 'pr76.tsp'
    tsp_problem_solution_file_name = 'pr76.opt.tour'

    tsp_problem = tsplib95.load(str(utils.input_data_folder / tsp_problem_file_name))

    cities = tsp_problem.node_coords
    city_coords = np.array([*cities.values()])

    if tsp_problem_solution_file_name:
        tsp_problem_solution = tsplib95.load(str(utils.input_data_folder / tsp_problem_solution_file_name))

        tsp_problem_solution_tour = np.array([cities[city] for city in tsp_problem_solution.tours[0]])
        tsp_problem_solution_tour_distance = utils.calculate_distance(tsp_problem_solution_tour)

        fig_solution = plt.figure('Solution')
        utils.plot_route(tsp_problem_solution_tour, f"Рішення, Дистанція: {tsp_problem_solution_tour_distance:.2f}")
        fig_solution.savefig(utils.output_data_folder / 'Solution.png')

    best_route, best_distance, worst_distance, difference, results = utils.ant_colony_optimization(city_coords, NUM_ANTS, NUM_ITERATIONS, EVAPORATION_RATE, PHEROMONE_INTENSITY, ALPHA, BETA)

    # Показ статистики
    print(f"Різниця між найкращим та гіршим рішеннями: {difference:.2f}")

    fig_histogram = plt.figure('Histogram')
    utils.plot_histogram(results)
    fig_histogram.savefig(utils.output_data_folder / 'Histogram.png')

    fig_convergence = plt.figure('Convergence')
    utils.plot_convergence(results)
    fig_convergence.savefig(utils.output_data_folder / 'Convergence.png')

    fig_hill_climbing = plt.figure('ACO')
    utils.plot_route(city_coords[best_route], f"ACO: Найкраща дистанція {best_distance:.2f}")
    fig_hill_climbing.savefig(utils.output_data_folder / 'ACO.png')

# Запуск програми
if __name__ == "__main__":
    main()
