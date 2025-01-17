import numpy as np
import matplotlib.pyplot as plt
import tsplib95

from src.lab_1 import utils as utils2
import utils
import random

def initialize_pheromones(num_cities):
    # Матриця феромонів ініціалізується малими рівними значеннями
    return np.ones((num_cities, num_cities))

def calculate_probabilities(pheromones, distances, current_city, visited, alpha = 2, beta = 3):
    num_cities = len(distances)
    probabilities = np.zeros(num_cities)

    for city in range(num_cities):
        if city not in visited:
            probabilities[city] = (
                    (pheromones[current_city][city] ** alpha) *
                    ((1 / distances[current_city][city]) ** beta)
            )

    # Нормалізуємо ймовірності
    probabilities_sum = np.sum(probabilities)
    return probabilities / probabilities_sum if probabilities_sum > 0 else probabilities

def ant_activity(distances, city_coords, pheromones, start, alpha = 2, beta = 3):
    num_cities = len(distances)
    current_city = start
    visited = [current_city]

    while len(visited) < num_cities:
        # Обчислюємо список доступних міст
        probabilities = calculate_probabilities(pheromones, distances, current_city, visited, alpha, beta)
        # Перевірка на можливість продовження маршруту
        if np.sum(probabilities) == 0:
            break
        # Вибір наступного міста
        next_city = np.random.choice(range(num_cities), p=probabilities)
        visited.append(next_city)
        current_city = next_city

    # Повертаємося до початкового міста для завершення маршруту
    visited.append(visited[0])
    total_distance = utils.calculate_distance(city_coords[visited])
    return visited, total_distance

def update_pheromones(pheromones, solutions, evaporation_rate, pheromone_intensity = 50):
    # Випаровування феромонів
    pheromones *= (1 - evaporation_rate)

    # Додавання нових феромонів
    for solution, distance in solutions:
        if distance == float('inf'):  # Ігноруємо невалідні маршрути
            continue
        for i in range(len(solution) - 1):
            a, b = solution[i], solution[i + 1]
            pheromones[a][b] += pheromone_intensity / distance
            pheromones[b][a] += pheromone_intensity / distance

def ant_colony_optimization(city_coords, num_ants = 10, num_iterations = 50, evaporation_rate = 0.4, pheromone_intensity = 5, alpha = 2, beta = 3, results_file_path=utils.csv_output_path):
    num_cities = len(city_coords)
    distances = np.array([
        [np.linalg.norm(city_coords[i] - city_coords[j]) for j in range(num_cities)]
        for i in range(num_cities)
    ])

    pheromones = initialize_pheromones(num_cities)

    best_solution = None
    distance = 0
    best_distance = float('inf')
    worst_distance = 0
    results = []

    # Ініціалізація CSV файлу
    utils2.create_csv_file(results_file_path)

    for iteration in range(num_iterations):
        solutions = []
        for _ in range(num_ants):
            # Використовуємо ant_activity для створення маршруту
            solution, distance = ant_activity(distances, city_coords, pheromones, random.randint(0, num_cities - 1), alpha, beta)
            solutions.append((solution, distance))

            if distance < best_distance:
                best_solution = solution
                best_distance = distance

            if distance > worst_distance:
                worst_distance = distance

        results.append(best_distance)
        # Глобальне оновлення феромонів
        update_pheromones(pheromones, solutions, evaporation_rate, pheromone_intensity)

        # Запис результатів у CSV
        parsed_iteration = str(iteration + 1).zfill(len(str(num_iterations)))
        utils2.write_row_csv_file([
            parsed_iteration, distance, best_distance, worst_distance, utils2.calculate_distance_difference(best_distance, worst_distance)
        ], results_file_path)

        print(f"Ітерація {iteration + 1}: Найкраща відстань = {best_distance:.2f}")

    # Повертаємо результати для аналізу
    difference = worst_distance - best_distance
    return best_solution, best_distance, worst_distance, difference, results

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

    best_route, best_distance, worst_distance, difference, results = ant_colony_optimization(city_coords, NUM_ANTS, NUM_ITERATIONS, EVAPORATION_RATE, PHEROMONE_INTENSITY, ALPHA, BETA)

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
