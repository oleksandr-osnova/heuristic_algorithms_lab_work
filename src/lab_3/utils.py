import numpy as np
from src.lab_1 import utils
import pathlib
import random

calculate_distance = utils.calculate_distance
plot_route = utils.plot_route
calculate_route_distance_difference = utils.calculate_route_distance_difference
plot_histogram = utils.plot_histogram
plot_convergence = utils.plot_convergence
generate_random_route = utils.generate_random_route

data_folder = pathlib.Path('../../data/lab_3')
input_data_folder = data_folder / 'input'
output_data_folder = data_folder / 'output'

csv_output = "output.csv"
csv_output_path = output_data_folder / csv_output

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

def ant_colony_optimization(city_coords, num_ants = 10, num_iterations = 50, evaporation_rate = 0.4, pheromone_intensity = 5, alpha = 2, beta = 3, results_file_path=csv_output_path):
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
    utils.create_csv_file(results_file_path)

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
        utils.write_row_csv_file([
            parsed_iteration, distance, best_distance, worst_distance, utils.calculate_distance_difference(best_distance, worst_distance)
        ], results_file_path)

        print(f"Ітерація {iteration + 1}: Найкраща відстань = {best_distance:.2f}")

    # Повертаємо результати для аналізу
    difference = worst_distance - best_distance
    return best_solution, best_distance, worst_distance, difference, results