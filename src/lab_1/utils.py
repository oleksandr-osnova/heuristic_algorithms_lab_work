import numpy as np
import pathlib
import random
import matplotlib.pyplot as plt

data_folder = pathlib.Path('../../data/lab_1')
input_data_folder = data_folder / 'input'
output_data_folder = data_folder / 'output'

def generate_random_route(cities):
    return np.array(random.sample(list(cities), len(cities)))

# Визначаємо функцію для обчислення довжини шляху
def calculate_distance(route, cycled=True):
    if cycled:
        # Створюємо копію маршруту із поверненням до початкового пункту
        route_extended = np.vstack([route, route[0]])
    else:
        route_extended = route

    # Обчислюємо довжину маршруту за допомогою евклідової відстані
    return np.sum(np.sqrt(np.sum(np.diff(route_extended, axis=0) ** 2, axis=1)))

# Застосовує 2-opt до маршруту, створюючи новий маршрут із реверсом підмаршруту.
def two_opt(route, i, j):
    return np.vstack([route[:i], route[i:j][::-1], route[j:]])

#Обчислює зміну довжини маршруту після застосування 2-opt
def incremental_distance_update(route, i, j):
    a, b = route[i - 1], route[i]
    c, d = route[j - 1], route[j]

    removed = calculate_distance([a, b], False) + calculate_distance([c, d], False)
    added = calculate_distance([a, c], False) + calculate_distance([b, d], False)

    return added - removed

def hill_climbing(cities, max_no_improve = None):
    num_cities = len(cities)
    max_no_improve = max(max_no_improve or num_cities * 5, 100)

    current_route = generate_random_route(cities)
    current_distance = calculate_distance(current_route)

    best_route, best_distance = current_route, current_distance

    no_improve_counter = 0

    while no_improve_counter < max_no_improve:
        improved = False

        # Генерація випадкових індексів для застосування 2-opt
        i, j = sorted(random.sample(range(1, num_cities), 2))
        if j - i == 1:  # Пропускаємо сусідні міста
            continue

        # Обчислюємо зміну довжини маршруту
        delta = incremental_distance_update(current_route, i, j)

        if delta < 0:  # Якщо покращення знайдено
            current_route = two_opt(current_route, i, j)
            current_distance += delta
            improved = True

        # Якщо знайдено поліпшення, оновлюємо кращий маршрут
        if improved:
            best_route, best_distance = current_route, current_distance
            no_improve_counter = 0
        else:
            no_improve_counter += 1

    return best_route, best_distance

def hill_climbing_multi_start(cities, num_starts = None, max_no_improve = None, interactive_plot = False):
    num_starts = num_starts or round(len(cities) * 0.2)

    best_route = None
    best_distance = float('inf')
    worst_route = None
    worst_distance = float(0)
    results = []
    iteration = 0

    if interactive_plot:
        plt.ion()  # Вмикаємо інтерактивний режим

    for _ in range(num_starts):
        current_route, current_distance = hill_climbing(cities, max_no_improve)
        results.append(current_distance)

        if current_distance < best_distance:
            best_route, best_distance = current_route, current_distance
        elif current_distance > worst_distance:
            worst_route, worst_distance = current_route, current_distance

        iteration += 1

        if iteration % 1 == 0:
            if interactive_plot:
                plot_route(best_route, best_distance, iteration, interactive_plot)
            print(f"Ітерація = {iteration}: Найкращий маршрут (довжина = {best_distance:.2f}), Найгірший маршрут (довжина = {worst_distance:.2f})")

    if interactive_plot:
        plt.ioff()  # Вимикаємо інтерактивний режим
        plt.show()

    return best_route, best_distance, worst_route, worst_distance, results

def plot_route(route, title, interactive_plot = False, block = True, pause = 0.5):
    plt.clf()

    # Замикаємо маршрут, зоб повернутися в початкову точку
    route_extended = np.vstack([route, route[0]])

    # Маршрут
    plt.plot(route_extended[:, 0], route_extended[:, 1], marker='o', color='green', label="Маршрут")

    # Початкова точка
    plt.plot(route[0, 0], route[0, 1], marker='.', markersize=15, color='red', label="Початкова точка")

    for i in range(len(route_extended) - 1):
        plt.quiver(
            route_extended[i, 0], route_extended[i, 1],  # Початок
            route_extended[i + 1, 0] - route_extended[i, 0],  # Зміщення по x
            route_extended[i + 1, 1] - route_extended[i, 1],  # Зміщення по y
            angles='xy', scale_units='xy', scale=1, color='green', width=0.003
        )

    plt.title(title)
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.legend()

    if interactive_plot:
        plt.draw()
        plt.pause(pause)
    else:
        plt.show(block=block)
        plt.pause(pause)

def plot_histogram(results):
    plt.figure()
    plt.hist(results, bins=20, color='blue', alpha=0.7, edgecolor='black')
    plt.title("Гістограма дистанцій для маршрутів")
    plt.xlabel("Довжина маршруту")
    plt.ylabel("Частота")
    plt.show()

def calculate_distance_difference(best_distance, worst_distance):
    return worst_distance - best_distance

def calculate_route_distance_difference(best_route, worst_route):
    best_distance = calculate_distance(best_route)
    worst_distance = calculate_distance(worst_route)
    return calculate_distance_difference(best_distance, worst_distance)

def plot_convergence(results):
    plt.figure()
    plt.plot(range(len(results)), sorted(results, reverse=True), marker='o', linestyle='-', color='green')
    plt.title("График сходимости целевой функции")
    plt.xlabel("Запуски")
    plt.ylabel("Лучшая длина маршрута")
    plt.grid()
    plt.show()
