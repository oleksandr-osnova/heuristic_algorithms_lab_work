import numpy as np
import csv
import pathlib
import random
import matplotlib.pyplot as plt

data_folder = pathlib.Path('../../data/lab_1')
input_data_folder = data_folder / 'input'
output_data_folder = data_folder / 'output'

csv_output = "output.csv"
csv_output_path = output_data_folder / csv_output
csv_output_columns = ["Iteration", "Distance", "Best Distance", "Worst Distance", "Difference"]

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


def hill_climbing(cities, max_no_improve=None):
    num_cities = len(cities)
    max_no_improve = max_no_improve or calculate_max_no_improve_tries(num_cities, 1.2, 20, 5)

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


def hill_climbing_multi_start(
        cities, num_starts=None, max_no_improve=None, interactive_plot=False, block=True, first_pause = 6, pause = 0.5,
        results_file_path = csv_output_path):
    num_starts = num_starts or round(len(cities) * 0.2)

    best_route = None
    best_distance = float('inf')
    worst_route = None
    worst_distance = float(0)
    results = []
    iteration = 0

    max_no_improve = max_no_improve or calculate_max_no_improve_tries(len(cities), 1.2, 20, 5)

    create_csv_file(results_file_path)

    print(f"Максимальна кількість ітерацій без покращень = {max_no_improve}")

    if interactive_plot:
        plt.ion()  # Вмикаємо інтерактивний режим

    for _ in range(num_starts):
        current_route, current_distance = hill_climbing(cities, max_no_improve)
        results.append(current_distance)

        if current_distance < best_distance:
            best_route, best_distance = current_route, current_distance
        if current_distance > worst_distance:
            worst_route, worst_distance = current_route, current_distance

        iteration += 1

        if iteration % 1 == 0:
            if interactive_plot:
                plot_pause = first_pause if iteration == 1 else pause
                plot_route(
                    best_route, f"Ітерація: {iteration}, Дистанція: {best_distance:.2f}",
                    interactive_plot=interactive_plot, block = block, pause = plot_pause)
            parsed_iteration = str(iteration).zfill(len(str(num_starts)))
            print(
                f"Ітерація = {parsed_iteration}: "
                f"Наявний маршрут (довжина = {current_distance:.2f}), "
                f"Найкращий маршрут (довжина = {best_distance:.2f}), "
                f"Найгірший маршрут (довжина = {worst_distance:.2f})"
            )
            write_row_csv_file([parsed_iteration, current_distance, best_distance, worst_distance, calculate_distance_difference(best_distance, worst_distance)])

    if interactive_plot:
        plt.ioff()  # Вимикаємо інтерактивний режим
        plt.show()

    return best_route, best_distance, worst_route, worst_distance, results


def plot_route(route, title, interactive_plot=False, block=True, pause=0.5):
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
    plt.plot(range(len(results)), sorted(results, reverse=True), marker='o', linestyle='-', color='green')
    plt.title("Графік збіжності цільової функції")
    plt.xlabel("Запуски")
    plt.ylabel("Найкраща довжина маршруту")
    plt.grid()
    plt.show()


def calculate_max_no_improve_tries(num, scale=1.25, min_count=20, multiplayer=5):
    fraction = max((int(num - (num * 0.1) ** scale)), 0)
    return (fraction + min_count) * multiplayer


def create_csv_file(file_path=csv_output_path, columns=None):
    if columns is None:
        columns = csv_output_columns
    with open(file_path, mode="w", newline='', encoding="utf-8") as file:  # Тип IO[str]
        writer = csv.writer(file)
        writer.writerow(columns)

def write_row_csv_file(row, file_path=csv_output_path):
    with open(file_path, mode="a", newline='', encoding="utf-8") as file:  # Тип IO[str]
        writer = csv.writer(file)
        writer.writerow(row)