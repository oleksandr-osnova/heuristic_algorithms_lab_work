from src.lab_1 import utils
import pathlib
import random
import matplotlib.pyplot as plt
import decimal

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

def generate_count_of_total_neighbours(num_cities):
    return (num_cities - 2) * (num_cities - 1) // 2

def generate_s_a_condition(delta, temperature):
    return decimal.Decimal(random.random()) < decimal.Decimal(-delta / temperature).exp()

def generate_initial_temperature(initial_temperature, route):
    # Використовуємо метод TRY для визначення оптимальної початкової температури
    while True:
        temperature = initial_temperature
        acceptance_ratio = try_method(route, temperature)
        if acceptance_ratio >= 0.9:
            break
        initial_temperature *= 1.1  # Підвищуємо температуру для досягнення необхідного порогу
    return initial_temperature

def try_method(current_route, temperature):
    num_cities = len(current_route)
    num_attempts = generate_count_of_total_neighbours(num_cities)
    accepted = 0
    for i in range(1, num_cities - 1):  # Починаємо з 1, щоб не змінювати стартову точку
        for j in range(i + 1, num_cities):  # j завжди більше за i
            delta = utils.incremental_distance_update(current_route, i, j)

            if generate_s_a_condition(delta, temperature):
                accepted += 1
    return accepted / num_attempts

def cooling_schedule(temperature, alpha=0.9):
    return temperature * alpha

def is_frozen(temperature, threshold=1e-3):
    return temperature < threshold

def is_frozen_converged(frozen_count, max_frozen=3):
    return frozen_count >= max_frozen

# fill climbing with simulated annealing
def h_c_with_s_a(cities, initial_temperature=100, cooling_rate=0.999, max_frozen=3):
    num_cities = len(cities)
    neighbors_count = generate_count_of_total_neighbours(num_cities)  # Кількість можливих 2-opt

    temperature = initial_temperature
    frozen_count = 0
    accepted = 0
    rejected = 0

    current_route = utils.generate_random_route(cities)
    current_distance = utils.calculate_distance(current_route)
    best_route, best_distance = current_route, current_distance


    while not is_frozen_converged(frozen_count, max_frozen):
        for i in range(1, num_cities - 1):  # Починаємо з 1, щоб не змінювати стартову точку
            for j in range(i + 1, num_cities):  # j завжди більше за i
                delta = utils.incremental_distance_update(current_route, i, j)

                if generate_s_a_condition(delta, temperature):
                    current_route, current_distance = utils.two_opt(current_route, i, j), current_distance + delta
                    if current_distance < best_distance:  # Якщо покращення знайдено
                        best_route, best_distance = current_route, current_distance

                    accepted += 1
                else:
                    rejected += 1

        if accepted >= neighbors_count:
            temperature = cooling_schedule(temperature, cooling_rate)
            accepted = 0
            rejected = 0
            frozen_count = 0

        if rejected >= 2 * neighbors_count:
            temperature = cooling_schedule(temperature, cooling_rate)
            accepted = 0
            rejected = 0
            frozen_count += 1

        # Умова замороження (низька температура)
        if is_frozen(temperature):
            break

    return best_route, best_distance

def h_c_with_s_a_multi_start(
        cities, num_starts=None, interactive_plot=False, block=True, first_pause = 6, pause = 0.5,
        results_file_path = csv_output_path, initial_temperature=100, cooling_rate=0.99):

    num_starts = num_starts or max(round(len(cities) * 0.05), 5)

    best_route = None
    best_distance = float('inf')
    worst_route = None
    worst_distance = float(0)
    results = []
    iteration = 0

    utils.create_csv_file(results_file_path)

    print(f"Початкова температура = {initial_temperature}")
    print(f"num_starts = {num_starts}")
    print(f"cooling_rate = {cooling_rate}")

    if interactive_plot:
        plt.ion()  # Вмикаємо інтерактивний режим

    for _ in range(num_starts):
        current_route, current_distance = h_c_with_s_a(cities, initial_temperature, cooling_rate)
        print(current_distance)
        results.append(current_distance)

        if current_distance < best_distance:
            best_route, best_distance = current_route, current_distance
        if current_distance > worst_distance:
            worst_route, worst_distance = current_route, current_distance

        iteration += 1

        if iteration % 1 == 0:
            if interactive_plot:
                plot_pause = first_pause if iteration == 1 else pause
                utils.plot_route(
                    best_route, f"Ітерація: {iteration}, Дистанція: {best_distance:.2f}",
                    interactive_plot=interactive_plot, block = block, pause = plot_pause)
            parsed_iteration = str(iteration).zfill(len(str(num_starts)))
            print(
                f"Ітерація = {parsed_iteration}: "
                f"Наявний маршрут (довжина = {current_distance:.2f}), "
                f"Найкращий маршрут (довжина = {best_distance:.2f}), "
                f"Найгірший маршрут (довжина = {worst_distance:.2f})"
            )
            utils.write_row_csv_file([
                parsed_iteration, current_distance, best_distance, worst_distance,
                utils.calculate_distance_difference(best_distance, worst_distance)
            ], csv_output_path)

    if interactive_plot:
        plt.ioff()  # Вимикаємо інтерактивний режим
        plt.show()

    return best_route, best_distance, worst_route, worst_distance, results
