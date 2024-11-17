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

data_folder = pathlib.Path('../../data/lab_2')
input_data_folder = data_folder / 'input'
output_data_folder = data_folder / 'output'

csv_output = "output.csv"
csv_output_path = output_data_folder / csv_output

def generate_s_a_condition(delta, temperature):
    # print(delta, temperature, decimal.Decimal(-delta / temperature).exp())
    return decimal.Decimal(random.random()) < decimal.Decimal(-delta / temperature).exp()

def try_method(current_route, temperature, num_attempts=100):
    accepted = 0
    for _ in range(num_attempts):
        delta = calculate_route_distance_difference(current_route, utils.generate_random_route(current_route))
        print('delta', delta)
        if generate_s_a_condition(delta, temperature):
            accepted += 1
    return accepted / num_attempts

# fill climbing with simulated annealing
def h_c_with_s_a(cities, initial_temperature=10000, cooling_rate=0.999):
    num_cities = len(cities)

    temperature = initial_temperature
    current_route = utils.generate_random_route(cities)
    current_distance = utils.calculate_distance(current_route)

    best_route, best_distance = current_route, current_distance

    while True:
        # Генерація випадкових індексів для застосування 2-opt
        i, j = sorted(random.sample(range(1, num_cities), 2))
        if j - i == 1:  # Пропускаємо сусідні міста
            continue

        # Обчислюємо зміну довжини маршруту
        delta = utils.incremental_distance_update(current_route, i, j)

        if delta < 0:  # Якщо покращення знайдено, оновлюємо кращий маршрут
            # print('better')
            current_route = utils.two_opt(current_route, i, j)
            current_distance += delta
            best_route, best_distance = current_route, current_distance
        elif generate_s_a_condition(delta, temperature):
            # print('random')
            current_route = utils.two_opt(current_route, i, j)
            current_distance += delta

        print('try_method', try_method(current_route, temperature), temperature)
        if try_method(current_route, temperature) >= 0.8:
            temperature *= cooling_rate
            print('try_method', temperature, best_distance)

        # Умова замороження (низька температура)
        if temperature < 1e-1:
            break

    return best_route, best_distance

def h_c_with_s_a_multi_start(
        cities, num_starts=10, interactive_plot=False, block=True, first_pause = 6, pause = 0.5,
        results_file_path = csv_output_path, initial_temperature=100, cooling_rate=0.99):

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
