import numpy as np
import matplotlib.pyplot as plt

# Визначаємо функцію для обчислення довжини шляху
def path_distance(path, distance_matrix):
    return sum(distance_matrix[path[i-1], path[i]] for i in range(len(path)))

# Ініціалізуємо випадковий шлях
def random_path(num_cities):
    path = np.arange(num_cities)
    np.random.shuffle(path)
    return path

# Реалізація одного кроку Hill Climbing з перестановкою
def hill_climbing_step(path, distance_matrix):
    best_distance = path_distance(path, distance_matrix)
    best_path = path.copy()

    # Пробуємо переставити два міста місцями
    for i in range(len(path) - 1):
        for j in range(i + 1, len(path)):
            new_path = path.copy()
            new_path[i], new_path[j] = new_path[j], new_path[i]
            new_distance = path_distance(new_path, distance_matrix)

            if new_distance < best_distance:
                best_distance = new_distance
                best_path = new_path

    return best_path, best_distance

# Функція мультистартового Hill Climbing
def multistart_hill_climbing(distance_matrix, num_starts):
    best_solution = None
    best_distance = float('inf')
    results = []

    for _ in range(num_starts):
        path = random_path(len(distance_matrix))
        distance = path_distance(path, distance_matrix)

        # Виконуємо Hill Climbing поки не стабілізується розв'язок
        while True:
            new_path, new_distance = hill_climbing_step(path, distance_matrix)
            if new_distance < distance:
                path, distance = new_path, new_distance
            else:
                break

        results.append(distance)

        if distance < best_distance:
            best_distance = distance
            best_solution = path

    return best_solution, best_distance, results

# Створення випадкової матриці відстаней (симетрична для TSP)
def create_random_distance_matrix(num_cities):
    matrix = np.random.rand(num_cities, num_cities) * 100
    matrix = (matrix + matrix.T) / 2  # Робимо її симетричною
    np.fill_diagonal(matrix, 0)
    return matrix

# Побудова гістограми отриманих рішень
def plot_histogram(results):
    plt.hist(results, bins=10, color='skyblue', edgecolor='black')
    plt.title("Гістограма отриманих рішень")
    plt.xlabel("Відстань")
    plt.ylabel("Частота")
    plt.show()

# Побудова графіку збіжності цільової функції для кількох запусків
def convergence_plot(distance_matrix, num_starts):
    for _ in range(5):  # Наприклад, показати 5 запусків
        path = random_path(len(distance_matrix))
        distances = []

        # Кроки Hill Climbing
        while True:
            new_path, new_distance = hill_climbing_step(path, distance_matrix)
            distances.append(path_distance(path, distance_matrix))
            if new_distance < path_distance(path, distance_matrix):
                path, distance = new_path, new_distance
            else:
                break

        plt.plot(distances, label=f"Запуск {_ + 1}")

    plt.title("Графік збіжності цільової функції")
    plt.xlabel("Крок")
    plt.ylabel("Відстань")
    plt.legend()
    plt.show()

# Головна функція для запуску обчислювального експерименту
def main():
    num_cities = 10
    num_starts = 100

    # Генеруємо матрицю відстаней для міст
    distance_matrix = create_random_distance_matrix(num_cities)

    # Виконуємо мультистартовий Hill Climbing
    best_solution, best_distance, results = multistart_hill_climbing(distance_matrix, num_starts)
    worst_distance = max(results)
    difference = worst_distance - best_distance

    print(f"Різниця між найкращим і найгіршим рішенням: {difference}")
    print(f"Найкраща відстань: {best_distance}")
    print(f"Найгірша відстань: {worst_distance}")

    # Побудова гістограми
    plot_histogram(results)

    # Побудова графіку збіжності цільової функції
    convergence_plot(distance_matrix, num_starts)

# Запуск програми
if __name__ == "__main__":
    main()
