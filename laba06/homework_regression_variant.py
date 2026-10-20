import matplotlib.pyplot as plt
import numpy as np


def linear_fit(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Оценка коэффициентов линейной регрессии методом наименьших квадратов."""
    slope = np.sum((x - x.mean()) * (y - y.mean())) / np.sum((x - x.mean()) ** 2)
    intercept = y.mean() - slope * x.mean()
    return float(intercept), float(slope)


def task_1_3() -> None:
    # Учебный пример: рост (см) и вес (кг) восьми взрослых людей.
    height = np.array([156, 162, 166, 171, 174, 178, 182, 188], dtype=float)
    weight = np.array([52, 58, 63, 66, 72, 75, 79, 88], dtype=float)

    n = len(height)
    b0, b1 = linear_fit(height, weight)
    predicted = b0 + b1 * height
    residuals = weight - predicted

    rss = np.sum(residuals ** 2)
    tss = np.sum((weight - weight.mean()) ** 2)
    r2 = 1 - rss / tss
    rse = np.sqrt(rss / (n - 2))

    print("Задания 1-3. Линейная регрессия веса по росту")
    print(f"Средний рост = {height.mean():.3f} см")
    print(f"Средний вес = {weight.mean():.3f} кг")
    print(f"b0 = {b0:.4f}; b1 = {b1:.4f}")
    print(f"Модель: ŷ = {b0:.4f} + {b1:.4f}x")
    print(f"RSS = {rss:.4f}; RSE = {rse:.4f} кг; R² = {r2:.4f}")
    print(f"{'Рост':>7} {'Вес':>7} {'Прогноз':>10} {'Остаток':>10} {'e²':>10}")
    for x, y, yh, e in zip(height, weight, predicted, residuals):
        print(f"{x:7.0f} {y:7.0f} {yh:10.2f} {e:10.2f} {e**2:10.2f}")

    grid = np.linspace(height.min() - 2, height.max() + 2, 200)
    plt.figure(figsize=(8, 5))
    plt.scatter(height, weight, label="Исходные данные")
    plt.plot(grid, b0 + b1 * grid, label=f"ŷ = {b0:.2f} + {b1:.3f}x")
    plt.xlabel("Рост, см")
    plt.ylabel("Вес, кг")
    plt.title("Линейная регрессия веса по росту")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()


def task_4() -> None:
    # Учебный пример: возраст (лет) и вес (кг) 16 человек.
    age = np.array([4, 6, 8, 11, 13, 15, 18, 21, 24, 28, 32, 37, 43, 49, 56, 63], dtype=float)
    weight = np.array([17, 22, 27, 34, 42, 50, 58, 64, 69, 73, 76, 81, 77, 84, 82, 80], dtype=float)

    n = len(age)
    z = np.log(age)
    v = np.log(weight)
    b0, b1 = linear_fit(z, v)
    alpha = np.exp(b0)
    gamma = b1

    v_predicted = b0 + b1 * z
    rss_log = np.sum((v - v_predicted) ** 2)
    r2_log = 1 - rss_log / np.sum((v - v.mean()) ** 2)
    rse_log = np.sqrt(rss_log / (n - 2))

    predicted = alpha * age ** gamma
    residuals = weight - predicted
    rss = np.sum(residuals ** 2)
    tss = np.sum((weight - weight.mean()) ** 2)
    r2 = 1 - rss / tss
    rse = np.sqrt(rss / (n - 2))

    print("\\Задание 4. Мультипликативная модель веса по возрасту")
    print(f"ln(alpha) = {b0:.4f}; alpha = {alpha:.4f}; gamma = {gamma:.4f}")
    print(f"Модель: ŷ = {alpha:.4f} · x^{gamma:.4f}")
    print(f"В логарифмах: R² = {r2_log:.4f}; RSE = {rse_log:.4f}")
    print(f"В исходных единицах: RSS = {rss:.2f}; RSE = {rse:.2f} кг; R² = {r2:.4f}")
    print(f"{'Возраст':>8} {'Вес':>7} {'Прогноз':>10} {'Остаток':>10}")
    for x, y, yh, e in zip(age, weight, predicted, residuals):
        print(f"{x:8.0f} {y:7.0f} {yh:10.2f} {e:10.2f}")

    grid = np.linspace(age.min(), age.max(), 300)
    plt.figure(figsize=(8, 5))
    plt.scatter(age, weight, label="Исходные данные")
    plt.plot(grid, alpha * grid ** gamma,
             label=rf"$\hat Y={alpha:.2f}X^{{{gamma:.3f}}}$")
    plt.xlabel("Возраст, лет")
    plt.ylabel("Вес, кг")
    plt.title("Мультипликативная модель веса по возрасту")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    task_1_3()
    task_4()
