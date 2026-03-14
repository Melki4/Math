# ----------------------------------------------------------------------
# Лабораторная работа. Вариант 12.
# ----------------------------------------------------------------------

import numpy as np
import sympy
import math
import scipy
from scipy.integrate import quad
import matplotlib.pyplot as plt

import warnings

# Ignore all warnings
warnings.filterwarnings("ignore")

# ----------------------------------------------------------------------
# Задание 1. Создание матриц с помощью NumPy
# ----------------------------------------------------------------------

# 1. Матрица 10x10, заполненная вещественными единицами.
matrix_ones = np.ones((10, 10), dtype=float)

# 2. Единичная матрица 10x10.
matrix_identity = np.identity(10, dtype=float)

# Вывод созданных матриц
print(matrix_ones, '\n')
print(matrix_identity, '\n')

# ----------------------------------------------------------------------
# Задание 2. Вычисление определителя матрицы 4x4
# ----------------------------------------------------------------------

# Исходная матрица
matrix_a = np.array([[5, 1, 2, 2],
                     [5, 2, 4, 1],
                     [4, 1, 3, 3],
                     [2, 1, 3, 6]])
matrix_det = np.linalg.det(matrix_a)
print("Определитель матрицы:", matrix_det)

# ----------------------------------------------------------------------
# Задание 3. Упрощение выражения и вычисление его значения
# ----------------------------------------------------------------------

# Определяем символьные переменные
x = sympy.Symbol('x')
y = sympy.Symbol('y')

# Заданное выражение: (x - 5y)^2 + (4x/y)*(x + 4)
expression = (x - 5*y)**2 + 4*x/y*(x + 4)

# Упрощение выражения 
expression = sympy.simplify(expression)
expression = sympy.expand(expression)
expression = sympy.powsimp(expression)

print(expression)

# Вычисление значения при x = 1.038 и y = sqrt(7)
val = expression.subs(x, 1.038).subs(y, math.sqrt(7))
print("Значение выражения при x=1.038, y=√7:", val)

# ----------------------------------------------------------------------
# Задание 4. Частные производные
# ----------------------------------------------------------------------

# Производная по x
der_x = sympy.diff(expression, x)
# Производная по y
der_y = sympy.diff(expression, y)

print("Частная производная по x:", der_x)
print("Частная производная по y:", der_y)

# ----------------------------------------------------------------------
# Задание 5. Решение системы линейных уравнений с помощью NumPy
# ----------------------------------------------------------------------

# Создание случайной матрицы A 4x4 из целых чисел в диапазоне [-3, 5]
A = np.random.randint(-3, 6, (4, 4))

# Создание вектора-столбца B 4x1 из случайных чисел
B = np.random.uniform(-3, 5, (4, 1))

# Проверка существования решения (определитель не равен нулю)
if np.linalg.det(A) - 1e-6 < 0:
    print("Матрица вырождена, система не имеет единственного решения.")
else:
    # Два способа решения:
    # 1) через обратную матрицу
    X_inv = np.matmul(scipy.linalg.inv(A), B)
    # 2) через встроенный решатель
    X_solve = np.linalg.solve(A, B)

    print("Матрица A:\n", A)
    print("Вектор B:\n", B)
    print("Решение (через обратную матрицу):\n", X_inv)
    print("Решение (через solve):\n", X_solve)

# ----------------------------------------------------------------------
# Задание 6. Вычисление интеграла ∫_{-1/2}^{1/2} dx / √(1-x²)
# ----------------------------------------------------------------------

# Численное интегрирование с SciPy
f = lambda x: 1 / np.sqrt(1 - x**2)
result_6_num = quad(f, -1/2, 1/2)[0]
print("Численное значение интеграла ∫_{-1/2}^{1/2} dx / √(1-x²) (SciPy):", result_6_num)

# Символьное интегрирование с SymPy
expr_6 = 1 / sympy.sqrt(1 - x**2)
result_6_sym = sympy.integrate(expr_6, (x, -1/2, 1/2))
print("Символьное значение интеграла ∫_{-1/2}^{1/2} dx / √(1-x²) (SymPy):", result_6_sym)

# ----------------------------------------------------------------------
# Задание 7. Вычисление интеграла ∫₀^∞ cos(2x) dx
# ----------------------------------------------------------------------

# Численное интегрирование (выдаст предупреждение о расходимости)
h = lambda x: np.cos(2 * x)
result_7_num = quad(h, 0, np.inf)[0]
print("Численное значение ∫₀^∞ cos(2x) dx (SciPy) для интеграла:", result_7_num)

# Символьное интегрирование (вернёт невычисленный интеграл)
expr_7 = sympy.cos(2 * x)
result_7_sym = sympy.integrate(expr_7, (x, 0, sympy.oo))
print("Символьное представление ∫₀^∞ cos(2x) dx (SymPy):", result_7_sym)

# ------------------------------------------------------------
# Построение графиков функций с помощью Mathplotlib
#
# Построить в одной системе координат графики функций:
#   y = sin(x + pi/3),  y = 2*x
# Оси координат должны быть подписаны, графики разного цвета,
# должна быть выведена легенда.
# ------------------------------------------------------------

# Создание объектов артборда и холста
plt.figure(figsize=(8, 5), dpi=80)
ax = plt.subplot(111)

# Удалить правую и верхнюю прямоугольные границы
ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')

# Установить направление данных на координатной оси
ax.xaxis.set_ticks_position('bottom')
ax.spines['bottom'].set_position(('data', 0))
ax.yaxis.set_ticks_position('left')
ax.spines['left'].set_position(('data', 0))

# Подготовить данные
X = np.linspace(-2*np.pi, 2*np.pi, 256, endpoint=True)
C = np.sin(X + np.pi/3)
L = 2*X

plt.plot(X, C, color="blue", linewidth=2.5, linestyle="-", label="sin(x + pi/3)")
plt.plot(X, L, color="red", linewidth=2.5, linestyle="-", label="2*x")

plt.xlim(X.min() * 1.1, X.max() * 1.1)
plt.xticks([-2*np.pi, -3*np.pi/2, -np.pi, -np.pi/2, 0, np.pi/2, np.pi, 3*np.pi/2, 2*np.pi])

plt.ylim(C.min() * 1.1, C.max() * 1.1)
plt.yticks([-2, -1, 1, 2])

from scipy.optimize import fsolve
x_cross = fsolve(lambda x: np.sin(x + np.pi/3) - 2*x, 0)[0]
y_cross = 2 * x_cross

ax.scatter(x=x_cross, y=y_cross, c='orange', s=50, zorder=5,)

plt.legend(loc='upper left', frameon=False)
plt.grid(True)
plt.show()