import numpy as np
import matplotlib.pyplot as plt

# установим размер графика
fig = plt.figure(figsize=(10, 10))

# создадим последовательность из 1000 точек в интервале от -5 до 5
# для осей w1 и w2
w1 = np.linspace(-15, 15, 1000)
w2 = np.linspace(-15, 15, 1000)

# создадим координатную плоскость из осей w1 и w2
w1, w2 = np.meshgrid(w1, w2)

# пропишем функцию
f = 2 * (w1 ** 2) + 2 * (w2 ** 2) - 4 * w1 - 8 * w2

# создадим трехмерное пространство
ax = fig.add_subplot(projection='3d')

# выведем график функции, alpha задает прозрачность
ax.plot_surface(w1, w2, f, alpha=0.8, cmap='Wistia')

# выведем точку A с координатами (3, 4, 25) и подпись к ней
ax.scatter(1, 2, -10, c='red', marker='^', s=100)
ax.text(3, 3.5, 28, 'A', size=25)

# укажем подписи к осям
ax.set_xlabel('w1', fontsize=15)
ax.set_ylabel('w2', fontsize=15)
ax.set_zlabel('f(w1, w2)', fontsize=15)

# выведем результат
plt.show()