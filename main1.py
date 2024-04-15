import math
import random
import matplotlib.pyplot as plt

'''Инициализация общих констант'''
max_iter = 10000  # предельное количество итераций
epsilon = 1 * math.pow(math.e, -6) # точность расчета
rate = 0.01  # скорость спуска

lambda_val = 0.1  # коэффициент забывания
alpha = 0.999

'''Инициализация вспомогательных констант'''
gamma = 1 - lambda_val
eta = (1 - gamma) * rate


'''Инициализация начального вектора со случайными значениями'''


'''Возвращает вектор искомых значений функции. Выводит число итераций или отсутствие решений'''
def print_message(mass, count, flag):
    if flag:
        print("Число итераций = {:d}".format(count))
        return mass
    else:
        print("Решение не найдено")


def grad_descent():
    w = []

    '''заполнение начального вектора случайными числами'''
    for i in range(2):
        w.append(random.randint(-100, 100))

    Grad_W = [0, 0]  # вектор частных производных
    W1 = [0, 0]  # вектор новых значений координат точки

    '''взятие частных производных и проверка условия минимума'''
    count_flag = 0
    flag = False
    while (count_flag < max_iter):
        Grad_W[0] = (4 * w[0] - 4)
        Grad_W[1] = (4 * w[1] - 8)

        for i in range(len(Grad_W)):
            W1[i] = (w[i] - rate * Grad_W[i])

        if abs(w[0] - W1[0]) < epsilon and abs(w[1] - W1[1]) < epsilon:
            flag = True
            break
        else:
            w[0] = W1[0]
            w[1] = W1[1]

        count_flag = count_flag + 1

    return print_message(W1, count_flag, flag)


def grad_descent_momentum():
    w = []
    v = []

    '''заполнение начального вектора случайными числами'''
    for i in range(2):
        w.append(random.randint(-100, 100))
        v.append(random.randint(-100, 100))

    Grad_W = [0, 0]  # вектор частных производных
    W1 = [0, 0]  # вектор новых значений координат точки

    '''взятие частных производных и проверка условия минимума'''
    count_flag = 0
    flag = False
    while (count_flag < max_iter):
        Grad_W[0] = (4 * w[0] - 4)
        Grad_W[1] = (4 * w[1] - 8)

        for i in range(len(Grad_W)):
            v[i] = gamma * v[i] + eta * Grad_W[i]
            W1[i] = w[i] - v[i]

        if abs(w[0] - W1[0]) < epsilon and abs(w[1] - W1[1]) < epsilon:
            flag = True
            break
        else:
            w[0] = W1[0]
            w[1] = W1[1]

        count_flag = count_flag + 1

    return print_message(W1, count_flag, flag)


def grad_descent_NAG():
    w = []
    v = []

    '''заполнение начального вектора случайными числами'''
    for i in range(2):
        w.append(random.randint(-100, 100))
        v.append(random.randint(-100, 100))

    Grad_W = [0, 0]  # вектор частных производных
    W1 = [0, 0]  # вектор новых значений координат точки

    '''взятие частных производных и проверка условия минимума'''
    count_flag = 0
    flag = False
    while (count_flag < max_iter):
        Grad_W[0] = (4 * w[0] - 4) - gamma * v[0]
        Grad_W[1] = (4 * w[1] - 8) - gamma * v[1]

        for i in range(len(Grad_W)):
            v[i] = gamma * v[i] + eta * Grad_W[i]
            W1[i] = w[i] - v[i]

        if abs(w[0] - W1[0]) < epsilon and abs(w[1] - W1[1]) < epsilon:
            flag = True
            break
        else:
            w[0] = W1[0]
            w[1] = W1[1]

        count_flag = count_flag + 1

    return print_message(W1, count_flag, flag)


def grad_descent_RMSPro():
    w = []
    G = []

    '''заполнение начального вектора случайными числами'''
    for i in range(2):
        w.append(random.randint(-100, 100))
        G.append(0)

    Grad_W = [0, 0]  # вектор частных производных
    W1 = [0, 0]  # вектор новых значений координат точки

    '''взятие частных производных и проверка условия минимума'''
    count_flag = 0
    flag = False
    while (count_flag < max_iter):
        Grad_W[0] = (4 * w[0] - 4)
        Grad_W[1] = (4 * w[1] - 8)

        for i in range(len(Grad_W)):
            G[i] = gamma * G[i] + (1 - gamma) * Grad_W[i] * Grad_W[i]
            W1[i] = w[i] - (1 - gamma) * Grad_W[i] / math.sqrt(G[i] + epsilon)

        if abs(w[0] - W1[0]) < epsilon and abs(w[1] - W1[1]) < epsilon:
            flag = True
            break
        else:
            w[0] = W1[0]
            w[1] = W1[1]

        count_flag = count_flag + 1

    return print_message(W1, count_flag, flag)


def grad_descent_AdaDelta():
    Delta = 0.01  # коэффициент забывания
    delta1 = 0.01

    w = []
    G = []

    '''заполнение начального вектора случайными числами'''
    for i in range(2):
        w.append(random.randint(-100, 100))
        G.append(0)

    Grad_W = [0, 0]  # вектор частных производных
    W1 = [0, 0]  # вектор новых значений координат точки

    '''взятие частных производных и проверка условия минимума'''
    count_flag = 0
    flag = False
    while (count_flag < max_iter):
        Grad_W[0] = (4 * w[0] - 4)
        Grad_W[1] = (4 * w[1] - 8)

        for i in range(len(Grad_W)):
            G[i] = alpha * G[i] + (1 - alpha) * Grad_W[i] * Grad_W[i]
            delta1 = Grad_W[i] * (math.sqrt(Delta) + epsilon) / math.sqrt(G[i] + epsilon)
            Delta = alpha * Delta + (1 - alpha) * delta1 * delta1
            W1[i] = w[i] - delta1

        if abs(w[0] - W1[0]) < epsilon and abs(w[1] - W1[1]) < epsilon:
            flag = True
            break
        else:
            w[0] = W1[0]
            w[1] = W1[1]

        count_flag = count_flag + 1

    return print_message(W1, count_flag, flag)


def grad_descent_Adam():
    w = []
    v = []
    G = []

    '''заполнение начального вектора случайными числами'''
    for i in range(2):
        w.append(random.randint(-100, 100))
        v.append(0)
        G.append(0)

    Grad_W = [0, 0]  # вектор частных производных
    W1 = [0, 0]  # вектор новых значений координат точки

    '''взятие частных производных и проверка условия минимума'''
    count_flag = 1
    flag = False
    while (count_flag < max_iter):
        Grad_W[0] = (4 * w[0] - 4)
        Grad_W[1] = (4 * w[1] - 8)

        for i in range(len(Grad_W)):
            v[i] = gamma * v[i] + (1 - gamma) * Grad_W[i]
            G[i] = alpha * G[i] + (1 - alpha) * Grad_W[i] * Grad_W[i]
            v_val = v[i] / (1 - math.pow(gamma, count_flag))
            g_val = G[i] / (1 - math.pow(alpha, count_flag))
            W1[i] = w[i] - rate * v_val / (math.sqrt(g_val) + epsilon)

        if abs(w[0] - W1[0]) < epsilon and abs(w[1] - W1[1]) < epsilon:
            flag = True
            break
        else:
            w[0] = W1[0]
            w[1] = W1[1]

        count_flag = count_flag + 1

    return print_message(W1, count_flag, flag)


resalt_classic = []
for i in range(1000):
    resalt_classic.append(grad_descent()[0])
for i in range(1000):
    print(resalt_classic[i])


fig = plt.figure(figsize=(10, 10))
# Creating plot
plt.boxplot(resalt_classic)
# show plot
plt.show()

# grad_descent_momentum()
# grad_descent_NAG()
# grad_descent_RMSPro()
# grad_descent_AdaDelta()
# grad_descent_Adam()
