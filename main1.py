import math
import random
import matplotlib.pyplot as plt

'''Инициализация общих констант'''
max_iter = 10000  # предельное количество итераций
epsilon = 1 * math.pow(math.e, -6) # точность расчета (чем меньше, тем лучше)
rate = 0.01  # скорость спуска

lambda_val = 0.1  # коэффициент забывания
alpha = 0.999

'''Инициализация вспомогательных констант'''
gamma = 1 - lambda_val
eta = (1 - gamma) * rate


'''Возвращает вектор искомых значений функции. Выводит число итераций или отсутствие решений'''
def print_message(mass, count, flag):
    if flag:
        print("Число итераций = {:d}".format(count))
        print(mass)
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

    print_message(W1, count_flag, flag)
    return count_flag


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

    print_message(W1, count_flag, flag)
    return count_flag


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

    print_message(W1, count_flag, flag)
    return count_flag


def grad_descent_RMSProp():
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

    print_message(W1, count_flag, flag)
    return count_flag


def grad_descent_AdaDelta():
    Delta = 0.01  # коэффициент забывания
    delta_1 = 0.01

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
            delta_1 = Grad_W[i] * (math.sqrt(Delta) + epsilon) / math.sqrt(G[i] + epsilon)
            Delta = alpha * Delta + (1 - alpha) * delta_1 * delta_1
            W1[i] = w[i] - delta_1

        if abs(w[0] - W1[0]) < epsilon and abs(w[1] - W1[1]) < epsilon:
            flag = True
            break
        else:
            w[0] = W1[0]
            w[1] = W1[1]

        count_flag = count_flag + 1

    print_message(W1, count_flag, flag)
    return count_flag


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

    print_message(W1, count_flag, flag)
    return count_flag

grad_descent()
# grad_descent_momentum()
# grad_descent_NAG()
# grad_descent_RMSProp()
# grad_descent_AdaDelta()
# grad_descent_Adam()

# data = []
# res = []
# for i in range(100):
#     res.append(grad_descent())
# res1 = []
# for i in range(100):
#     res1.append(grad_descent_momentum())
# res2 = []
# for i in range(100):
#     res2.append(grad_descent_NAG())
# res3 = []
# for i in range(100):
#     res3.append(grad_descent_RMSProp())
# res4 = []
# for i in range(100):
#     res4.append(grad_descent_AdaDelta())
# res5 = []
# for i in range(100):
#     res5.append(grad_descent_Adam())
#
# data.append(res)
# data.append(res1)
# data.append(res2)
# data.append(res3)
# data.append(res4)
# data.append(res5)
#
# fig, ax = plt.subplots()
# bar = ['Classic', 'Momentum', 'NAG', 'RMSProp', 'AdaDelta', 'Adam']
# ax.set_xlabel('Данные по каждому методу')
# ax.set_xticklabels(bar)
# ax.set_ylabel('Количество итераций')
# ax.set_title('Усатая диаграмма')
# ax.boxplot(data)
# plt.show()
