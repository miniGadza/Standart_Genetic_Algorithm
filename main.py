import math
import random
import numpy as np


def func_one(XY1, index):
    XY = XY1.copy()
    x = XY[index, 0]
    y = XY[index, 1]
    result = (0.1 * x ** 2 + 0.1 * y ** 2 - 4 * math.cos(0.8 * x) - 4 * math.cos(0.8 * y) + 8)
    return -result


def func_two(XY1, index):
    XY = XY1.copy()
    A1 = math.pi / 4
    kx = 1.5
    ky = 0.8
    x = XY[index, 0]
    y = XY[index, 1]
    A = x * math.cos(A1) - y * math.cos(A1)
    B = x * math.sin(A1) + y * math.sin(A1)
    result = (0.1 * kx * A) + (0.1 * ky * B) - 4 * math.cos(0.8 * kx * A) - 4 * math.cos(0.8 * ky * B) + 8
    return -result


def func_three(XY1, index):
    XY = XY1.copy()
    x = XY[index, 0]
    y = XY[index, 1]
    result = 100 * (y - x ** 2) ** 2 + (1 - x) ** 2
    return -result


def func_four(XY1, index):
    XY = XY1.copy()
    x = XY[index, 0]
    y = XY[index, 1]
    result = ((-10) / 0.005 * (x ** 2 + y ** 2) - math.cos(x) * math.cos(y / math.sqrt(2)) + 2) + 10
    return -result


def func_five(XY1, index):
    XY = XY1.copy()
    x = XY[index, 0]
    y = XY[index, 1]
    result = (-100 / (100 * (x ** 2 - y) ** 2 + (1 - x) ** 2 + 1)) + 1
    return -result


def func_six(XY1, index):
    XY = XY1.copy()
    x = XY[index, 0]
    y = XY[index, 1]
    result = - ((1 - math.sin(math.sqrt(x ** 2 + y ** 2)) * math.sin(math.sqrt(x ** 2 + y ** 2))) / (
                1 + 0.001 * (x ** 2 + y ** 2)))
    return -result


def func_seven(XY1, index):  # Диапазон [-2.5 ; 2.5]
    XY = XY1.copy()
    x = XY[index, 0]
    y = XY[index, 1]
    result = 0.5 * (x ** 2 + y ** 2) * (
                2 * 0.8 + 0.8 * math.cos(1.5 * x) * math.cos(3.14 * y) + 0.8 * math.cos(math.sqrt(5) * x) * math.cos(
            3.5 * y))
    return -result


def func_eight(XY1, index):  # Диапазон [-5 ; 5]
    XY = XY1.copy()
    x = XY[index, 0]
    y = XY[index, 1]
    result = 0.5 * (x ** 2 + y ** 2) * (
                2 * 0.8 + 0.8 * math.cos(1.5 * x) * math.cos(3.14 * y) + 0.8 * math.cos(math.sqrt(5) * x) * math.cos(
            3.5 * y))
    return -result


def func_nine(XY1, index):
    XY = XY1.copy()
    x = XY[index, 0]
    y = XY[index, 1]
    result = (x ** 2 * abs(math.sin(2 * x)) + y ** 2 * abs(math.sin(2 * y)) - (1 / (5 * x ** 2 + 5 * y ** 2 + 0.2)) + 5)
    return -result


def func_ten(XY1, index):
    XY = XY1.copy()
    x = XY[index, 0]
    y = XY[index, 1]
    result = (0.5 * (x ** 2 + x * y + y ** 2) * (
                1 + 0.5 * math.cos(1.5 * x) * math.cos(3.2 * x * y) * math.cos(3.14 * y) + 0.5 * math.cos(
            2.2 * x) * math.cos(4.8 * x * y) * math.cos(3.5 * y)))
    return -result


def func_eleven(XY1, index):
    XY = XY1.copy()
    x = XY[index, 0]
    y = XY[index, 1]
    result = -((1 / ((x - 1) ** 2 + 0.2)) + (1 / (2 * (x - 2) ** 2 + 0.15)) + (1 / (3 * (x - 3) ** 2 + 0.3))) * (
                (1 / ((y - 1) ** 2 + 0.2)) + (1 / (2 * (y - 2) ** 2 + 0.15)) + (1 / (3 * (y - 3) ** 2 + 0.3)))
    return -result


def func_twelve(XY1, index):
    XY = XY1.copy()
    x = XY[index, 0]
    y = XY[index, 1]
    result = -(((1 / ((x - 1) ** 2 + 0.2)) + (1 / (2 * (x - 2) ** 2 + 0.15)) + (1 / (3 * (x - 3) ** 2 + 0.3))) + (
                (1 / ((y - 1) ** 2 + 0.2)) + (1 / (2 * (y - 2) ** 2 + 0.15)) + (1 / (3 * (y - 3) ** 2 + 0.3))))
    return -result


def func_thirteen(XY1, index):
    XY = XY1.copy()
    x = XY[index, 0]
    y = XY[index, 1]
    Arr1 = [[0 for j in range(25)] for i in range(2)]
    Arr = np.array(Arr1, dtype=float)
    adder = -32
    count = 1
    for j in range(25):
        if (count == 1):
            Arr[0, j] = -32
            count += 1
            continue
        if (count == 6):
            count = 2
            Arr[0, j] = -32
            continue
        Arr[0, j] = Arr[0, j - 1] + 16
        count += 1
    count = 1
    adder = -32
    for j in range(25):
        if (count == 6):
            adder += 16
            count = 1
        Arr[1, j] = adder
        count += 1
    sum_j = 0
    sum_i = 0
    for j in range(25):
        sum_i = 0
        for i in range(2):
            if (i == 0):
                sum_i = sum_i + (x - Arr[i, j]) ** 4
            if (i == 1):
                sum_i = sum_i + (y - Arr[i, j]) ** 4
        sum_j = sum_j + 1 / (j + 1 + sum_i)
    result = 1 / (1 / 500 + sum_j)
    return -result


def Tournament(sizePopulation, sizeBit, ArrayOdinCount1, Matrix1, Array_parents):
    ArrayOdinCount = ArrayOdinCount1.copy()
    Matrix = Matrix1.copy()
    Array_parents1 = Array_parents.copy()
    a = 0
    rand = 0
    while a < 2 * sizePopulation:
        firstInd = random.randint(0, sizePopulation - 1)
        secondInd = random.randint(0, sizePopulation - 1)

        if ArrayOdinCount[firstInd] > ArrayOdinCount[secondInd]:
            for i in range(sizeBit):
                Array_parents1[a, i] = Matrix[firstInd, i]
        if ArrayOdinCount[firstInd] < ArrayOdinCount[secondInd]:
            for i in range(sizeBit):
                Array_parents1[a, i] = Matrix[secondInd, i]
        if ArrayOdinCount[firstInd] == ArrayOdinCount[secondInd]:  # Если попадутся одинаковые индивиды
            rand = random.randint(0, 1)
            if rand == 0:
                for i in range(sizeBit):
                    Array_parents1[a, i] = Matrix[firstInd, i]
            else:
                for i in range(sizeBit):
                    Array_parents1[a, i] = Matrix[secondInd, i]
        a += 1
    return Array_parents1


def Proportional(sizePopulation, sizeBit, Indiv1, Matrix1, Array_parents1):
    Indiv = Indiv1.copy()
    Matrix = Matrix1.copy()
    Array_parents = Array_parents1.copy()
    counter_while = 0
    rand = random.random()
    while (counter_while < sizePopulation * 2):
        pArr = [0] * sizePopulation
        p = np.array(pArr, dtype=float)
        sum_indiv = 0
        for i in range(sizePopulation):
            sum_indiv += Indiv[i]
        for i in range(sizePopulation):
            p[i] = Indiv[i] / sum_indiv  # Вероятность каждого
        xArr = [0] * sizePopulation
        x = np.array(xArr, dtype=float)  # Отрезки с вероятностями
        for i in range(sizePopulation):
            if (i == 0):
                x[i] = p[i]
                continue
            x[i] = x[i - 1] + p[i]
        for i in range(sizePopulation - 1):
            if ((rand > x[i]) and (rand < x[i + 1])):
                for j in range(sizeBit):
                    Array_parents[counter_while, j] = Matrix[i, j]
        counter_while += 1
    return Array_parents


def Rank(sizePopulation, sizeBit, ResultFunction1, Matrix1, Array_parents1):
    ResultFunction = ResultFunction1.copy()
    Matrix = Matrix1.copy()
    Array_parents = Array_parents1.copy()
    ArrayRank_buff = [0 for i in range(sizePopulation)]
    ArrayRank = np.array(ArrayRank_buff, dtype=float)
    ArrayRank_buff2 = [0 for i in range(sizePopulation)]
    ArrayRank2 = np.array(ArrayRank_buff2, dtype=float)
    ArrayRank_buff3 = [0 for i in range(sizePopulation)]
    ArrayRank3 = np.array(ArrayRank_buff3, dtype=float)
    for i in range(sizePopulation):
        ArrayRank[i] = i
        ArrayRank2[i] = i + 1
    for i in range(int(sizePopulation * sizePopulation / 10)):
        for j in range(sizePopulation - 1):
            if ResultFunction[j] > ResultFunction[j + 1]:
                remember = ResultFunction[j]
                ResultFunction[j] = ResultFunction[j + 1]
                ResultFunction[j + 1] = remember

                remember = ArrayRank[j]
                ArrayRank[j] = ArrayRank[j + 1]
                ArrayRank[j + 1] = remember
    start = 0
    end = 0
    for i in range(sizePopulation - 1):
        rankSum = 0
        count = 0
        for j in range(i + 1, sizePopulation - 1):
            if ResultFunction[i] == ResultFunction[j]:
                count += 1
                end = j
            else:
                start = i
        if count != 0:
            for x in range(start, end + 1):
                rankSum += ArrayRank2[x]
            result = rankSum / (count + 1)
            for x in range(start, end + 1):
                ArrayRank2[x] = rankSum
        else:
            i += count + 1
    for i in range(sizePopulation):
        for j in range(sizePopulation):
            if i == ArrayRank[j]:
                ArrayRank3[i] = ArrayRank2[j]
    Array_parents = Proportional(sizePopulation, sizeBit, ArrayRank3, Matrix, Array_parents)
    return Array_parents


def SinglePointCrossover(sizePopulation, sizeBit, Array_potomki1, Array_parents1):
    Array_parents = Array_parents1.copy()
    Array_potomki = Array_potomki1.copy()
    counter_while = 0
    counter_massive = 0
    PointArr = [0] * sizePopulation
    Point = np.array(PointArr, dtype=int)
    while (counter_while < sizePopulation * 2):
        rand = random.randint(1, sizeBit)
        Point[counter_massive] = rand
        for i in range(rand):
            Array_potomki[counter_massive, i] = Array_parents[counter_while, i]
        for i in range(rand, sizeBit):
            Array_potomki[counter_massive, i] = Array_parents[counter_massive + 1, i]
        counter_massive += 1
        counter_while += 2
    return Array_potomki


def TwoPointCrossover(sizePopulation, sizeBit, Array_potomki1, Array_parents1):
    Array_potomki = Array_potomki1.copy()
    Array_parents = Array_parents1.copy()
    counterWhile = 0
    counterMassive = 0
    while (counterWhile < sizePopulation * 2):
        rand1 = random.randint(1, sizeBit - 1)
        rand2 = random.randint(1, sizeBit - 1)
        if rand1 == rand2:
            rand1 = sizeBit - rand2
        if rand1 > rand2:
            remember = rand1
            rand1 = rand2
            rand2 = remember
        for i in range(rand1):
            Array_potomki[counterMassive, i] = Array_parents[counterWhile, i]
        for i in range(rand1, rand2):
            Array_potomki[counterMassive, i] = Array_parents[counterWhile + 1, i]
        for i in range(rand2, sizeBit):
            Array_potomki[counterMassive, i] = Array_parents[counterWhile, i]
        counterMassive += 1
        counterWhile += 2
    return Array_potomki


def Uniform(sizePopulation, sizeBit, Array_potomki1, Array_parents1):
    Array_potomki = Array_potomki1.copy()
    Array_parents = Array_parents1.copy()
    counterWhile = 0
    counterMassive = 0
    while (counterWhile < sizePopulation * 2):
        for i in range(sizeBit):
            rand1 = random.randint(0, 1)
            if rand1 == 0:
                Array_potomki[counterMassive, i] = Array_parents[counterWhile, i]
            if rand1 == 1:
                Array_potomki[counterMassive, i] = Array_parents[counterWhile + 1, i]
        counterMassive += 1
        counterWhile += 2
    return Array_potomki


def MutationWeak(sizePopulation, sizeBit, Array_potomki1, Array_mutation1, Matrix1):
    Array_mutation = Array_mutation1.copy()
    Matrix = Matrix1.copy()
    Array_potomki = Array_potomki1.copy()
    for i in range(sizePopulation):
        for j in range(sizeBit):
            randMut = random.random()
            if (randMut < (1 / sizeBit * 3)):
                if Array_potomki[i, j] == 1:
                    Array_mutation[i, j] = 0
                else:
                    Array_mutation[i, j] = 1
            else:
                Array_mutation[i, j] = Array_potomki[i, j]
    return Array_mutation


def MutationAverage(sizePopulation, sizeBit, Array_potomki1, Array_mutation1, Matrix1):
    Array_mutation = Array_mutation1.copy()
    Matrix = Matrix1.copy()
    Array_potomki = Array_potomki1.copy()
    for i in range(sizePopulation):
        for j in range(sizeBit):
            randMut = random.random()
            if (randMut < (1 / sizeBit)):
                if Array_potomki[i, j] == 1:
                    Array_mutation[i, j] = 0
                else:
                    Array_mutation[i, j] = 1
            else:
                Array_mutation[i, j] = Array_potomki[i, j]
    return Array_mutation


def MutationStrong(sizePopulation, sizeBit, Array_potomki1, Array_mutation1, Matrix1):
    Array_mutation = Array_mutation1.copy()
    Matrix = Matrix1.copy()
    Array_potomki = Array_potomki1.copy()
    for i in range(sizePopulation):
        for j in range(sizeBit):
            randMut = random.random()
            if randMut < (3 / sizeBit):
                if Array_potomki[i, j] == 1:
                    Array_mutation[i, j] = 0
                else:
                    Array_mutation[i, j] = 1
            else:
                Array_mutation[i, j] = Array_potomki[i, j]
    return Array_mutation


if __name__ == '__main__':
    counterGlobalWhile = 0
    counterGoal = 0
    while counterGlobalWhile < 100:
        sizePopulation = 100
        sizeEpoch = 100
        selection_number = 1
        crossover_number = 1
        mutation_number = 1
        eE = 0.01
        aA = 0
        bB = 0
        goal = 0

        func_number = 13
        match func_number:
            case 1:
                aA = -16
                bB = 16
                goal = 0
            case 2:
                aA = -16
                bB = 16
                goal = 0
            case 3:
                aA = 0
                bB = 2
                goal = 1
            case 4:
                aA = -16
                bB = 16
                goal = 0
            case 5:
                aA = -10
                bB = 10
                goal = 1
            case 6:
                aA = -10
                bB = 10
                goal = 0
            case 7:
                aA = -2.5
                bB = 2.5
                goal = 0
            case 8:
                aA = -5
                bB = 5
                goal = 0
            case 9:
                aA = -4
                bB = 4
                goal = 0
            case 10:
                aA = 0
                bB = 4
                goal = 0
            case 11:
                aA = -4
                bB = 4
                goal = 1.99516
            case 12:
                aA = -4
                bB = 4
                goal = 1.99516
            case 13:
                aA = -65
                bB = 65
                goal = -32

        countStep = (bB - aA) / eE
        S = 1
        while 2 ** S < countStep:
            S += 1
        sizeBit = S * 2
        sizeBit = int(sizeBit)
        h = (bB - aA) / (2 ** S - 1)
        sizePopulation = int(sizePopulation)
        sizeBit = int(sizeBit)
        sizeEpoch = int(sizeEpoch)
        # Создание и заполнение основного массива
        MatrixArr = [[random.randint(0, 1) for j in range(sizeBit)] for i in range(sizePopulation)]
        Matrix = np.array(MatrixArr, dtype=int)

        bestResultFunction = 0.0
        bestArrayArr = [0] * sizeBit
        bestArray = np.array(bestArrayArr, dtype=int)
        bestX = 0.0
        bestY = 0.0

        for iEpoch in range(sizeEpoch):
            ArrayOdinCountArr = [0] * sizePopulation  # Массив ResultFunction
            XYArr = [[0 for j in range(2)] for i in range(sizePopulation)]
            XY = np.array(XYArr, dtype=float)

            for i in range(sizePopulation):  # Заполняем X, Y
                summa = 0
                for j in range(sizeBit // 2):
                    summa += int(2 ** j * Matrix[i, j])
                XY[i, 0] = aA + h * summa
                summa = 0  # -----
                j2 = 0
                for j in range(sizeBit // 2, sizeBit):
                    summa += int(2 ** j2 * Matrix[i, j])
                    j2 += 1
                XY[i, 1] = aA + h * summa

            for i1 in range(sizePopulation):  # Подсчёт X, Y
                match func_number:
                    case 1:
                        ArrayOdinCountArr[i1] += func_one(XY, i1)
                    case 2:
                        ArrayOdinCountArr[i1] += func_two(XY, i1)
                    case 3:
                        ArrayOdinCountArr[i1] += func_three(XY, i1)
                    case 4:
                        ArrayOdinCountArr[i1] += func_four(XY, i1)
                    case 5:
                        ArrayOdinCountArr[i1] += func_five(XY, i1)
                    case 6:
                        ArrayOdinCountArr[i1] += func_six(XY, i1)
                    case 7:
                        ArrayOdinCountArr[i1] += func_seven(XY, i1)
                    case 8:
                        ArrayOdinCountArr[i1] += func_eight(XY, i1)
                    case 9:
                        ArrayOdinCountArr[i1] += func_nine(XY, i1)
                    case 10:
                        ArrayOdinCountArr[i1] += func_ten(XY, i1)
                    case 11:
                        ArrayOdinCountArr[i1] += func_eleven(XY, i1)
                    case 12:
                        ArrayOdinCountArr[i1] += func_twelve(XY, i1)
                    case 13:
                        ArrayOdinCountArr[i1] += func_thirteen(XY, i1)

            ArrayOdinCount = np.array(ArrayOdinCountArr, dtype=float)
            Array_parentsArr = [[0 for j in range(sizeBit)] for i in range(sizePopulation * 2)]  # Селекция
            Array_parentsObj = np.array(Array_parentsArr, dtype=int)

            match selection_number:
                case 1:
                    Array_parents = Tournament(sizePopulation, sizeBit, ArrayOdinCount, Matrix, Array_parentsObj)
                case 2:
                    Array_parents = Proportional(sizePopulation, sizeBit, ArrayOdinCount, Matrix, Array_parentsObj)
                case 3:
                    Array_parents = Rank(sizePopulation, sizeBit, ArrayOdinCount, Matrix, Array_parentsObj)

            Array_potomkiArr = [[0 for j in range(sizeBit)] for i in range(sizePopulation)]  # Скрещивание
            Array_potomkiObj = np.array(Array_potomkiArr, dtype=int)

            match crossover_number:
                case 1:
                    Array_potomki = SinglePointCrossover(sizePopulation, sizeBit, Array_potomkiObj, Array_parents)
                case 2:
                    Array_potomki = TwoPointCrossover(sizePopulation, sizeBit, Array_potomkiObj, Array_parents)
                case 3:
                    Array_potomki = Uniform(sizePopulation, sizeBit, Array_potomkiObj, Array_parents)

            Array_mutationsArr = [[0 for j in range(sizeBit)] for i in range(sizePopulation)]  # Мутация
            Array_mutationsObj = np.array(Array_mutationsArr, dtype=int)
            match mutation_number:
                case 1:
                    Array_mutations = MutationWeak(sizePopulation, sizeBit, Array_potomki, Array_mutationsObj, Matrix)
                case 2:
                    Array_mutations = MutationAverage(sizePopulation, sizeBit, Array_potomki, Array_mutationsObj,
                                                      Matrix)
                case 3:
                    Array_mutations = MutationStrong(sizePopulation, sizeBit, Array_potomki, Array_mutationsObj, Matrix)

            ArrayLastArr = [[0 for j in range(sizeBit)] for i in
                            range(sizePopulation * 3)]  # Финальный массив из родителей и мутантов
            ArrayLast = np.array(ArrayLastArr, dtype=int)
            ArrayResultFuncArr = [0] * sizePopulation * 3
            ArrayResultFunc = np.array(ArrayResultFuncArr, dtype=float)
            for iLast in range(sizePopulation * 2):
                for j in range(sizeBit):
                    ArrayLast[iLast, j] = Array_parents[iLast, j]
            for iLast in range(sizePopulation):
                for j in range(sizeBit):
                    ArrayLast[sizePopulation * 2 + iLast, j] = Array_mutations[iLast, j]
            XY2Arr = [[0 for j in range(2)] for i in range(sizePopulation * 3)]
            XY2 = np.array(XY2Arr, dtype=float)
            for i in range(sizePopulation * 3):
                sum = 0
                for j in range(sizeBit // 2):
                    sum += int(2 ** j * ArrayLast[i, j])
                XY2[i, 0] = aA + h * sum
                sum = 0
                j1 = 0
                for j in range(sizeBit // 2, sizeBit):
                    sum += int(2 ** j1 * ArrayLast[i, j])
                    j1 += 1
                XY2[i, 1] = aA + h * sum
            ArrayIndivLastArr = [0] * (sizePopulation * 3)
            for iLast in range(sizePopulation * 3):
                match func_number:
                    case 1:
                        ArrayIndivLastArr[iLast] += func_one(XY2, iLast)
                    case 2:
                        ArrayIndivLastArr[iLast] += func_two(XY2, iLast)
                    case 3:
                        ArrayIndivLastArr[iLast] += func_three(XY2, iLast)
                    case 4:
                        ArrayIndivLastArr[iLast] += func_four(XY2, iLast)
                    case 5:
                        ArrayIndivLastArr[iLast] += func_five(XY2, iLast)
                    case 6:
                        ArrayIndivLastArr[iLast] += func_six(XY2, iLast)
                    case 7:
                        ArrayIndivLastArr[iLast] += func_seven(XY2, iLast)
                    case 8:
                        ArrayIndivLastArr[iLast] += func_eight(XY2, iLast)
                    case 9:
                        ArrayIndivLastArr[iLast] += func_nine(XY2, iLast)
                    case 10:
                        ArrayIndivLastArr[iLast] += func_ten(XY2, iLast)
                    case 11:
                        ArrayIndivLastArr[iLast] += func_eleven(XY2, iLast)
                    case 12:
                        ArrayIndivLastArr[iLast] += func_twelve(XY2, iLast)
                    case 13:
                        ArrayIndivLastArr[iLast] += func_thirteen(XY2, iLast)

            ArrayIndivLast = np.array(ArrayIndivLastArr, dtype=float)
            rememberBuffer = 0
            ArrayRememberArr = [0] * sizeBit
            ArrayRemember = np.array(ArrayRememberArr, dtype=int)
            for iLast in range(int(sizePopulation * sizePopulation / 10)):  # Убрал +1
                for j in range((sizePopulation * 3) - 1):
                    if ArrayIndivLast[j] < ArrayIndivLast[j + 1]:
                        rememberBuffer = ArrayIndivLast[j]
                        ArrayIndivLast[j] = ArrayIndivLast[j + 1]
                        ArrayIndivLast[j + 1] = rememberBuffer

                        rememberBuffer = XY2[j, 0]
                        XY2[j, 0] = XY2[j + 1, 0]
                        XY2[j + 1, 0] = rememberBuffer

                        rememberBuffer = XY2[j, 1]
                        XY2[j, 1] = XY2[j + 1, 1]
                        XY2[j + 1, 1] = rememberBuffer
                        for i1 in range(sizeBit):
                            ArrayRemember[i1] = ArrayLast[j, i1]
                            ArrayLast[j, i1] = ArrayLast[j + 1, i1]
                            ArrayLast[j + 1, i1] = ArrayRemember[i1]
            if iEpoch == 0:
                bestResultFunction = ArrayIndivLast[0]
                for i in range(sizeBit):
                    bestArray[i] = ArrayLast[0, i]
                bestX = XY2[0, 0]
                bestY = XY2[0, 1]
            if bestResultFunction < ArrayIndivLast[0]:
                for i in range(sizeBit):
                    bestArray[i] = ArrayLast[0, i]
                bestResultFunction = ArrayIndivLast[0]
                bestX = XY2[0, 0]
                bestY = XY2[0, 1]

            for i in range(sizePopulation):
                for j in range(sizeBit):
                    Matrix = ArrayLast.copy()

            if ((XY2[0, 0] >= goal - 0.01) and (XY2[0, 0] <= goal + 0.01) and (XY2[0, 1] >= goal - 0.01) and
                    (XY2[0, 1] <= goal + 0.01)):
                counterGoal += 1
                print("Попадание в результат № " + str(counterGoal))
                break

        counterGlobalWhile += 1
        print("Лучшее значение функции : " + str(-bestResultFunction))
        print("Лучший x = " + str(bestX) + " ; Лучший y = " + str(bestY))
    print(counterGoal)

