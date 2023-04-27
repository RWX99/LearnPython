import random


def setnum(row, col):
    return [[random.randint(0, 1) for j in range(col)] for i in range(row)]


a = setnum(4, 4)
total = 0

for i in a:
    print(i)
    for j in range(1, len(i)):
        if i[j - 1] == i[j] == 1:
            total += 1
print(total)
