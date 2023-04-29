def chicken_rabbit(x, y):
    """
    x: 腿的总数
    y: 头的总数
    """
    b = (x - 2 * y) / 2
    a = y - b
    if b == int(b) and a == int(a) and b >= 0 <= a:
        return f'{int(a)}只鸡，{int(b)}只兔'
    else:
        return "无解"


if __name__ == '__main__':
    print(chicken_rabbit(20, 10))
    print(chicken_rabbit(28, 10))
    print(chicken_rabbit(10, 4))
