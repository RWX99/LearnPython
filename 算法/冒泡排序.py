def cocktail_sort(arr):
    """
    鸡尾酒排序法
    我们维护两个指针 left 和 right，它们分别指向数组的首尾元素
    我们先从左向右遍历数组，将较大的元素交换到右侧。
    然后从右向左遍历数组，将较小的元素交换到左侧。
    """
    n = len(arr)
    left, right = 0, n - 1
    while left < right:
        for i in range(left, right):
            if arr[i] > arr[i + 1]:
                arr[i], arr[i + 1] = arr[i + 1], arr[i]
        right -= 1
        for i in range(right, left, -1):
            if arr[i - 1] > arr[i]:
                arr[i], arr[i - 1] = arr[i - 1], arr[i]
        left += 1
    return arr


def bubble_sort(arr):
    """
    冒泡排序法
    我们可以记录每一轮比较中最后一次交换的位置，下一轮比较时，只需要比较到该位置即可。这样可以减少比较次数，提高排序效率
    """
    n = len(arr)
    last_swap = n - 1
    while last_swap > 0:
        k = last_swap
        last_swap = 0
        for i in range(k):
            if arr[i] > arr[i + 1]:
                arr[i], arr[i + 1] = arr[i + 1], arr[i]
                last_swap = i
    return arr


if __name__ == '__main__':
    print(cocktail_sort([1, 5, 9, 7, 3, 4, 6, 8, 2]))
    print(bubble_sort([1, 5, 9, 7, 3, 4, 6, 8, 2]))
