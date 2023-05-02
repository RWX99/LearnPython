def most_frequent_letter(s):
    frequency = {}
    for letter in s:
        if letter in frequency:
            frequency[letter] += 1
        else:
            frequency[letter] = 1
    max_frequency = max(frequency.values())
    most_frequent = {key: value for key, value in frequency.items() if value == max_frequency}
    return most_frequent


if __name__ == '__main__':
    # 统计字符串中出现次数最多的字符
    print(most_frequent_letter('abcbdhhklbkassadsad'))
