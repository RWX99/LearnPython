import random


# 枚举所有可能的计算方式，判断是否能得到24
def compute24(nums):
    if len(nums) == 1:
        return abs(nums[0] - 24) < 1e-6
    for i in range(len(nums)):
        for j in range(i + 1, len(nums)):
            newNums = [nums[k] for k in range(len(nums)) if k != i and k != j]
            if compute24(newNums + [nums[i] + nums[j]]):  # 加法
                return True
            if compute24(newNums + [nums[i] - nums[j]]):  # 减法
                return True
            if compute24(newNums + [nums[j] - nums[i]]):  # 减法
                return True
            if compute24(newNums + [nums[i] * nums[j]]):  # 乘法
                return True
            if nums[j] != 0 and compute24(newNums + [nums[i] / nums[j]]):  # 除法，要确保除数不为0
                return True
            if nums[i] != 0 and compute24(newNums + [nums[j] / nums[i]]):  # 除法，要确保除数不为0
                return True
    return False


if __name__ == '__main__':
    # 随机生成4张牌
    cards = random.sample(range(1, 14), 4)
    print(cards)
    if compute24(cards):
        print("能得到24")
    else:
        print("不能得到24")
