import random


def dfs_numIsland(grid, i, j):
    if i < 0 or i >= len(grid) or j < 0 or j >= len(grid[0]) or grid[i][j] == 0:
        return
    grid[i][j] = 0
    dfs_numIsland(grid, i - 1, j)
    dfs_numIsland(grid, i + 1, j)
    dfs_numIsland(grid, i, j - 1)
    dfs_numIsland(grid, i, j + 1)


def numIslands(grid):
    for i in grid:
        print(i)
    if len(grid) == 0:
        return 0
    count = 0
    for i in range(len(grid)):
        for j in range(len(grid[0])):
            if grid[i][j] == 1:
                dfs_numIsland(grid, i, j)
                count += 1
    return count


def setnum(row, col):
    return [[random.randint(0, 1) for j in range(col)] for i in range(row)]


if __name__ == '__main__':
    # 0代表海，1代表岛，如果两个1相邻，那么这两个1属于同一个岛。我们只考虑上下左右为相邻
    # 使用了深度优先搜索（DFS）来遍历和标记矩阵中的岛屿。
    print(numIslands(setnum(4, 4)))
