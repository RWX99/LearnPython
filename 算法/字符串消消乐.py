def string_removal(s):
    stack = []
    for char in s:
        if stack and stack[-1] == char:
            stack.pop()
        else:
            stack.append(char)
    return "".join(stack)


# 规则：相邻两个字符相同就可以消掉，在消除后，剩余的字符重新排列组成新的字符串，
# 再根据规则执行消除，直到所有相邻的字符都不相同或变成空串为止，输出最后获得的字符串
# 在上述代码中，我们使用一个栈来维护未消除的字符。遍历字符串  s ，
# 如果当前字符与栈顶字符相同，就将栈顶字符弹出；否则，将当前字符压入栈中。
# 最后，我们将栈中的字符合并成一个字符串并返回。
print(string_removal('abbaccd'))
