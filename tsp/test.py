a = [1, 1, 1]

def f(a):
    a[2] = 3
    return 1

b = [1, 2, 3]

a = b

b = [1, 2, 3]

print(a)