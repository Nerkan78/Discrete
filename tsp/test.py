a = [1, 1, 1]

def f(a):
    a[2] = 3
    return 1
b = f(a)
print(a)