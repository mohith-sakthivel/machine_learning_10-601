def Lucas(n):
    if n == 0:
        return 2
    elif n == 1:
        return 1
    else:
        return Lucas(n-1) + Lucas(n-2)

print (Lucas(32))