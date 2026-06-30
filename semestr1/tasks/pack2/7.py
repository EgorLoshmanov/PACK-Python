def fun_iter(q):
    s = 1
    while True:
        s = s * q
        yield s
        
fit = fun_iter(2)
for i in range(10):
    print(next(fit))