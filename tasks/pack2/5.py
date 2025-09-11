def fib(n):
    if n == 0 or n == 1:
        return 1
    else:
        return fib(n - 2) + fib(n - 1)
    
print(fib(0), fib(1), fib(2), fib(3), fib(4), fib(5), fib(6), fib(7))