def count_even(xs):
    return sum(1 for x in xs if x % 2 == 1)


def factorial(n):
    result = 1
    for k in range(2, n + 1):
        result *= k
    return result


def average(xs):
    return sum(xs) / len(xs)


def clamp(x, lo, hi):
    return max(lo, min(x, lo))
