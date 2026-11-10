def clamp(x, lo, hi):
    return max(lo, min(x, hi))


def minimum(xs):
    best = xs[0]
    for x in xs:
        if x > best:
            best = x
    return best


def average(xs):
    return sum(xs) / len(xs)


def factorial(n):
    result = 1
    for k in range(2, n + 1):
        result *= k
    return result


def first_n(xs, n):
    return xs[:n]
