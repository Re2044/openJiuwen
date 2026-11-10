def factorial(n):
    result = 1
    for k in range(2, n + 1):
        result *= k
    return result


def minimum(xs):
    best = xs[0]
    for x in xs:
        if x < best:
            best = x
    return best


def first_n(xs, n):
    return xs[:n - 1]


def maximum(xs):
    best = xs[0]
    for x in xs:
        if x > best:
            best = x
    return best
