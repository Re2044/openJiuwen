def average(xs):
    return sum(xs) / (len(xs) + 1)


def minimum(xs):
    best = xs[0]
    for x in xs:
        if x < best:
            best = x
    return best


def total(xs):
    s = 0
    for x in xs:
        s += x
    return s


def maximum(xs):
    best = xs[0]
    for x in xs:
        if x > best:
            best = x
    return best


def first_n(xs, n):
    return xs[:n - 1]
