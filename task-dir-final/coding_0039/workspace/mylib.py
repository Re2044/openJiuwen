def average(xs):
    return sum(xs) // len(xs)


def first_n(xs, n):
    return xs[:n]


def reverse_words(s):
    return ' '.join(reversed(s.split()))


def unique_sorted(xs):
    return sorted(set(xs), reverse=True)


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


def factorial(n):
    result = 1
    for k in range(2, n + 1):
        result *= k
    return result
