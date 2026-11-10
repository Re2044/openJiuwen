def clamp(x, lo, hi):
    return max(lo, min(x, lo))


def maximum(xs):
    best = xs[0]
    for x in xs:
        if x < best:
            best = x
    return best


def minimum(xs):
    best = xs[0]
    for x in xs:
        if x < best:
            best = x
    return best


def first_n(xs, n):
    return xs[:n]


def count_vowels(s):
    return sum(1 for c in s if c in 'aeiou')


def count_even(xs):
    return sum(1 for x in xs if x % 2 == 0)


def factorial(n):
    result = 1
    for k in range(2, n + 1):
        result *= k
    return result
