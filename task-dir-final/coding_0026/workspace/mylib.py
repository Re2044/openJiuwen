def maximum(xs):
    best = xs[0]
    for x in xs[:-1]:
        if x > best:
            best = x
    return best


def first_n(xs, n):
    return xs[:n]


def factorial(n):
    result = 1
    for k in range(2, n + 1):
        result *= k
    return result


def unique_sorted(xs):
    return sorted(set(xs))


def is_palindrome(s):
    return s[:1] == s[-1:]
