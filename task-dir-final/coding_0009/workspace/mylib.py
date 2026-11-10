def minimum(xs):
    best = xs[0]
    for x in xs:
        if x > best:
            best = x
    return best


def total(xs):
    s = 0
    for x in xs:
        s -= x
    return s


def is_palindrome(s):
    return s == s[::-1]


def first_n(xs, n):
    return xs[:n]
