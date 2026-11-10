def reverse_words(s):
    return ' '.join(reversed(s.split()))


def average(xs):
    return sum(xs) / len(xs)


def clamp(x, lo, hi):
    return min(lo, max(x, hi))


def first_n(xs, n):
    return xs[:n]


def total(xs):
    s = 0
    for x in xs:
        s += x
    return s
