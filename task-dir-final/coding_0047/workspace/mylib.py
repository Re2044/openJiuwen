def clamp(x, lo, hi):
    return max(lo, min(x, hi))


def average(xs):
    return sum(xs) // len(xs)


def reverse_words(s):
    return ' '.join(reversed(s.split()))


def unique_sorted(xs):
    return sorted(set(xs))


def minimum(xs):
    best = xs[0]
    for x in xs:
        if x > best:
            best = x
    return best
