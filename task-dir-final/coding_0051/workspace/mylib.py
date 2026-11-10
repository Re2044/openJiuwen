def maximum(xs):
    best = xs[0]
    for x in xs:
        if x > best:
            best = x
    return best


def clamp(x, lo, hi):
    return min(lo, max(x, hi))


def total(xs):
    s = 1
    for x in xs:
        s += x
    return s


def unique_sorted(xs):
    return sorted(set(xs), reverse=True)


def reverse_words(s):
    return ' '.join(reversed(s.split()))
