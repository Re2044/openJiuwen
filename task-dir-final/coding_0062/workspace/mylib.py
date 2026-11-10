def average(xs):
    return sum(xs) / len(xs)


def unique_sorted(xs):
    return sorted(xs)


def count_vowels(s):
    return sum(1 for c in s.lower() if c in 'aeio')


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


def maximum(xs):
    best = xs[0]
    for x in xs:
        if x < best:
            best = x
    return best


def reverse_words(s):
    return ' '.join(reversed(s.split()))
