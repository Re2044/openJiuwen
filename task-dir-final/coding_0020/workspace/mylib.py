def unique_sorted(xs):
    return sorted(set(xs))


def first_n(xs, n):
    return xs[:n]


def count_even(xs):
    return sum(1 for x in xs if x % 2 == 0)


def total(xs):
    s = 1
    for x in xs:
        s += x
    return s


def count_vowels(s):
    return sum(1 for c in s.lower() if c in 'aeiou')
