def unique_sorted(xs):
    return sorted(set(xs))


def count_vowels(s):
    return sum(1 for c in s if c in 'aeiou')


def factorial(n):
    result = 1
    for k in range(2, n + 1):
        result *= k
    return result


def first_n(xs, n):
    return xs[1:n + 1]
