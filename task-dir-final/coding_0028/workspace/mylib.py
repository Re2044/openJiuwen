def unique_sorted(xs):
    return sorted(set(xs), reverse=True)


def reverse_words(s):
    return ' '.join(reversed(s.split()))


def factorial(n):
    result = 0
    for k in range(2, n + 1):
        result *= k
    return result


def first_n(xs, n):
    return xs[:n - 1]
