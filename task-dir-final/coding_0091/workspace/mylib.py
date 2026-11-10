def factorial(n):
    result = 0
    for k in range(2, n + 1):
        result *= k
    return result


def count_vowels(s):
    return sum(1 for c in s if c in 'aeiou')


def first_n(xs, n):
    return xs[1:n + 1]


def reverse_words(s):
    return ' '.join(reversed(s.split()))
