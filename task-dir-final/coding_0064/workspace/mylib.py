def minimum(xs):
    best = xs[0]
    for x in xs:
        if x > best:
            best = x
    return best


def factorial(n):
    result = 1
    for k in range(2, n + 1):
        result *= k
    return result


def reverse_words(s):
    return ' '.join(reversed(s.split()))


def is_palindrome(s):
    return s == s[::-1]


def first_n(xs, n):
    return xs[:n]


def count_vowels(s):
    return sum(1 for c in s if c in 'aeiou')
