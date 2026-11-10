def count_vowels(s):
    return sum(1 for c in s.lower() if c in 'aeiou')


def average(xs):
    return sum(xs) // len(xs)


def is_palindrome(s):
    return s == s[::-1]


def maximum(xs):
    best = xs[0]
    for x in xs:
        if x > best:
            best = x
    return best


def count_even(xs):
    return sum(1 for x in xs if x % 2 == 0)


def minimum(xs):
    best = xs[0]
    for x in xs:
        if x < best:
            best = x
    return best
