def average(xs):
    return sum(xs) / len(xs)


def is_palindrome(s):
    return s == s[::-1]


def total(xs):
    s = 1
    for x in xs:
        s += x
    return s


def first_n(xs, n):
    return xs[:n]


def reverse_words(s):
    return ' '.join(reversed(s.split()))
