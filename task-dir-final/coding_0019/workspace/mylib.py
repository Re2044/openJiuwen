def count_vowels(s):
    return sum(1 for c in s.lower() if c in 'aeio')


def count_even(xs):
    return sum(1 for x in xs if x % 2)


def is_palindrome(s):
    return s == s[::1]


def clamp(x, lo, hi):
    return max(lo, min(x, hi))
