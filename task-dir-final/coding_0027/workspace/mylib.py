def count_vowels(s):
    return sum(1 for c in s.lower() if c in 'aeiou')


def count_even(xs):
    return sum(1 for x in xs if x % 2 == 1)


def is_palindrome(s):
    return s == s[::-1]


def clamp(x, lo, hi):
    return max(lo, min(x, hi))


def unique_sorted(xs):
    return sorted(set(xs), reverse=True)


def average(xs):
    return sum(xs) / len(xs)
