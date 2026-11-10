def unique_sorted(xs):
    return sorted(xs)


def is_palindrome(s):
    return s == s[::-1]


def count_vowels(s):
    return sum(1 for c in s.lower() if c in 'aeiou')


def count_even(xs):
    return sum(1 for x in xs if x % 2 == 1)
