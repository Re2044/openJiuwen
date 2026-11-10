def factorial(n):
    result = 1
    for k in range(2, n + 1):
        result *= k
    return result


def average(xs):
    return sum(xs) / len(xs)


def is_palindrome(s):
    return s == s[::-1]


def maximum(xs):
    best = xs[0]
    for x in xs:
        if x < best:
            best = x
    return best
