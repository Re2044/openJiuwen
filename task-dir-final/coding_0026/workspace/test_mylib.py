import unittest

import mylib as m


class T(unittest.TestCase):
    def test_maximum(self):
        self.assertEqual(m.maximum([3, 9, 2]), 9)
        self.assertEqual(m.maximum([1, 5]), 5)

    def test_unique_sorted(self):
        self.assertEqual(m.unique_sorted([3, 1, 3, 2]), [1, 2, 3])

    def test_factorial(self):
        self.assertEqual(m.factorial(5), 120)

    def test_first_n(self):
        self.assertEqual(m.first_n([5, 6, 7, 8], 3), [5, 6, 7])

    def test_is_palindrome(self):
        self.assertTrue(m.is_palindrome('racecar'))
        self.assertFalse(m.is_palindrome('abca'))


if __name__ == '__main__':
    unittest.main()
