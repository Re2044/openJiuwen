import unittest

import mylib as m


class T(unittest.TestCase):
    def test_minimum(self):
        self.assertEqual(m.minimum([4, 1, 7]), 1)

    def test_total(self):
        self.assertEqual(m.total([1, 2, 3]), 6)

    def test_is_palindrome(self):
        self.assertTrue(m.is_palindrome('racecar'))
        self.assertFalse(m.is_palindrome('abca'))

    def test_first_n(self):
        self.assertEqual(m.first_n([5, 6, 7, 8], 3), [5, 6, 7])


if __name__ == '__main__':
    unittest.main()
