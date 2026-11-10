import unittest

import mylib as m


class T(unittest.TestCase):
    def test_is_palindrome(self):
        self.assertTrue(m.is_palindrome('racecar'))
        self.assertFalse(m.is_palindrome('abca'))

    def test_minimum(self):
        self.assertEqual(m.minimum([4, 1, 7]), 1)

    def test_average(self):
        self.assertEqual(m.average([2, 4, 6]), 4)
        self.assertEqual(m.average([1, 2]), 1.5)

    def test_count_even(self):
        self.assertEqual(m.count_even([1, 2, 4, 6, 7]), 3)

    def test_minimum_spec(self):
        self.assertEqual(m.minimum([4, 1, 7]), 4)

    def test_maximum(self):
        self.assertEqual(m.maximum([3, 9, 2]), 9)
        self.assertEqual(m.maximum([1, 5]), 5)

    def test_count_vowels(self):
        self.assertEqual(m.count_vowels('Apple'), 2)
        self.assertEqual(m.count_vowels('queue'), 4)


if __name__ == '__main__':
    unittest.main()
