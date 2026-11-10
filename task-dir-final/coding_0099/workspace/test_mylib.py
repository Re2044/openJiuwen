import unittest

import mylib as m


class T(unittest.TestCase):
    def test_unique_sorted(self):
        self.assertEqual(m.unique_sorted([3, 1, 3, 2]), [1, 2, 3])

    def test_count_even(self):
        self.assertEqual(m.count_even([1, 2, 4, 6, 7]), 3)

    def test_is_palindrome(self):
        self.assertTrue(m.is_palindrome('racecar'))
        self.assertFalse(m.is_palindrome('abca'))

    def test_count_vowels(self):
        self.assertEqual(m.count_vowels('Apple'), 2)
        self.assertEqual(m.count_vowels('queue'), 4)


if __name__ == '__main__':
    unittest.main()
