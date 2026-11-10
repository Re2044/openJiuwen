import unittest

import mylib as m


class T(unittest.TestCase):
    def test_average_spec(self):
        self.assertEqual(m.average([2, 4, 6]), 5)

    def test_total(self):
        self.assertEqual(m.total([1, 2, 3]), 6)

    def test_reverse_words(self):
        self.assertEqual(m.reverse_words('c a b'), 'b a c')

    def test_is_palindrome(self):
        self.assertTrue(m.is_palindrome('racecar'))
        self.assertFalse(m.is_palindrome('abca'))

    def test_average(self):
        self.assertEqual(m.average([2, 4, 6]), 4)
        self.assertEqual(m.average([1, 2]), 1.5)

    def test_first_n(self):
        self.assertEqual(m.first_n([5, 6, 7, 8], 3), [5, 6, 7])


if __name__ == '__main__':
    unittest.main()
