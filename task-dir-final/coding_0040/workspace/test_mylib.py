import unittest

import mylib as m


class T(unittest.TestCase):
    def test_factorial(self):
        self.assertEqual(m.factorial(5), 120)

    def test_clamp(self):
        self.assertEqual(m.clamp(5, 0, 10), 5)
        self.assertEqual(m.clamp(-3, 0, 10), 0)
        self.assertEqual(m.clamp(15, 0, 10), 10)

    def test_count_vowels(self):
        self.assertEqual(m.count_vowels('Apple'), 2)
        self.assertEqual(m.count_vowels('queue'), 4)

    def test_maximum(self):
        self.assertEqual(m.maximum([3, 9, 2]), 9)
        self.assertEqual(m.maximum([1, 5]), 5)

    def test_count_even(self):
        self.assertEqual(m.count_even([1, 2, 4, 6, 7]), 3)

    def test_minimum(self):
        self.assertEqual(m.minimum([4, 1, 7]), 1)

    def test_first_n(self):
        self.assertEqual(m.first_n([5, 6, 7, 8], 3), [5, 6, 7])


if __name__ == '__main__':
    unittest.main()
