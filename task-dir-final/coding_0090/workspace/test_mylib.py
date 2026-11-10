import unittest

import mylib as m


class T(unittest.TestCase):
    def test_factorial(self):
        self.assertEqual(m.factorial(5), 120)

    def test_first_n(self):
        self.assertEqual(m.first_n([5, 6, 7, 8], 3), [5, 6, 7])

    def test_count_vowels(self):
        self.assertEqual(m.count_vowels('Apple'), 2)
        self.assertEqual(m.count_vowels('queue'), 4)

    def test_unique_sorted(self):
        self.assertEqual(m.unique_sorted([3, 1, 3, 2]), [1, 2, 3])


if __name__ == '__main__':
    unittest.main()
