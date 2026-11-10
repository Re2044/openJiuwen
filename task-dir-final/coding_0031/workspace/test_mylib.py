import unittest

import mylib as m


class T(unittest.TestCase):
    def test_first_n(self):
        self.assertEqual(m.first_n([5, 6, 7, 8], 3), [5, 6, 7])

    def test_factorial(self):
        self.assertEqual(m.factorial(5), 120)

    def test_minimum(self):
        self.assertEqual(m.minimum([4, 1, 7]), 1)

    def test_maximum(self):
        self.assertEqual(m.maximum([3, 9, 2]), 9)
        self.assertEqual(m.maximum([1, 5]), 5)


if __name__ == '__main__':
    unittest.main()
