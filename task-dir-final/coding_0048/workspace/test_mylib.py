import unittest

import mylib as m


class T(unittest.TestCase):
    def test_factorial(self):
        self.assertEqual(m.factorial(5), 120)

    def test_first_n(self):
        self.assertEqual(m.first_n([5, 6, 7, 8], 3), [5, 6, 7])

    def test_minimum(self):
        self.assertEqual(m.minimum([4, 1, 7]), 1)

    def test_clamp(self):
        self.assertEqual(m.clamp(5, 0, 10), 5)
        self.assertEqual(m.clamp(-3, 0, 10), 0)
        self.assertEqual(m.clamp(15, 0, 10), 10)

    def test_average(self):
        self.assertEqual(m.average([2, 4, 6]), 4)
        self.assertEqual(m.average([1, 2]), 1.5)


if __name__ == '__main__':
    unittest.main()
