import unittest

import mylib as m


class T(unittest.TestCase):
    def test_count_even(self):
        self.assertEqual(m.count_even([1, 2, 4, 6, 7]), 3)

    def test_clamp(self):
        self.assertEqual(m.clamp(5, 0, 10), 5)
        self.assertEqual(m.clamp(-3, 0, 10), 0)
        self.assertEqual(m.clamp(15, 0, 10), 10)

    def test_factorial(self):
        self.assertEqual(m.factorial(5), 120)

    def test_average(self):
        self.assertEqual(m.average([2, 4, 6]), 4)
        self.assertEqual(m.average([1, 2]), 1.5)

    def test_average_spec(self):
        self.assertEqual(m.average([2, 4, 6]), 5)


if __name__ == '__main__':
    unittest.main()
