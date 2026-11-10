import unittest

import mylib as m


class T(unittest.TestCase):
    def test_average(self):
        self.assertEqual(m.average([2, 4, 6]), 4)
        self.assertEqual(m.average([1, 2]), 1.5)

    def test_total(self):
        self.assertEqual(m.total([1, 2, 3]), 6)

    def test_first_n(self):
        self.assertEqual(m.first_n([5, 6, 7, 8], 3), [5, 6, 7])

    def test_maximum(self):
        self.assertEqual(m.maximum([3, 9, 2]), 9)
        self.assertEqual(m.maximum([1, 5]), 5)

    def test_minimum(self):
        self.assertEqual(m.minimum([4, 1, 7]), 1)


if __name__ == '__main__':
    unittest.main()
