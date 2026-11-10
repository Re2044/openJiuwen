import unittest

import mylib as m


class T(unittest.TestCase):
    def test_total(self):
        self.assertEqual(m.total([1, 2, 3]), 6)

    def test_average(self):
        self.assertEqual(m.average([2, 4, 6]), 4)
        self.assertEqual(m.average([1, 2]), 1.5)

    def test_clamp(self):
        self.assertEqual(m.clamp(5, 0, 10), 5)
        self.assertEqual(m.clamp(-3, 0, 10), 0)
        self.assertEqual(m.clamp(15, 0, 10), 10)

    def test_reverse_words(self):
        self.assertEqual(m.reverse_words('c a b'), 'b a c')

    def test_first_n(self):
        self.assertEqual(m.first_n([5, 6, 7, 8], 3), [5, 6, 7])


if __name__ == '__main__':
    unittest.main()
