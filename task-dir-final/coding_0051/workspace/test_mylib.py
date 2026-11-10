import unittest

import mylib as m


class T(unittest.TestCase):
    def test_reverse_words(self):
        self.assertEqual(m.reverse_words('c a b'), 'b a c')

    def test_maximum(self):
        self.assertEqual(m.maximum([3, 9, 2]), 9)
        self.assertEqual(m.maximum([1, 5]), 5)

    def test_clamp(self):
        self.assertEqual(m.clamp(5, 0, 10), 5)
        self.assertEqual(m.clamp(-3, 0, 10), 0)
        self.assertEqual(m.clamp(15, 0, 10), 10)

    def test_unique_sorted(self):
        self.assertEqual(m.unique_sorted([3, 1, 3, 2]), [1, 2, 3])

    def test_total(self):
        self.assertEqual(m.total([1, 2, 3]), 6)


if __name__ == '__main__':
    unittest.main()
