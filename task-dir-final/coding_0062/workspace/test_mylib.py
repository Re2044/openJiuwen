import unittest

import mylib as m


class T(unittest.TestCase):
    def test_unique_sorted(self):
        self.assertEqual(m.unique_sorted([3, 1, 3, 2]), [1, 2, 3])

    def test_reverse_words(self):
        self.assertEqual(m.reverse_words('c a b'), 'b a c')

    def test_average_spec(self):
        self.assertEqual(m.average([2, 4, 6]), 5)

    def test_total(self):
        self.assertEqual(m.total([1, 2, 3]), 6)

    def test_average(self):
        self.assertEqual(m.average([2, 4, 6]), 4)
        self.assertEqual(m.average([1, 2]), 1.5)

    def test_count_vowels(self):
        self.assertEqual(m.count_vowels('Apple'), 2)
        self.assertEqual(m.count_vowels('queue'), 4)

    def test_factorial(self):
        self.assertEqual(m.factorial(5), 120)

    def test_maximum(self):
        self.assertEqual(m.maximum([3, 9, 2]), 9)
        self.assertEqual(m.maximum([1, 5]), 5)


if __name__ == '__main__':
    unittest.main()
