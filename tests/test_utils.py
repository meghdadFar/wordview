import unittest

from wordview.mwes.mwe_utils import get_ngrams, get_counts


class TestUtils(unittest.TestCase):
    
    def test_get_ngrams_type(self):
        self.assertRaises(TypeError, get_ngrams(sentence=5, n=2))
        self.assertRaises(TypeError, get_ngrams(sentence=[1, 2, 3], n=2))
        self.assertRaises(TypeError, get_ngrams(sentence=[1, 2, 'test'], n=2))

    def test_get_ngrams_val(self):
        self.assertEqual(["Cat sat", "sat on", "on mat"], get_ngrams(sentence="Cat sat on mat", n=2))
        self.assertEqual(["happy cat"], get_ngrams(sentence="happy cat", n=2))
        self.assertEqual([], get_ngrams(sentence="cat", n=2))
        
if __name__ == '__main__':
    unittest.main()