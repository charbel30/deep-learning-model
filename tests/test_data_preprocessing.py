import unittest
import sys
import pandas as pd
sys.path.append('../')
from utils.Preprocessing_utils import output_selection_prepro

class TestDataPreprocessing(unittest.TestCase):
    def test_output_selection_prepro(self):
        # Load your data
        df = pd.read_csv('../data/raw/dukecathr.csv')
        
        targets = ['CHFSEV', 'ACS', 'NUMDZV', 'DEATH', 'LADST', 'LCXST', 'LMST', 'PRXLADST', 'RCAST']
        
        for target in targets:
            # Call the output_selection_prepro function
            X, y, num_cols, cat_cols = output_selection_prepro(df, target)
            
            # Write assertions to check the output
            self.assertIsNotNone(X)
            self.assertIsNotNone(y)
            self.assertIsNotNone(num_cols)
            self.assertIsNotNone(cat_cols)

if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)