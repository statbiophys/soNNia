"""
Tests: to be expanded
"""

import pandas as pd
import numpy as np
from sonnia.sonnia import SoNNia
from sonnia.sonia import Sonia
import os
import unittest
import shutil

class Test(unittest.TestCase):

    def test_save_load(self):
        seqs=[['CASSAF','TRBV10-1','TRBJ2-7'],
             ['CASSAF','TRBV10-1','TRBJ2-7'],
             ['CASSRF','TRBV10-1','TRBJ2-7'],
             ['CRSSRF','TRBV10-1','TRBJ2-7']]
        df=pd.DataFrame(seqs,columns=['amino_acid','v_gene','j_gene'])
        df['read_count']=10
        qm=SoNNia(data_seqs=seqs)
        qm.add_generated_seqs(10)
        qm.save_model('test')
        qm1=SoNNia(load_dir='test')
        shutil.rmtree('test')
        self.assertTrue(qm.feature_dict==qm1.feature_dict)
        
    def test_load_default(self):
        chains = [ 'human_T_alpha', 'human_T_beta', 'human_B_heavy','human_B_kappa','human_B_lambda','mouse_T_beta','mouse_T_alpha']
        for chain in chains:
            qm=Sonia()
            qm.load_default_model(chain_type=chain)
            
    def test_infer(self):
        qm=Sonia()
        qm.load_default_model(chain_type='human_T_beta')
        seqs=qm.generate_sequences_post(int(1e4))
        qm1=SoNNia(data_seqs=seqs)
        qm1.add_generated_seqs(int(1e5))
        qm1.infer_selection(epochs=5)
        self.assertTrue(len(qm1.likelihood_test)==5)
        
    def test_evaluate(self):
        qm=Sonia()
        qm.load_default_model()
        pre_seqs=qm.generate_sequences_pre(int(1e3))
        q,pgen,ppost=qm.evaluate_seqs(pre_seqs)
        self.assertTrue(np.sum(q)>0)

        


if __name__ == '__main__':
    unittest.main()