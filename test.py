"""
Tests: to be expanded
"""
import logging
import os
import shutil
import sys
import unittest

import numpy as np
from sonnia import Sonia, SoNNia, SoniaPaired, SoNNiaPaired

SINGLE_CHAIN_MODELS = [
    'human_T_beta', 'human_T_alpha', 'human_B_heavy', 'human_B_kappa', 'human_B_lambda',
    'mouse_T_beta', 'mouse_T_alpha',
]
PAIRED_CHAIN_MODELS = [
    'human_B_heavy_kappa', 'human_B_heavy_lambda', 'human_T_beta_alpha'
]

class Test(unittest.TestCase):
    def test_save_load(self):
        seqs = np.array(
            [['CASSAF','TRBV10-1','TRBJ2-7'],
             ['CASSAF','TRBV10-1','TRBJ2-7'],
             ['CASSRF','TRBV10-1','TRBJ2-7'],
             ['CRSSRF','TRBV10-1','TRBJ2-7']]
        )
        qm = SoNNia(pgen_model='humanTRB',data_seqs=seqs)
        qm.add_generated_seqs(10)
        qm.save_model('test')
        qm1 = SoNNia(ppost_model='test')
        shutil.rmtree('test')
        self.assertTrue(qm.feature_dict == qm1.feature_dict)

    def test_load_default(self):
        for chain in SINGLE_CHAIN_MODELS:
            qm = Sonia(ppost_model=chain)
        for chain in PAIRED_CHAIN_MODELS:
            qm = SoniaPaired(ppost_model=chain)

    def test_infer(self):
        qm = Sonia(ppost_model='humanTRB')
        seqs = qm.generate_sequences_post(int(1e4))
        qm1 = SoNNia(pgen_model='humanTRB', data_seqs=seqs)
        qm1.add_generated_seqs(int(1e5))
        qm1.infer_selection(epochs=5)
        self.assertTrue(len(qm1.likelihood_test) == 5)

        qm2 = SoniaPaired(ppost_model='humanTCR')
        qm2.add_generated_seqs(int(1e4))
        seqs = qm2.generate_sequences_post(int(1e4))
        qm2.update_model(add_data_seqs=seqs)
        qm2.infer_selection(epochs=5)
        self.assertTrue(len(qm2.likelihood_test) == 5)

        qm3 = SoNNiaPaired(
            gen_seqs=qm2.gen_seqs, data_seqs=qm2.data_seqs,
            pgen_model_light=qm2.pgen_dir_light, pgen_model_heavy=qm2.pgen_dir_heavy
        )
        qm3.infer_selection(epochs=5)
        self.assertTrue(len(qm3.likelihood_test) == 5)

    def test_evaluate(self):
        qm = Sonia(ppost_model='humanTRB')
        pre_seqs = qm.generate_sequences_pre(int(1e3))
        q, pgen, ppost=qm.evaluate_seqs(pre_seqs)
        self.assertTrue(np.sum(q) > 0)

    def test_gene_encoding(self):
        logger = logging.getLogger('SoniaTests')

        # Default model parameters
        max_length = 30
        max_depth = 25
        gene_feature_start = max_length + max_depth * 20 * 2

        num_seqs = int(1e6)

        for model in SINGLE_CHAIN_MODELS:
            qm = Sonia(ppost_model=model)
            seqs = qm.generate_sequences_pre(num_seqs)
            seqs[:, 0] = ''
            counter = num_seqs
            for seq in seqs:
                gene_features = qm.find_seq_features(seq)
                if not gene_features:
                    logger.debug(
                        f'{model} has a gene feature encoding issue. '
                        f'{seq.tolist()} could not be encoded.'
                    )
                    counter -= 1
            self.assertTrue(counter == num_seqs)

        gene_feature_start *= 2
        for model in PAIRED_CHAIN_MODELS:
            qm = SoniaPaired(ppost_model=model)
            seqs = qm.generate_sequences_pre(num_seqs)
            seqs[:, 0] = ''
            seqs[:, 3] = ''
            counter = num_seqs
            for seq in seqs:
                gene_features = qm.find_seq_features(seq)
                if len(gene_features) != 2:
                    logger.debug(
                        f'{model} has a gene feature encoding issue. '
                        f'{seq.tolist()} could not be encoded.'
                    )
                    counter -= 1
            self.assertTrue(counter == num_seqs)

    def test_norms(self):
        for model in NORM_PRODUCTIVES:
            qm = Sonia(pgen_model=model)
            res = qm.pgen_model.compute_regex_CDR3_template_pgen('CX{0,}')
            self.assertTRUE(np.isclose(NORM_PRODUCTIVES[model], res))

if __name__ == '__main__':
    logging.basicConfig(stream=sys.stderr)
    logging.getLogger('SoniaTests').setLevel(logging.DEBUG)
    unittest.main()
