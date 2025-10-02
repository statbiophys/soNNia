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
from sonnia.utils import (
    NORM_PRODUCTIVES,
    run_terminal,
    get_model_dir,
    define_pgen_model,
    filter_seqs,
    sample_olga,
    add_random_error,
    gene_to_num_str,
    generate_sequence,
    generate_paired_sequence,
)
SINGLE_CHAIN_MODELS = [
    'human_T_beta', 'human_T_alpha', 'human_B_heavy', 'human_B_kappa', 'human_B_lambda',
    'mouse_T_beta', 'mouse_T_alpha',
]
PAIRED_CHAIN_MODELS = [
    'human_B_heavy_kappa', 'human_B_heavy_lambda', 'human_T_beta_alpha'
]

class Test(unittest.TestCase):

    def setUp(self):
        self.seqs = np.array(
            [['CASSAF', 'TRBV10-1', 'TRBJ2-7'],
             ['CASSAF', 'TRBV10-1', 'TRBJ2-7'],
             ['CASSRF', 'TRBV10-1', 'TRBJ2-7'],
             ['CRSSRF', 'TRBV10-1', 'TRBJ2-7']]
        )

        self.seqs_paired = np.array([
            ['CASSLLWRSEQYF', 'TRBV7-3', 'TRBJ2-7', 'CIVRVGAAGNKLTF','TRAV26-1', 'TRAJ17'],
            ['CSARAYLSMNTEAFF', 'TRBV20-1', 'TRBJ1-1', 'CAPKLSF', 'TRAV12-3','TRAJ20'],
            ['CAPPGSNQPQHF', 'TRBV6-4', 'TRBJ1-5', 'CATGGSNSNSGYALNF', 'TRAV17', 'TRAJ41'],
            ['CASSKRPDQPQHF', 'TRBV9', 'TRBJ1-5', 'CAENKGQGNLIF', 'TRAV13-2', 'TRAJ42']])

    def test_save_load(self):
        for model in [Sonia, SoNNia]:
            qm = model(pgen_model='humanTRB',data_seqs=self.seqs)
            qm.add_generated_seqs(10)
            qm.save_model('test')
            qm1 = model(ppost_model='test')
            shutil.rmtree('test')
            self.assertTrue(qm.feature_dict == qm1.feature_dict)

        for model in [SoniaPaired, SoNNiaPaired]:
            qm = model(pgen_model='humanTCR',data_seqs=self.seqs_paired)
            qm.add_generated_seqs(10)
            qm.save_model('test')
            qm1 = model(ppost_model='test')
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

        qm3 = SoNNiaPaired(gen_seqs=qm2.gen_seqs, data_seqs=qm2.data_seqs,pgen_model='humanTCR')
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
            #seqs[:, 0] = ''
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
            self.assertTrue(np.isclose(NORM_PRODUCTIVES[model], res))


    def test_run_terminal(self):
        result = run_terminal("echo 'Hello, World!'")
        self.assertEqual(result[0][0], "Hello, World!")

    def test_get_model_dir(self):
        model_dir = get_model_dir("human_T_beta")
        self.assertTrue(os.path.isdir(model_dir))

    def test_define_pgen_model(self):
        result = define_pgen_model("human_T_beta")
        self.assertEqual(len(result), 5)

    def test_filter_seqs(self):
        seqs = [
            ["CASSAF", "TRBV10-1", "TRBJ2-7"],
            ["CASSAF", "TRBV10-1", "TRBJ2-7"],
            ["CASSRF", "TRBV10-1", "TRBJ2-7"],
            ["CRSSRF", "TRBV10-1", "TRBJ2-7"]
        ]
        model = "human_T_beta"
        filtered_seqs = filter_seqs(seqs, model)
        self.assertEqual(len(filtered_seqs), 4)

    def test_sample_olga(self):
        sequences = sample_olga(num_gen_seqs=2, custom_model_folder = "human_T_beta")
        self.assertEqual(len(sequences), 2)

    def test_add_random_error(self):
        seq = "ATGC"
        error_seq = add_random_error(seq, 0.5)
        self.assertEqual(len(seq), len(error_seq))

    def test_gene_to_num_str(self):
        gene_name = "TRBV10-1*01"
        gene_type = "V"
        result = gene_to_num_str(gene_name, gene_type)
        self.assertEqual(result, "v10-1")

    def test_generate_sequence(self):
        genomic_data, generative_model, pgen_model, seqgen_model, _ = define_pgen_model("human_T_beta")
        seq = generate_sequence(seqgen_model, genomic_data)
        self.assertEqual(len(seq), 4)

    def test_generate_paired_sequence(self):
        genomic_data_light, generative_model_light, pgen_model_light, seqgen_model_light, _ = define_pgen_model("human_B_kappa")
        genomic_data_heavy, generative_model_heavy, pgen_model_heavy, seqgen_model_heavy, _ = define_pgen_model("human_B_heavy")
        seq = generate_paired_sequence(seqgen_model_light, seqgen_model_heavy, genomic_data_light, genomic_data_heavy)
        self.assertEqual(len(seq), 8)

    def test_evaluate_cli(self):
        result = run_terminal("sonnia evaluate -i examples/data_seqs.csv.gz --model human_T_beta -m 10")
        self.assertTrue(not "\x1b" in result[1][0])
    
    def test_infer_cli(self):
        result = run_terminal("sonnia infer -i examples/data_seqs.csv.gz --model human_T_beta -m 1000")
        self.assertTrue(not "\x1b" in result[1][0])
    
    def test_generate_cli(self):
        result = run_terminal("sonnia generate --model human_T_beta -n 10 --pre")
        self.assertTrue(not "\x1b" in result[1][0])

if __name__ == "__main__":
    unittest.main()
