#!/usr/bin/env python
# -*- coding: utf-8 -*-

# @author: Giulio Isacchini
import logging
import multiprocessing as mp
import os
from typing import Any, Dict, Iterable, List, Optional, Tuple

logging.getLogger('tensorflow').disabled = True

import numpy as np
from tensorflow import keras
import tensorflow.keras.backend as K
from tqdm import tqdm

from sonnia.sonia import Sonia
from sonnia.utils import define_pgen_model, gene_to_num_str

class SoniaPaired(Sonia):
    def __init__(self,
                 *args: Tuple[Any],
                 load_dir: Optional[str] = None,
                 pgen_model_light: Optional[str] = None,
                 pgen_model_heavy: Optional[str] = None,
                 recompute_productive_norm: bool = False,
                 **kwargs: Dict[str, Any]
                ) -> None:
        if load_dir is None and (pgen_model_light is None or pgen_model_heavy is None):
            raise RuntimeError('Either load_dir must not be None or both pgen_model_light '
                               'and pgen_model_heavy must not be None.')
        if pgen_model_light is None:
            self.pgen_model_light = os.path.join(load_dir, 'light_chain')
        else:
            self.pgen_model_light = pgen_model_light
        if pgen_model_heavy is None:
            self.pgen_model_heavy = os.path.join(load_dir, 'heavy_chain')
        else:
            self.pgen_model_heavy = pgen_model_heavy

        if load_dir is None:
            self.recompute_productive_norm = True
        else:
            self.recompute_productive_norm = recompute_productive_norm
        self.load_pgen_models()

        Sonia.__init__(self, *args, load_dir=load_dir, **kwargs)

    def load_pgen_models(self
                        ) -> None:
        (self.genomic_data_light, self.generative_model_light,
         self.pgen_model_light, self.seqgen_model_light,
         self.norm_light, self.pgen_light_dir,
         model_str, chain_light) = define_pgen_model(self.pgen_model_light,
                                                     self.recompute_productive_norm,
                                                     True, True, True)
        if model_str != 'VJ':
            raise RuntimeError('A VDJ model was given to pgen_model_light. Please '
                               'rerun and point pgen_model_light to a VJ pgen model.')

        (self.genomic_data_heavy, self.generative_model_heavy,
         self.pgen_model_heavy, self.seqgen_model_heavy,
         self.norm_heavy, self.pgen_heavy_dir,
         model_str, chain_heavy) = define_pgen_model(self.pgen_model_heavy,
                                                     self.recompute_productive_norm,
                                                     True, True, True)
        if model_str != 'VDJ':
            raise RuntimeError('A VJ model was given to pgen_model_heavy. Please '
                               'rerun and point pgen_model_heavy to a VDJ pgen model.')

        valid_chain_pairs = [('IGL', 'IGH'), ('IGK', 'IGH'),
                             ('TRA', 'TRB'), ('TRG', 'TRD')]

        if (chain_light, chain_heavy) not in valid_chain_pairs:
            valid_chain_pairs_str = f'{valid_chain_pairs}'[1:-1]
            raise RuntimeError(f'A light-heavy chain pair of {(chain_light, chain_heavy)} does '
                               'constitute a valid receptor. Acceptable chain '
                               f'pairs: {valid_chain_pairs_str}.')

        if self.recompute_productive_norm:
            self.norm_productive = self.norm_heavy * self.norm_light

    def add_features(self,
                    ) -> None:
        """Generates a list of feature_lsts for L/R pos model.

        Parameters
        ----------
        max_depth : int
            Maximum index (from right or left) for amino acid positions
        max_L : int
            Maximum length CDR3 sequence
        include_genes : bool
            If true, features for gene selection are also generated. Currently
            joint V/J pairs used.
        """
        features = []

        if self.gene_features == 'vjl':
            for l in range(1, self.max_L + 1):
                features += [[v, j, f'l_l{l}']
                             for v in set(['v_l' + gene_to_num_str(genV[0],'V')[1:]
                                       for genV in self.genomic_data_light.genV])
                             for j in set(['j_l' + gene_to_num_str(genJ[0],'J')[1:]
                                       for genJ in self.genomic_data_light.genJ])]
                features += [[v, j, f'l_h{l}']
                             for v in set(['v_h' + gene_to_num_str(genV[0],'V')[1:]
                                       for genV in self.genomic_data_heavy.genV])
                             for j in set(['j_h' + gene_to_num_str(genJ[0],'J')[1:]
                                       for genJ in self.genomic_data_heavy.genJ])]
        else:
            for l in range(1, self.max_L + 1):
                features += [[f'l_l{l}'], [f'l_h{l}']]

        if self.include_aminoacids:
            for aa in self.amino_acids:
                for l in range(-self.max_depth, self.max_depth):
                    features += [[f'a_l{aa}{l}'], [f'a_h{aa}{l}']]

        if self.gene_features == 'joint_vj':
            features += [[v, j]
                         for v in set(['v_l' + gene_to_num_str(genV[0],'V')[1:]
                                       for genV in self.genomic_data_light.genV])
                         for j in set(['j_l' + gene_to_num_str(genJ[0],'J')[1:]
                                       for genJ in self.genomic_data_light.genJ])]
            features += [[v, j]
                         for v in set(['v_h'  + gene_to_num_str(genV[0],'V')[1:]
                                       for genV in self.genomic_data_heavy.genV])
                         for j in set(['j_h'  + gene_to_num_str(genJ[0],'J')[1:]
                                       for genJ in self.genomic_data_heavy.genJ])]
        if self.gene_features in {'indep_vj', 'v'}:
            features += [[v]
                         for v in set(['v_l' + gene_to_num_str(genV[0],'V')[1:]
                                       for genV in self.genomic_data_light.genV])]
            features += [[v]
                         for v in set(['v_h' + gene_to_num_str(genV[0],'V')[1:]
                                       for genV in self.genomic_data_heavy.genV])]
        if self.gene_features in {'indep_vj', 'j'}:
            features += [[j]
                         for j in set(['j_l'  + gene_to_num_str(genJ[0],'J')[1:]
                                       for genJ in self.genomic_data_light.genJ])]
            features += [[j]
                         for j in set(['j_h'  + gene_to_num_str(genJ[0],'J')[1:]
                                       for genJ in self.genomic_data_heavy.genJ])]

        self.update_model(add_features=features)

    def find_seq_features(self,
                          seq: Iterable[str],
                          features: Optional[Iterable[Tuple[str]]] = None
                         ) -> List[int]:
        """Finds which features match seq


        If no features are provided, the left/right indexing amino acid model
        features will be assumed.

        Parameters
        ----------
        seq : list
            CDR3 sequence and any associated genes
        features : ndarray
            Array of feature lists. Each list contains individual subfeatures which
            all must be satisfied.

        Returns
        -------
        seq_features : list
            Indices of features seq projects onto.

        """
        if features is None:
            feature_dict = self.feature_dict
        else:
            feature_dict = {tuple(feature): idx
                            for idx, feature in enumerate(features)}

        seq_features = set()

        # NOTE It's quicker to have the code written explicitly than perform
        # a for loop.

        # Heavy chain.
        cdr3_len = len(seq[0])
        cdr3_len_key = (f'l_h{cdr3_len}',)
        if cdr3_len_key in feature_dict:
            seq_features.add(feature_dict[cdr3_len_key])

        for idx, amino_acid in enumerate(list(seq[0])):
            fwd_key = (f'a_h{amino_acid}{idx}',)
            bkd_key = (f'a_h{amino_acid}{idx - cdr3_len}',)
            if fwd_key in feature_dict:
                seq_features.add(feature_dict[fwd_key])
            if bkd_key in feature_dict:
                seq_features.add(feature_dict[bkd_key])

        v_key = ('v_h' + gene_to_num_str(seq[1], 'V')[1:],)
        j_key = ('j_h' + gene_to_num_str(seq[2], 'J')[1:],)
        vj_key = v_key + j_key
        vjl_key = vj_key + cdr3_len_key
        if v_key in feature_dict:
            seq_features.add(feature_dict[v_key])
        if j_key in feature_dict:
            seq_features.add(feature_dict[j_key])
        if vj_key in feature_dict:
            seq_features.add(feature_dict[vj_key])
        if vjl_key in feature_dict:
            seq_features.add(feature_dict[vjl_key])

        # Light chain.
        cdr3_len = len(seq[3])
        cdr3_len_key = (f'l_l{cdr3_len}',)
        if cdr3_len_key in feature_dict:
            seq_features.add(feature_dict[cdr3_len_key])

        for idx, amino_acid in enumerate(list(seq[3])):
            fwd_key = (f'a_l{amino_acid}{idx}',)
            bkd_key = (f'a_l{amino_acid}{idx - cdr3_len}',)
            if fwd_key in feature_dict:
                seq_features.add(feature_dict[fwd_key])
            if bkd_key in feature_dict:
                seq_features.add(feature_dict[bkd_key])

        v_key = ('v_l' + gene_to_num_str(seq[4], 'V')[1:],)
        j_key = ('j_l' + gene_to_num_str(seq[5], 'J')[1:],)
        vj_key = v_key + j_key
        vjl_key = vj_key + cdr3_len_key
        if v_key in feature_dict:
            seq_features.add(feature_dict[v_key])
        if j_key in feature_dict:
            seq_features.add(feature_dict[j_key])
        if vj_key in feature_dict:
            seq_features.add(feature_dict[vj_key])
        if vjl_key in feature_dict:
            seq_features.add(feature_dict[vjl_key])

        return list(seq_features)

    def generate_sequences_pre(self,
                               num_seqs: int = 1,
                               nucleotide: bool = False,
                               custom_error: Optional[int] = None,
                               add_error: bool = False
                              ) -> np.ndarray:
        """Generates MonteCarlo sequences for gen_seqs using OLGA.
        Only generates seqs from a V(D)J model. Requires the OLGA package
        (pip install olga).
        Parameters
        ----------
        num_seqs : int or float
            Number of MonteCarlo sequences to generate and add to the specified
            sequence pool.
        custom_model_folder : str
            Path to a folder specifying a custom IGoR formatted model to be
            used as a generative model. Folder must contain 'model_params.txt'
            and 'model_marginals.txt'
        Returns
        --------------
        seqs : list
            MonteCarlo sequences drawn from a VDJ recomb model
        """
        #Generate sequences
        seqs = []
        for i in tqdm(range(int(num_seqs))):
            seq_light = self.seqgen_model_light.gen_rnd_prod_CDR3(conserved_J_residues='ABCEDFGHIJKLMNOPQRSTUVWXYZ')
            seq_heavy = self.seqgen_model_heavy.gen_rnd_prod_CDR3(conserved_J_residues='ABCEDFGHIJKLMNOPQRSTUVWXYZ')

            seq = [seq_heavy[1],
                   self.genomic_data_heavy.genV[seq_heavy[2]][0].split('*')[0],
                   self.genomic_data_heavy.genJ[seq_heavy[3]][0].split('*')[0],
                   seq_light[1],
                   self.genomic_data_light.genV[seq_light[2]][0].split('*')[0],
                   self.genomic_data_light.genJ[seq_light[3]][0].split('*')[0],
                   seq_heavy[0], seq_light[0]]
            seqs.append(seq)

        if nucleotide: return np.array(seqs)
        else: return np.array(seqs)[:,:-2]

    def compute_all_pgens(self,
                          seqs: Iterable[Iterable[str]],
                          include_genes: bool = True
                         ) -> np.ndarray:
        '''
        Compute Pgen of sequences using OLGA
        '''
        #Load OLGA for seq pgen estimation

        # every process needs to access this vector, for sure there is a smarter way to implement this.
        final_models_heavy = [self.pgen_model_heavy for i in range(len(seqs))]
        final_models_light = [self.pgen_model_light for i in range(len(seqs))]
        if include_genes:
            pool = mp.Pool(processes=self.processes)
            f1 = pool.map(compute_pgen_expand_heavy, zip(seqs,final_models_heavy))
            pool.close()
            pool = mp.Pool(processes=self.processes)
            f2 = pool.map(compute_pgen_expand_light, zip(seqs,final_models_light))
            pool.close()
            return np.array(f1) * np.array(f2)
        else:
            pool = mp.Pool(processes=self.processes)
            f1 = pool.map(compute_pgen_expand_novj_heavy, zip(seqs,final_models_light))
            pool.close()
            pool = mp.Pool(processes=self.processes)
            f2 = pool.map(compute_pgen_expand_novj_heavy, zip(seqs,final_models_light))
            pool.close()
            return np.array(f1) * np.array(f2)

    def set_gauge(self
                 ) -> None:
        '''
        placeholder for gauges.
        '''
        pass

    def _save_pgen_model(self,
                         save_dir: str
                        ) -> None:
        import shutil
        zipped = zip([self.pgen_light_dir, self.pgen_heavy_dir],
                     ['light_chain', 'heavy_chain'])
        for pgen_dir, folder_name in zipped:
            chain_dir = os.path.join(save_dir, folder_name)
            if not os.path.isdir(chain_dir): os.mkdir(chain_dir)
            shutil.copy2(os.path.join(pgen_dir, 'model_params.txt'), chain_dir)
            shutil.copy2(os.path.join(pgen_dir, 'model_marginals.txt'), chain_dir)
            shutil.copy2(os.path.join(pgen_dir, 'V_gene_CDR3_anchors.csv'), chain_dir)
            shutil.copy2(os.path.join(pgen_dir, 'J_gene_CDR3_anchors.csv'), chain_dir)

def compute_pgen_expand_light(x
                             ) -> float:
    return x[1].compute_aa_CDR3_pgen(x[0][3],x[0][4],x[0][5])

def compute_pgen_expand_heavy(x
                             ) -> float:
    return x[1].compute_aa_CDR3_pgen(x[0][0],x[0][1],x[0][2])

def compute_pgen_expand_novj_light(x
                                  ) -> float:
    return x[1].compute_aa_CDR3_pgen(x[0][3])

def compute_pgen_expand_novj_heavy(x
                                  ) -> float:
    return x[1].compute_aa_CDR3_pgen(x[0][0])
