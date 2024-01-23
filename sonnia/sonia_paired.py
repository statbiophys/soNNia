#!/usr/bin/env python
# -*- coding: utf-8 -*-

# @author: Giulio Isacchini
import multiprocessing as mp
import os
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
from tensorflow import keras
import tensorflow.keras.backend as K
from tqdm import tqdm

from sonnia.sonia import Sonia
from sonnia.utils import define_pgen_model, gene_to_num_str, PRODUCTIVE_NORMS

FILEDIR = os.path.dirname(os.path.abspath(__file__))

class SoniaPaired(Sonia):
    def __init__(self,
                 *args: Tuple[Any],
                 chain_type_heavy: str = 'human_T_beta',
                 chain_type_light: str = 'human_T_alpha',
                 custom_pgen_model_light: Optional[str] = None,
                 custom_pgen_model_heavy: Optional[str] = None,
                 **kwargs: Dict[str, Any]
                ) -> None:
        self.chain_type_heavy = chain_type_heavy
        self.chain_type_light = chain_type_light
        self.custom_pgen_model_light = custom_pgen_model_light
        self.custom_pgen_model_heavy = custom_pgen_model_heavy

        Sonia.__init__(self, *args, **kwargs)
        recompute_norm = not (custom_pgen_model_light is None and custom_pgen_model_heavy is None)
        self.define_models(recompute_norm=recompute_norm)

    def define_models(self,
                      recompute_norm: bool = False
                     ) -> None:
        (self.genomic_data_light, self.generative_model_light,
         self.pgen_model_light,
         self.seqgen_model_light) = define_pgen_model(self.custom_pgen_model_light,
                                                      self.chain_type_light,
                                                      vj=True, return_files=False)
        (self.genomic_data_heavy, self.generative_model_heavy,
         self.pgen_model_heavy,
         self.seqgen_model_heavy) = define_pgen_model(self.custom_pgen_model_heavy,
                                                      self.chain_type_heavy,
                                                      vj=False, return_files=False)

        if recompute_norm:
            self.norm_light = self.pgen_model_light.compute_regex_CDR3_template_pgen('CX{0,}')
            self.norm_heavy = self.pgen_model_heavy.compute_regex_CDR3_template_pgen('CX{0,}')
        else:
            self.norm_light= PRODUCTIVE_NORMS[self.chain_type_light]
            self.norm_heavy=  PRODUCTIVE_NORMS[self.chain_type_heavy]

        self.norm_productive = self.norm_heavy * self.norm_light

    def add_features(self,
                     include_joint_genes: bool = True
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
        features=[]
        features += [['l_l' + str(L)] for L in range(1, self.max_L + 1)]
        features += [['l_h' + str(L)] for L in range(1, self.max_L + 1)]

        for aa in self.amino_acids:
            features += [['a_l' + aa + str(L)] for L in range(self.max_depth)]
            features += [['a_l' + aa + str(L)] for L in range(-self.max_depth, 0)]
        for aa in self.amino_acids:
            features += [['a_h' + aa + str(L)] for L in range(self.max_depth)]
            features += [['a_h' + aa + str(L)] for L in range(-self.max_depth, 0)]

        if self.include_joint_genes:
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
        else:
            features += [[v]
                         for v in set(['v_l' + gene_to_num_str(genV[0],'V')[1:]
                                       for genV in self.genomic_data_light.genV])]
            features += [[j]
                         for j in set(['j_l'  + gene_to_num_str(genJ[0],'J')[1:]
                                       for genJ in self.genomic_data_light.genJ])]
            features += [[v]
                         for v in set(['v_h' + gene_to_num_str(genV[0],'V')[1:]
                                       for genV in self.genomic_data_heavy.genV])]
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

        vh_key = ('v_h' + gene_to_num_str(seq[1], 'V')[1:],)
        jh_key = ('j_h' + gene_to_num_str(seq[2], 'J')[1:],)
        vhjh_key = vh_key + jh_key
        if vh_key in feature_dict:
            seq_features.add(feature_dict[vh_key])
        if jh_key in feature_dict:
            seq_features.add(feature_dict[jh_key])
        if vhjh_key in feature_dict:
            seq_features.add(feature_dict[vhjh_key])

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

        vl_key = ('v_l' + gene_to_num_str(seq[4], 'V')[1:],)
        jl_key = ('j_l' + gene_to_num_str(seq[5], 'J')[1:],)
        vljl_key = vl_key + jl_key

        if vl_key in feature_dict:
            seq_features.add(feature_dict[vl_key])
        if jl_key in feature_dict:
            seq_features.add(feature_dict[jl_key])
        if vljl_key in feature_dict:
            seq_features.add(feature_dict[vljl_key])

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
        zipped = zip([self.custom_pgen_model_light, self.custom_pgen_model_heavy],
                     [self.chain_type_light, self.chain_type_heavy],
                     ['light_chain', 'heavy_chain'])
        for custom_pgen_model, chain_type, folder_name in zipped:
            try:
                if self.custom_pgen_model is None: main_folder = os.path.join(FILEDIR, 'default_models', chain_type)
                else: main_folder = custom_pgen_model
            except:
                main_folder = os.path.join(FILEDIR, 'default_models', chain_type)

            chain_dir = os.path.join(save_dir, folder_name)
            if not os.path.isdir(chain_dir): os.mkdir(chain_dir)
            shutil.copy2(os.path.join(main_folder, 'model_params.txt'), chain_dir)
            shutil.copy2(os.path.join(main_folder, 'model_marginals.txt'), chain_dir)
            shutil.copy2(os.path.join(main_folder, 'V_gene_CDR3_anchors.csv'), chain_dir)
            shutil.copy2(os.path.join(main_folder, 'J_gene_CDR3_anchors.csv'), chain_dir)

    def _load_pgen_model(self,
                         load_dir: str
                        ) -> None:
        self.custom_pgen_model_light = os.path.join(load_dir, 'light_chain')
        self.custom_pgen_model_heavy = os.path.join(load_dir, 'heavy_chain')
        self.define_models(recompute_norm=False)

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
