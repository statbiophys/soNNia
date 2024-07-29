#!/usr/bin/env python
# -*- coding: utf-8 -*-

# @author: Giulio Isacchini
import itertools
import logging
import multiprocessing as mp
import os
from typing import *

logging.getLogger('tensorflow').disabled = True

import numpy as np
from tensorflow import keras
import tensorflow.keras.backend as K
from tqdm import tqdm

from sonnia.sonia import Sonia
from sonnia.utils import define_pgen_model, gene_to_num_str, get_model_dir

ACROSS_CHAIN_FEATURES_OPTIONS = {'jhjl', 'jhvl', 'vhjl', 'vhvl'}

class SoniaPaired(Sonia):
    def __init__(
        self,
        *args: Tuple[Any],
        ppost_model: Optional[str] = None,
        pgen_model_light: Optional[str] = None,
        pgen_model_heavy: Optional[str] = None,
        recompute_productive_norm: bool = False,
        across_chain_features: Optional[Iterable[str]] = None,
        **kwargs: Dict[str, Any]
    ) -> None:
        if across_chain_features is None or not across_chain_features:
            self.across_chain_features = {}
        else:
            if isinstance(across_chain_features, str):
                raise TypeError('across_chain_features must be an iterable of strings.')
            else:
                try:
                    iter(across_chain_features)
                except Exception:
                    raise TypeError('across_chain_features must be an iterable of strings.')
                else:
                    self.across_chain_features = set(across_chain_features)

            if not self.across_chain_features.issubset(ACROSS_CHAIN_FEATURES_OPTIONS):
                options = f'{ACROSS_CHAIN_FEATURES_OPTIONS}'[1:-1]
                raise RuntimeError(f'across_chain_features ({across_chain_features}) '
                                   'contains unacceptable options. across_chain_features '
                                   'must be None or be an iterable containing only '
                                   f'the following strings: {options}.')

        if ppost_model is None and (pgen_model_light is None or pgen_model_heavy is None):
            raise RuntimeError('Either ppost_model must not be None or both pgen_model_light '
                               'and pgen_model_heavy must not be None.')
        elif ppost_model is not None and pgen_model_light is not None and pgen_model_heavy is not None:
            raise RuntimeError('ppost_model, pgen_model_light, and pgen_model_heavy '
                               'all cannot be given. Either give ppost_model, or '
                               'give pgen_model_light and pgen_model_heavy.')
        elif ppost_model is not None:
            if pgen_model_light is not None:
                raise RuntimeError('pgen_model_light must not be None if ppost_model is given.')
            if pgen_model_heavy is not None:
                raise RuntimeError('pgen_model_heavy must not be None if ppost_model is given.')
            model_dir = get_model_dir(ppost_model, True)
            self.pgen_model_light = os.path.join(model_dir, 'light_chain')
            self.pgen_model_heavy = os.path.join(model_dir, 'heavy_chain')
        else:
            self.pgen_model_light = pgen_model_light
            self.pgen_model_heavy = pgen_model_heavy

        if ppost_model is None:
            self.recompute_productive_norm = True
        else:
            self.recompute_productive_norm = recompute_productive_norm

        self.load_pgen_models()

        Sonia.__init__(self, *args, ppost_model=ppost_model, **kwargs)

    def load_pgen_models(
        self
    ) -> None:
        (self.genomic_data_light, self.generative_model_light,
         self.pgen_model_light, self.seqgen_model_light,
         self.norm_light, self.pgen_dir_light,
         model_str, self.chain_light) = define_pgen_model(self.pgen_model_light,
                                                          self.recompute_productive_norm,
                                                          True, True, True)
        if model_str != 'VJ':
            raise RuntimeError('A VDJ model was given to pgen_model_light. Please '
                               'rerun and point pgen_model_light to a VJ pgen model.')

        (self.genomic_data_heavy, self.generative_model_heavy,
         self.pgen_model_heavy, self.seqgen_model_heavy,
         self.norm_heavy, self.pgen_dir_heavy,
         model_str, self.chain_heavy) = define_pgen_model(self.pgen_model_heavy,
                                                          self.recompute_productive_norm,
                                                          True, True, True)
        if model_str != 'VDJ':
            raise RuntimeError('A VJ model was given to pgen_model_heavy. Please '
                               'rerun and point pgen_model_heavy to a VDJ pgen model.')

        valid_chain_pairs = [('IGL', 'IGH'), ('IGK', 'IGH'),
                             ('TRA', 'TRB'), ('TRG', 'TRD')]

        if (self.chain_light, self.chain_heavy) not in valid_chain_pairs:
            valid_chain_pairs_str = f'{valid_chain_pairs}'[1:-1]
            raise RuntimeError(f'A light-heavy chain pair of {(self.chain_light, self.chain_heavy)} '
                               'does constitute a valid receptor. Acceptable chain '
                               f'pairs: {valid_chain_pairs_str}.')

        for chain in ['light', 'heavy']:
            with open(os.path.join(getattr(self, f'pgen_dir_{chain}'), 'model_params.txt'), 'r') as fin:
                sep = 0
                error_rate = ''
                lines = fin.read().splitlines()
                while len(error_rate) < 1:
                    error_rate = lines[-1 + sep]
                    sep -= 1
                setattr(self, f'error_rate_{chain}', float(error_rate))

        if self.recompute_productive_norm:
            self.norm_productive = self.norm_heavy * self.norm_light

    def add_features(
        self,
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

        if 'vhvl' in self.across_chain_features:
            features += [[vh, vl]
                         for vh in set(['v_h'  + gene_to_num_str(genV[0],'V')[1:]
                                        for genV in self.genomic_data_heavy.genV])
                         for vl in set(['v_l' + gene_to_num_str(genV[0],'V')[1:]
                                        for genV in self.genomic_data_light.genV])]
        if 'jhjl' in self.across_chain_features:
            features += [[jh, jl]
                         for jh in set(['j_h'  + gene_to_num_str(genJ[0],'J')[1:]
                                        for genJ in self.genomic_data_heavy.genJ])
                         for jl in set(['j_l'  + gene_to_num_str(genJ[0],'J')[1:]
                                        for genJ in self.genomic_data_light.genJ])]
        if 'vhjl' in self.across_chain_features:
            features += [[vh, jl]
                         for vh in set(['v_h'  + gene_to_num_str(genV[0],'V')[1:]
                                        for genV in self.genomic_data_heavy.genV])
                         for jl in set(['j_l'  + gene_to_num_str(genJ[0],'J')[1:]
                                        for genJ in self.genomic_data_light.genJ])]
        if 'jhvl' in self.across_chain_features:
            features += [[jh, vl]
                         for jh in set(['j_h'  + gene_to_num_str(genJ[0],'J')[1:]
                                        for genJ in self.genomic_data_heavy.genJ])
                         for vl in set(['v_l' + gene_to_num_str(genV[0],'V')[1:]
                                        for genV in self.genomic_data_light.genV])]

        self.update_model(add_features=features)

    def find_seq_features(
        self,
        seq: Sequence[str],
        feature_dict: Optional[Dict[Tuple[str], int]] = None
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
        if feature_dict is None:
            feature_dict = self.feature_dict

        seq_features = set()

        # NOTE It's quicker to have the code written explicitly than perform
        # a for loop.
        # Heavy chain.
        cdr3_len_h = len(seq[0])
        cdr3_len_key_h = (f'l_h{cdr3_len_h}',)
        if cdr3_len_key_h in feature_dict:
            seq_features.add(feature_dict[cdr3_len_key_h])

        for idx, amino_acid in enumerate(list(seq[0])):
            fwd_key = (f'a_h{amino_acid}{idx}',)
            bkd_key = (f'a_h{amino_acid}{idx - cdr3_len_h}',)
            if fwd_key in feature_dict:
                seq_features.add(feature_dict[fwd_key])
            if bkd_key in feature_dict:
                seq_features.add(feature_dict[bkd_key])

        v_key_h = ('v_h' + gene_to_num_str(seq[1], 'V')[1:],)
        j_key_h = ('j_h' + gene_to_num_str(seq[2], 'J')[1:],)
        vj_key = v_key_h + j_key_h
        vjl_key = vj_key + cdr3_len_key_h
        if v_key_h in feature_dict:
            seq_features.add(feature_dict[v_key_h])
        if j_key_h in feature_dict:
            seq_features.add(feature_dict[j_key_h])
        if vj_key in feature_dict:
            seq_features.add(feature_dict[vj_key])
        if vjl_key in feature_dict:
            seq_features.add(feature_dict[vjl_key])

        # Light chain.
        cdr3_len_l = len(seq[3])
        cdr3_len_key_l = (f'l_l{cdr3_len_l}',)
        if cdr3_len_key_l in feature_dict:
            seq_features.add(feature_dict[cdr3_len_key_l])

        for idx, amino_acid in enumerate(list(seq[3])):
            fwd_key = (f'a_l{amino_acid}{idx}',)
            bkd_key = (f'a_l{amino_acid}{idx - cdr3_len_l}',)
            if fwd_key in feature_dict:
                seq_features.add(feature_dict[fwd_key])
            if bkd_key in feature_dict:
                seq_features.add(feature_dict[bkd_key])

        v_key_l = ('v_l' + gene_to_num_str(seq[4], 'V')[1:],)
        j_key_l = ('j_l' + gene_to_num_str(seq[5], 'J')[1:],)
        vj_key = v_key_l + j_key_l
        vjl_key = vj_key + cdr3_len_key_l
        if v_key_l in feature_dict:
            seq_features.add(feature_dict[v_key_l])
        if j_key_l in feature_dict:
            seq_features.add(feature_dict[j_key_l])
        if vj_key in feature_dict:
            seq_features.add(feature_dict[vj_key])
        if vjl_key in feature_dict:
            seq_features.add(feature_dict[vjl_key])

        vhvl_key = v_key_h + v_key_l
        vhjl_key = v_key_h + j_key_l
        jhvl_key = j_key_h + v_key_l
        jhjl_key = j_key_h + j_key_l

        if vhvl_key in feature_dict:
            seq_features.add(feature_dict[vhvl_key])
        if vhjl_key in feature_dict:
            seq_features.add(feature_dict[vhjl_key])
        if jhvl_key in feature_dict:
            seq_features.add(feature_dict[jhvl_key])
        if jhjl_key in feature_dict:
            seq_features.add(feature_dict[jhjl_key])

        return list(seq_features)

    def generate_sequences_pre(
        self,
        num_seqs: int = 1,
        nucleotide: bool = False,
        error_rate_light: Optional[np.float64] = None,
        error_rate_heavy: Optional[np.float64] = None,
        add_error: bool = False,
        seed: Optional[int | np.random.Generator | np.random.BitGenerator | np.random.SeedSequence] = None,
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
        from sonnia.utils import add_random_error, generate_paired_sequence
        from olga.utils import nt2aa

        if error_rate_light is None:
            error_rate_light = self.error_rate_light
        if error_rate_heavy is None:
            error_rate_heavy = self.error_rate_heavy

        if seed is None:
            rng = self.rng
        else:
            rng = np.random.default_rng(seed)

        if num_seqs > 5000:
            seeds = rng.integers(low=0, high=2**32 - 1, size=num_seqs)
            zipped = zip(itertools.repeat(self.seqgen_model_light, num_seqs),
                         itertools.repeat(self.seqgen_model_heavy, num_seqs),
                         itertools.repeat(self.genomic_data_light, num_seqs),
                         itertools.repeat(self.genomic_data_heavy, num_seqs),
                         seeds,
                         itertools.repeat(add_error, num_seqs),
                         itertools.repeat(error_rate_light, num_seqs),
                         itertools.repeat(error_rate_heavy, num_seqs))

            with mp.Pool(processes=self.processes) as pool:
                seqs = pool.starmap(generate_paired_sequence, zipped)
        else:
            seqs = []
            np.random.seed(rng.integers(0, 2**32 - 1))
            for i in tqdm(range(int(num_seqs))):
                seq_light = self.seqgen_model_light.gen_rnd_prod_CDR3(conserved_J_residues='ABCEDFGHIJKLMNOPQRSTUVWXYZ')
                seq_heavy = self.seqgen_model_heavy.gen_rnd_prod_CDR3(conserved_J_residues='ABCEDFGHIJKLMNOPQRSTUVWXYZ')

                if add_error:
                    err_seq = add_random_error(seq_light[0], error_rate_light)
                    seq_light = [err_seq, nt2aa(err_seq), seq_light[2], seq_light[3]]
                    err_seq = add_random_error(seq_heavy[0], error_rate_heavy)
                    seq_heavy = [err_seq, nt2aa(err_seq), seq_heavy[2], seq_heavy[3]]

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

    def compute_all_pgens(
        self,
        seqs: Iterable[Iterable[str]],
        include_genes: bool = True
    ) -> np.ndarray:
        '''
        Compute Pgen of sequences using OLGA
        '''
        if include_genes:
            with mp.Pool(processes=self.processes) as pool:
                f1 = pool.map(compute_pgen_expand_heavy, zip(seqs, itertools.repeat(self.pgen_model_heavy)))
                f2 = pool.map(compute_pgen_expand_light, zip(seqs, itertools.repeat(self.pgen_model_light)))
            return np.array(f1) * np.array(f2)

        with mp.Pool(processes=self.processes) as pool:
            f1 = pool.map(compute_pgen_expand_novj_heavy, zip(seqs, itertools.repeat(self.pgen_model_heavy)))
            f2 = pool.map(compute_pgen_expand_novj_light, zip(seqs, itertools.repeat(self.pgen_model_light)))
        return np.array(f1) * np.array(f2)

    def set_gauge(
        self
    ) -> None:
        '''
        placeholder for gauges.
        '''
        pass

    def _save_pgen_model(
        self,
        save_dir: str
    ) -> None:
        import shutil
        zipped = zip([self.pgen_dir_light, self.pgen_dir_heavy],
                     ['light_chain', 'heavy_chain'])
        for pgen_dir, folder_name in zipped:
            chain_dir = os.path.join(save_dir, folder_name)
            if not os.path.isdir(chain_dir): os.mkdir(chain_dir)
            shutil.copy2(os.path.join(pgen_dir, 'model_params.txt'), chain_dir)
            shutil.copy2(os.path.join(pgen_dir, 'model_marginals.txt'), chain_dir)
            shutil.copy2(os.path.join(pgen_dir, 'V_gene_CDR3_anchors.csv'), chain_dir)
            shutil.copy2(os.path.join(pgen_dir, 'J_gene_CDR3_anchors.csv'), chain_dir)

def compute_pgen_expand_light(
    x
) -> float:
    return x[1].compute_aa_CDR3_pgen(x[0][3],x[0][4],x[0][5])

def compute_pgen_expand_heavy(
    x
) -> float:
    return x[1].compute_aa_CDR3_pgen(x[0][0],x[0][1],x[0][2])

def compute_pgen_expand_novj_light(
    x
) -> float:
    return x[1].compute_aa_CDR3_pgen(x[0][3])

def compute_pgen_expand_novj_heavy(
    x
) -> float:
    return x[1].compute_aa_CDR3_pgen(x[0][0])
