#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 30 12:06:58 2019
@author: Giulio Isacchini and Zachary Sethna
"""
from __future__ import print_function, division, absolute_import
import inspect
import itertools
import logging
import multiprocessing as mp
import os
from typing import *
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

import keras
from keras.callbacks import TerminateOnNaN
import keras.ops as ko
from keras.layers import Dense, Input, Lambda
from keras.losses import BinaryCrossentropy
from keras.models import load_model, Model
from keras.optimizers import RMSprop
from keras.regularizers import l1_l2, l2
import numpy as np
from numpy.typing import ArrayLike, NDArray
import scipy.sparse as sparse
from tqdm import tqdm

from sonnia.sonia_dataset import SoniaDataset
from sonnia.utils import (
    CSV_READER_PARAMS, compute_pgen_expand, compute_pgen_expand_novj, define_pgen_model,
    filter_seqs, gene_to_num_str, get_model_dir, partial_joint_marginals
)

FILTER_SEQS_PARAMS = inspect.signature(filter_seqs).parameters.keys()

logging.getLogger().setLevel(logging.INFO)
logging.basicConfig(format='%(asctime)s: %(message)s')

GENE_FEATURE_OPTIONS = {'vjl', 'joint_vj', 'indep_vj', 'v', 'j', 'none'}

class Sonia(object):
    """Class used to infer a Q selection model.
    Attributes
    ----------
    features : ndarray
        Array of feature lists. Each list contains individual subfeatures which
            all must be satisfied.
    features_dict : dict
        Dictionary keyed by tuples of the feature lists. Values are the index
        of the feature, i.e. self.features[self.features_dict[tuple(f)]] = f.
    constant_features : list
        List of feature strings to not update parameters during learning. These
            features are still used to compute energies (not currently used)
    data_seqs : list
        Data sequences used to infer selection model. Note, each 'sequence'
        is a list where the first element is the CDR3 sequence which is
        followed by any V or J genes.
    gen_seqs : list
        Sequences from generative distribution used to infer selection model.
        Note, each 'sequence' is a list where the first element is the CDR3
        sequence which is followed by any V or J genes.
    data_seq_features : list
        Lists of features that data_seqs project onto
    gen_seq_features : list
        Lists of features that gen_seqs project onto
    data_marginals : ndarray
        Array of the marginals of each feature over data_seqs
    gen_marginals : ndarray
        Array of the marginals of each feature over gen_seqs
    model_marginals : ndarray
        Array of the marginals of each feature over the model weighted gen_seqs
    L1_converge_history : list
        L1 distance between data_marginals and model_marginals at each
        iteration.
    chain_type : str
        Type of receptor. This specification is used to determine gene names
        and allow integrated OLGA sequence generation. Options: 'humanTRA',
        'humanTRB' (default), 'humanIGH', 'humanIGL', 'humanIGK' and 'mouseTRB'.
    l2_reg : float or None
        L2 regularization. If None (default) then no regularization.
    Methods
    ----------
    seq_feature_proj(feature, seq)
        Determines if a feature matches/is found in a sequence.
    find_seq_features(seq, features = None)
        Determines all model features of a sequence.
    compute_energy(seqs_features)
        Computes the energies of a list of seq_features according to the model.
    compute_marginals(self, features = None, seq_model_features = None, seqs = None, use_flat_distribution = False)
        Computes the marginals of features over a set of sequences.
    infer_selection(self, epochs = 20, batch_size=5000, initialize = True, seed = None)
        Infers model parameters (energies for each feature).
    update_model_structure(self,output_layer=[],input_layer=[],initialize=False)
        Sets keras model structure and compiles.
    update_model(self, add_data_seqs = [], add_gen_seqs = [], add_features = [], remove_features = [], add_constant_features = [], update_marginals = False, update_seq_features = False)
        Updates model by adding/removing model features or data/generated seqs.
        Marginals and seq_features can also be updated.
    add_generated_seqs(self, num_gen_seqs = 0, reset_gen_seqs = True)
        Generates synthetic sequences using OLGA and adds them to gen_seqs.
    plot_model_learning(self, save_name = None)
        Plots current marginal scatter plot as well as L1 convergence history.
    save_model(self, save_dir, attributes_to_save = None)
        Saves the model.
    load_model(self, load_dir)
        Loads a model.
    """
    def __init__(
        self,
        ppost_model: Optional[str] = None,
        data_seqs: List[Sequence[str]] = [],
        gen_seqs: List[Sequence[str]] = [],
        pgen_model: Optional[str] = None,
        load_seqs: bool = True,
        gene_features: str = 'joint_vj',
        include_aminoacids: bool = True,
        features: Sequence[Sequence[str]] = [],
        recompute_productive_norm: bool = False,
        max_depth: int = 25,
        max_L: int = 30,
        objective: str = 'BCE',
        l2_reg: float = 0.,
        l1_reg: float = 0.,
        gamma: float = 1.,
        min_energy_clip: int = -5,
        max_energy_clip: int = 10,
        seed: Optional[int | np.random.Generator | np.random.BitGenerator | np.random.SeedSequence] = None,
        processes: Optional[int] = None,
        **kwargs: Dict[str, Any]
    ) -> None:
        """
        Init Sonia/SoNNia object.

        Parameters
        ----------
        load_seqs : bool, default True
            Load the data and generated sequences used for training the ppost model.
        seed : int, optional
            The seed used for the random number generator.
        **kwargs : dict of {str : any}
            Keyword arguments for sonnia.utils.filter_seqs for preprocessing.
        """
        for keyword in kwargs:
            if keyword not in CSV_READER_PARAMS and keyword not in FILTER_SEQS_PARAMS:
                raise RuntimeError(f'Unknown keyword: {keyword}.')

        if gene_features not in GENE_FEATURE_OPTIONS:
            gene_feature_options_str = f'{GENE_FEATURE_OPTIONS}'[1:-1]
            raise ValueError(f'{gene_features} is not a valid option for '
                             'gene_features. gene_features must be one of '
                             f'{gene_feature_options_str}.')

        if ppost_model is None:
            self.recompute_productive_norm = True
        else:
            self.recompute_productive_norm = recompute_productive_norm

        if 'Paired' in type(self).__name__:
            pass
        else:
            if ppost_model is None and pgen_model is None:
                raise ValueError('Both ppost_model and pgen_model cannot be None.')
            elif ppost_model is not None and pgen_model is None:
                self.pgen_model = ppost_model
            elif ppost_model is None and pgen_model is not None:
                self.pgen_model = pgen_model
            else:
                raise ValueError('Both ppost_model and pgen_model cannot be given. '
                                 'One of them must be None.')
            self.load_pgen_model()

        self.features = np.array(features, dtype=object)
        self.feature_dict = {tuple(f): i for i, f in enumerate(self.features)}
        self.data_seqs = []
        self.gen_seqs = []
        self.data_encoding = np.array([])
        self.gen_encoding = np.array([])
        self.data_marginals = np.zeros(len(features))
        self.gen_marginals = np.zeros(len(features))
        self.model_marginals = np.zeros(len(features))
        self.L1_converge_history = []
        self.l2_reg = l2_reg
        self.l1_reg = l1_reg
        self.min_energy_clip = min_energy_clip
        self.max_energy_clip = max_energy_clip
        self.likelihood_train = []
        self.likelihood_test = []
        self.objective = objective
        self.gene_features = gene_features
        self.include_aminoacids = include_aminoacids
        self.max_depth = max_depth
        self.max_L = max_L
        if processes is None: self.processes = mp.cpu_count()
        self.gamma = gamma
        self.Z = 1.
        self.amino_acids = 'ACDEFGHIKLMNPQRSTVWY'

        if ppost_model is None:
            self.update_model(add_data_seqs=data_seqs, add_gen_seqs=gen_seqs, **kwargs)
            self.add_features()
        else:
            self.load_model(ppost_model=ppost_model, load_seqs=load_seqs)
            if len(data_seqs) != 0: self.update_model(add_data_seqs=data_seqs, **kwargs)
            if len(gen_seqs) != 0: self.update_model(add_gen_seqs=gen_seqs, **kwargs)

        if seed is not None:
            self.rng = np.random.default_rng(seed)
        else:
            self.rng = np.random.default_rng()

    def add_features(
        self
    ) -> None:
        """
        Generate a list of feature_lsts for L/R pos model.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        features = []
        if self.gene_features == 'vjl':
            features += [[v, j, 'l'+str(l)]
                         for v in set([gene_to_num_str(genV[0],'V')
                                       for genV in self.genomic_data.genV])
                         for j in set([gene_to_num_str(genJ[0],'J')
                                       for genJ in self.genomic_data.genJ])
                         for l in range(1, self.max_L + 1)]
        else:
            features += [['l' + str(L)] for L in range(1, self.max_L + 1)]

        if self.include_aminoacids:
            for aa in self.amino_acids:
                features += [['a' + aa + str(L)] for L in range(-self.max_depth, self.max_depth)]

        if self.gene_features == 'joint_vj':
            features += [[v, j]
                         for v in set([gene_to_num_str(genV[0],'V')
                                       for genV in self.genomic_data.genV])
                         for j in set([gene_to_num_str(genJ[0],'J')
                                       for genJ in self.genomic_data.genJ])]

        if self.gene_features in {'indep_vj', 'v'}:
            features += [[v] for v in set([gene_to_num_str(genV[0],'V')
                                           for genV in self.genomic_data.genV])]
        if self.gene_features in {'indep_vj', 'j'}:
            features += [[j] for j in set([gene_to_num_str(genJ[0],'J')
                                           for genJ in self.genomic_data.genJ])]

        self.update_model(add_features=features)

    def find_seq_features(
        self,
        seq: Sequence[str],
        feature_dict: Optional[Dict[Tuple[str], int]] = None
    ) -> List[int]:
        """
        Obtain the one-hot encoding of features for a sequence.

        If no features are provided, the left/right indexing amino acid model
        features will be assumed.

        Parameters
        ----------
        seq : sequence of str
            The order must be the CDR3 amino acid sequence, V gene, and J gene.
        feature_dict: dict of { tuple of str : int}
            A dictionary where the key is the feature and the value is the index
            in the one-hot encoding..

        Returns
        -------
        None
        """
        if feature_dict is None:
            feature_dict = self.feature_dict

        seq_features = set()

        cdr3_len = len(seq[0])
        cdr3_len_key = (f'l{cdr3_len}',)
        if cdr3_len_key in feature_dict:
            seq_features.add(feature_dict[cdr3_len_key])

        for idx, amino_acid in enumerate(list(seq[0])):
            fwd_key = (f'a{amino_acid}{idx}',)
            bkd_key = (f'a{amino_acid}{idx - cdr3_len}',)
            if fwd_key in feature_dict:
                seq_features.add(feature_dict[fwd_key])
            if bkd_key in feature_dict:
                seq_features.add(feature_dict[bkd_key])

        v_key = (gene_to_num_str(seq[1], 'V'),)
        j_key = (gene_to_num_str(seq[2], 'J'),)
        vj_key = v_key + j_key
        vjl_key = v_key + j_key + cdr3_len_key
        if v_key in feature_dict:
            seq_features.add(feature_dict[v_key])
        if j_key in feature_dict:
            seq_features.add(feature_dict[j_key])
        if vj_key in feature_dict:
            seq_features.add(feature_dict[vj_key])
        if vjl_key in feature_dict:
            seq_features.add(feature_dict[vjl_key])

        return list(seq_features)

    def encode_data(
        self,
        sequences: Sequence[Sequence[str]],
        features: Optional[Sequence[Tuple[str]]] = None,
    ) -> sparse.csr_array:
        """
        One-hot encode all the features from the given sequences with a sparse matrix.

        Parameters
        ----------
        sequences : sequence of sequence of str
            A sequence of receptor sequences (e.g., CDR3 amino acid, V gene,
            and J gene strings.)
        features: sequence of tuples of str, optional
            An iterable containing tuples of features.

        Returns
        -------
        csr_arr : scipy.sparse.csr_array
            A sparse array representation of the one-hot encoding.
        """
        if features is not None:
            feature_dict = {
                tuple(feature): idx for idx, feature in enumerate(features)
            }
            num_features = len(features)
        else:
            num_features = len(self.features)

        indices = []
        indptr = [0]

        tqdm_desc = 'Encoding sequence features'
        for seq in tqdm(sequences, position=0, desc=tqdm_desc):
            specified_features = self.find_seq_features(seq, features)
            indices += specified_features
            indptr.append(len(specified_features) + indptr[-1])

        data = np.ones(len(indices), dtype=np.int8)
        csr_arr = sparse.csr_array(
            (data, indices, indptr),
            shape=(len(sequences), num_features)
        )
        return csr_arr

    def encoding_to_feature_strs(
        self,
        encoding: sparse.csr_array,
        features: Optional[Sequence[Tuple[str]]] = None
    ) -> Sequence[Sequence[Tuple[str]]]:
        """
        Convert the one-hot encoded sequences to a list of their sequence features.

        Parameters
        ----------
        encoding : scipy.sparse.csr_array
            The sparse representation of one-hot encoded sequence features.
        features : sequence of tuples of str, optional
            A sequence containing tuples of features.

        Returns
        -------
        feature_strs : sequence of sequence of tuple of str
            Each sublist contains all the features of the sequence represented
            as tuples of strings.
        """
        if features is None:
            np_features = np.fromiter(self.feature_dict.keys(), dtype=object)
        else:
            np_features = np.array(features)

        feature_strs = []
        zipped = zip(encoding.indptr[:-1], encoding.indptr[1:])
        tqdm_desc = 'Getting feature strings'
        for idx1, idx2 in tqdm(
            zipped, total=encoding.shape[0], position=0, desc=tqdm_desc
        ):
            feature_strs.append(np_features[encoding.indices[idx1:idx2]].tolist())

        return feature_strs

    def encoding_to_feature_idxs(
        self,
        encoding: sparse.csr_array,
    ) -> Sequence[Sequence[int]]:
        """
        Convert the one-hot encoded sequences to a list of the indices corresponding
        to their sequence features.

        Parameters
        ----------
        encoding : scipy.sparse.csr_array
            The sparse representation of one-hot encoded sequences.

        Returns
        -------
        feature_strs : sequence of sequence of tuple of str
            Each sublist contains all the features of the sequence represented
            as ints.
        """
        feature_idxs = []
        zipped = zip(encoding.indptr[:-1], encoding.indptr[1:])
        tqdm_desc = 'Getting feature indices'
        for idx1, idx2 in tqdm(
            zipped, total=encoding.shape[0], position=0, desc=tqdm_desc,
        ):
            feature_idxs.append(encoding.indices[idx1:idx2].tolist())

        return feature_idxs

    def compute_energy(
        self,
        encoding: sparse.csr_array,
        chunksize: int = int(1e6),
        verbose: bool = True,
    ) -> NDArray[np.float32]:
        """
        Compute the energy of a list of sequences according to the model.

        Parameters
        ----------
        encoding : scipy.sparse.csr_array
            Sparse representation of one-hot-encoded sequence features.
        chunksize : int, default int(1e6)
            The amount of sequences to be evaluated in a single call to the model.
            Since a dense one-hot encoding is used, RAM usage will blow up for
            very large amounts of sequences.
        verbose : bool, default True
            Show a progress bar for the chunks being evaluated.

        Returns
        -------
        energies : numpy.ndarray of numpy.float32
            Energies of sequences according to the model.
        """
        length_encoding = encoding.shape[0]
        num_slices = (
            length_encoding // chunksize
            + 1 * (length_encoding % chunksize != 0)
        )

        energies = []
        tqdm_desc = 'Computing energies'
        disable = not verbose
        for idx in tqdm(
            range(num_slices), position=0, desc=tqdm_desc, disable=disable
        ):
            start_idx = idx * chunksize
            encoding_slice = encoding[start_idx:start_idx + chunksize]

            dense_encoding = encoding_slice.toarray()

            if hasattr(self, 'split_encoding'):
                dense_encoding = self.split_encoding(dense_encoding)
            try:
                energies_slice = self.model(dense_encoding)[:, 0].numpy()
            except Exception as e:
                if 'Failed copying' in str(e):
                    raise RuntimeError(
                        'There is not enough GPU memory available to copy the '
                        'one-hot encoding from CPU to GPU. Try requesting more '
                        'GPU memory or using a smaller chunksize when calling '
                        'this function (compute_energy).'
                    )
                else:
                    raise e
            energies.append(energies_slice)

        energies = np.concatenate(energies)
        return energies

    def compute_marginals(
        self,
        encoding: Optional[sparse.csr_array] = None,
        seqs: Optional[Sequence[Sequence[str]]] = None,
        features: Optional[Sequence[Tuple[str]]] = None,
        use_flat_distribution: bool = False,
    ) -> np.ndarray:
        """Computes the marginals of each feature over sequences.

        Computes marginals either with a flat distribution over the sequences
        or weighted by the model energies. Note, finding the features of each
        sequence takes time and should be avoided if it has already been done.
        If computing marginals of model features use the default setting to
        prevent searching for the model features a second time. Similarly, if
        seq_model_features has already been determined use this to avoid
        recalculating it.

        Parameters
        ----------
        encoding : scipy.sparse.csr_array, optional
            A sparse represention of the one hot encoded sequence features.
        seqs : sequence of sequence of str, optional
            List of sequences to compute the feature marginals over. Note, each
            'sequence' is a list where the first element is the CDR3 sequence
            which is followed by any V or J genes.
        features : sequence of tuple of str, optional
            List of features. This does not need to match the model
            features. If None, the model features will be used.
        use_flat_distribution : bool, default False
            Marginals will be computed using a flat distribution (each seq is
            weighted as 1) if True. If False, the marginals are computed using
            model weights (each sequence is weighted as exp(-E) = Q).

        Returns
        -------
        marginals : numpy.ndarray of numpy.float32
            Marginals of model features over seqs.
        """
        if encoding is None and seqs is None:
            raise RuntimeError('Both encoding and seqs cannot be None.')
        if encoding is not None and seqs is not None:
            raise RuntimeError('Both encoding and seqs cannot be given. '
                             'One of them must be None.')
        if encoding is not None and features is not None:
            raise RuntimeError('Both encoding and features cannot be given. '
                               'If features is given, seqs must be given to redo '
                               'the one-hot encoding.')

        if features is not None:
            num_features = len(features)
        else:
            num_features = len(self.features)

        if seqs is not None:
            encoding = self.encode_data(seqs, features)

        if use_flat_distribution:
            marginals = (
                np.bincount(encoding.indices, minlength=num_features) / encoding.shape[0]
            )
        else:
            energies = self.compute_energy(encoding)
            qs = sparse.csr_array(np.exp(-energies))
            marginals = (qs.dot(encoding) / qs.data.sum())
            # In older versions of scipy, performing a dot product ensued in
            # a two-dimensional array.
            marginals = marginals.toarray().ravel()
            return marginals

        return marginals

    def infer_selection(
        self,
        epochs: int = 10,
        batch_size: int = 5000,
        initialize: bool = True,
        seed: Optional[int | np.random.Generator | np.random.BitGenerator | np.random.SeedSequence] = None,
        validation_split: float = 0.2,
        verbose: int = 0,
        set_gauge: bool = True,
        sampling: Optional[str] = None
    ) -> None:
        """
        Infer model parameters, i.e. energies for each model feature.

        Parameters
        ----------
        epochs : int, default 10
            Maximum number of learning epochs
        intialize : bool, default True
            Resets data shuffle.
        batch_size : int, default 5000
            Size of the batches in the inference
        seed : int or np.random.Generator or np.random.BitGenerator or np.random.SeedSequence, optional
            Sets random seed.
        validation_split : float, default 0.2
            The fraction of data used for validation.
        verbose : bool, default False
            Output the training progress.
        set_gauge : bool, default True
            Set the gauge for the model output.
        sampling : str, default 'unbalanced'
            How the data seqs and gen seqs should be loaded into mini-batches.
            If 'legacy' the sonnia.SoniaDataset class will not be used, and
            a data sequence always appearing in a mini-batch will not be guaranteed.
            See sonnia.SoniaDataset for other sampling options.

        Returns
        -------
        None
        """
        if seed is not None:
            rng = np.random.default_rng(seed)
        else:
            rng = self.rng

        if initialize:
            self.X = sparse.vstack((self.data_encoding, self.gen_encoding))
            self.Y = np.zeros(
                self.data_encoding.shape[0] + self.gen_encoding.shape[0],
                dtype=np.int8
            )
            self.Y[self.data_encoding.shape[0]:] += 1

            shuffle = rng.permutation(self.X.shape[0])
            self.X = self.X[shuffle]
            self.Y = self.Y[shuffle]

        num_data_seqs = np.count_nonzero(self.Y == 0)
        if num_data_seqs == 0:
            raise RuntimeError('No data seqs were given. Cannot infer a selection model.')
        if num_data_seqs == self.Y.shape[0]:
            raise RuntimeError('No gen seqs were given. Cannot infer a selection model.')

        callbacks = [
            TerminateOnNaN(),
        ]

        if sampling is None:
            # Presently, GPU cannot process sparse arrays, so the encoding
            # must be in the dense representation.
            if hasattr(self, 'split_encoding'):
                input_data = self.split_encoding(self.X.toarray())
            else:
                input_data = self.X.toarray()
            self.learning_history = self.model.fit(
                input_data, self.Y, epochs=epochs, batch_size=batch_size,
                validation_split=validation_split, verbose=verbose, callbacks=callbacks,
            )
        else:
            if validation_split < 0 or validation_split >= 1:
                raise ValueError('validation_split must be in [0, 1).')
            if validation_split == 0:
                validation_data = None
            else:
                if hasattr(self, 'split_encoding'):
                    split_encoding = self.split_encoding
                else:
                    split_encoding = None

                val_end_idx = int(validation_split * len(self.Y))
                val_x, val_y = self.X[:val_end_idx], self.Y[:val_end_idx]
                train_x, train_y = self.X[val_end_idx:], self.Y[val_end_idx:]

                child_rngs = [
                    np.random.default_rng(child_state)
                    for child_state in rng.bit_generator._seed_seq.spawn(2)
                ]

                train_generator = SoniaDataset(
                    train_x, train_y, sampling, batch_size, seed=child_rngs[0],
                    split_encoding=split_encoding,
                )
                val_generator = SoniaDataset(
                    val_x, val_y, sampling, batch_size, seed=child_rngs[1],
                    split_encoding=split_encoding,
                )

                self.learning_history = self.model.fit(
                    train_generator, validation_data=val_generator, epochs=epochs,
                    batch_size=batch_size, verbose=verbose, callbacks=callbacks,
                )

        self.likelihood_train = -np.array(self.learning_history.history['_likelihood']) * 1.44
        self.likelihood_test = -np.array(self.learning_history.history['val__likelihood']) * 1.44
        self.model_params = self.model.get_weights()

        if np.isnan(self.likelihood_train).any() or np.isnan(self.likelihood_test).any():
            raise RuntimeError(
                'The training or validation likelihood history contains nans. '
                'Report a bug.'
            )

        logging.info('Finished training.')

        # set Z    
        self.energies_gen = self.compute_energy(self.gen_encoding)
        self.Z = np.mean(np.exp(-self.energies_gen))
        if set_gauge and self.gene_features != 'vjl': self.set_gauge()
        logging.info('Updating marginals.')
        self.update_model(update_marginals=True)
        logging.info('Finished updating marginals.')
        self.model_params = self.model.get_weights()

    def set_gauge(
        self
    ) -> None:
        """
        sets gauge such as sum(q)_i =1 at each position of CDR3 (left and right).
        """
        logging.info('Setting gauge.')
        model_energy_parameters = self.model.get_weights()[0].flatten()

        Gs_plus=[]
        for i in list(range(self.max_depth)):
            norm_p=sum([self.gen_marginals[self.feature_dict[( 'a' + aa + str(i),)]] for aa in self.amino_acids])
            if norm_p==0:
                G=1
            else:
                G = sum([self.gen_marginals[self.feature_dict[( 'a' + aa + str(i),)]] /norm_p *
                          np.exp(-model_energy_parameters[self.feature_dict[( 'a' + aa + str(i),)]])
                          for aa in self.amino_acids])
            Gs_plus.append(G)
            for aa in self.amino_acids: model_energy_parameters[self.feature_dict[( 'a' + aa + str(i),)]] += np.log(G)
        Gs_minus=[]
        for i in list(range(-self.max_depth,0)):
            norm_p=sum([self.gen_marginals[self.feature_dict[( 'a' + aa + str(i),)]] for aa in self.amino_acids])
            if norm_p==0:
                G=1
            else:
                G = sum([self.gen_marginals[self.feature_dict[( 'a' + aa + str(i),)]] /norm_p *
                          np.exp(-model_energy_parameters[self.feature_dict[( 'a' + aa + str(i),)]])
                          for aa in self.amino_acids])
            Gs_minus.append(G)
            for aa in self.amino_acids: model_energy_parameters[self.feature_dict[( 'a' + aa + str(i),)]] += np.log(G)
        for i in range(1,self.max_L+1):
            for j in list(range(i,self.max_depth)):
                model_energy_parameters[self.feature_dict[( 'l' + str(i),)]] += np.log(Gs_plus[j])
            for j in list(range(-self.max_depth,-i)):
                model_energy_parameters[self.feature_dict[( 'l' + str(i),)]] += np.log(Gs_minus[j])
        delta_Z=np.sum([np.log(g) for g in Gs_plus])+np.sum([np.log(g) for g in Gs_minus])

        self.min_energy_clip=self.min_energy_clip+delta_Z
        self.max_energy_clip=self.max_energy_clip+delta_Z
        self.Z=self.Z*np.exp(-delta_Z)
        self.update_model_structure(initialize=True)
        self.model.set_weights([np.array([model_energy_parameters]).T])

    def update_model_structure(
        self,
        output_layer: List = [],
        input_layer: List = [],
        initialize: bool = False
    ) -> bool:
        """
        Defines the model structure and compiles it.

        Parameters
        ----------
        structure : Sequential Model Keras
            structure of the model
        initialize: bool
            if True, it initializes to linear model, otherwise it updates to new structure
        """
        length_input = np.max([len(self.features), 1])
        min_clip = self.min_energy_clip
        max_clip = self.max_energy_clip
        l2_reg = self.l2_reg
        l1_reg = self.l1_reg

        if initialize:
            input_layer = Input(shape=(length_input,))
            output_layer = Dense(
                1, use_bias=False, activation='linear',
                kernel_regularizer=l1_l2(l2=l2_reg,l1=l1_reg)
            )(input_layer) #normal glm model

        # Define model
        clipped_out = Lambda(
            ko.clip, arguments={'x_min': min_clip, 'x_max': max_clip}
        )(output_layer)
        self.model = Model(inputs=input_layer, outputs=clipped_out)

        self.optimizer = RMSprop()
        if self.objective=='BCE':
            self.model.compile(
                optimizer=self.optimizer,
                loss=BinaryCrossentropy(from_logits=True),
                metrics=[
                    self._likelihood,
                    BinaryCrossentropy(from_logits=True, name='binary_crossentropy')
                ]
            )
        else:
            self.model.compile(
                optimizer=self.optimizer,
                loss=self._loss,
                metrics=[
                    self._likelihood,
                    BinaryCrossentropy(from_logits=True, name='binary_crossentropy')
                ]
            )
        self.model_params = self.model.get_weights()
        return True

    def _loss(
        self,
        y_true,
        y_pred
    ) -> float:
        """
        Loss function for keras training.

        We assume a model of the form P(x)=exp(-E(x))P_0(x)/Z.
        We minimize the neg-loglikelihood: <-logP> = log(Z) - <-E>.
        Normalization of P gives Z=<exp(-E)>_{P_0}.
        We fix the gauge by adding the constraint (Z-1)**2 to the likelihood.
        """
        y = ko.cast(y_true, dtype='bool')
        data = ko.nan_to_num(ko.mean(y_pred[ko.logical_not(y)]))
        gen = ko.nan_to_num(
            ko.logsumexp(-y_pred[y]) - ko.log(ko.sum(y_true)), neginf=0
        )
        return gen + data + self.gamma * gen * gen

    def _likelihood(
        self,
        y_true,
        y_pred
    ) -> float:
        """
        This is the "I" loss in the arxiv paper with added regularization

        A likelihood value of nan means no data sequences were present in
        the mini-batch.
        """
        y = ko.cast(y_true, dtype='bool')
        data = ko.nan_to_num(ko.mean(y_pred[ko.logical_not(y)]))
        gen = ko.nan_to_num(
            ko.logsumexp(-y_pred[y]) - ko.log(ko.sum(y_true)), neginf=0
        )
        return gen + data

    def update_model(
        self,
        add_data_seqs: List[Sequence[str]] = [],
        add_gen_seqs: List[Sequence[str]] = [],
        add_features: List[Sequence[str]] = [],
        remove_features: List[Sequence[str]] = [],
        add_constant_features: List[Sequence[str]] = [],
        update_marginals: bool = False,
        update_seq_features: bool = False,
        **kwargs
    ) -> None:
        """
        Update the model attributes

        This method is used to add/remove model features or data/generated
        sequences. These changes will be propagated through the class to update
        any other attributes that need to match (e.g. the marginals or
        seq_features).

        Parameters
        ----------
        add_data_seqs : list
            List of CDR3 sequences to add to data_seq pool.
        add_gen_seqs : list
            List of CDR3 sequences to add to data_seq pool.
        add_gen_seqs : list
            List of CDR3 sequences to add to data_seq pool.
        add_features : list
            List of feature lists to add to self.features
        remove_featurese : list
            List of feature lists and/or indices to remove from self.features
        add_constant_features : list
            List of feature lists to add to constant features. (Not currently used)
        update_marginals : bool
            Specifies to update marginals.
        update_seq_features : bool
            Specifies to update seq features.
        **kwargs : dict of {str : any}
            Keyword arguments for sonnia.utils.filter_seqs for preprocessing.

        Attributes set
        --------------
        features : list
            List of model features
        data_seq_features : list
            Features data_seqs have been projected onto.
        gen_seq_features : list
            Features gen_seqs have been projected onto.
        data_marginals : ndarray
            Marginals over the data sequences for each model feature.
        gen_marginals : ndarray
            Marginals over the generated sequences for each model feature.
        model_marginals : ndarray
            Marginals over the generated sequences, reweighted by the model,
            for each model feature.
        """
        if len(remove_features) > 0:
            indices_to_keep = [
                i for i, feature_lst in enumerate(self.features)
                if feature_lst not in remove_features and i not in remove_features
            ]
            self.features = self.features[indices_to_keep]
            self.update_model_structure(initialize=True)
            self.feature_dict = {tuple(f): i for i, f in enumerate(self.features)}

        if len(add_features) > 0:
            if len(self.features) == 0:
                self.features = np.array(add_features, dtype=object)
            else:
                self.features = np.append(self.features, add_features, axis=0)
            self.update_model_structure(initialize=True)
            self.feature_dict = {tuple(f): i for i, f in enumerate(self.features)}

        if len(add_data_seqs) > 0:
            logging.info('Adding data seqs.')
            try:
                if 'Paired' not in type(self).__name__:
                    add_data_seqs = filter_seqs(add_data_seqs, self.pgen_dir, **kwargs)
                else:
                    pass
                    #add_data_seqs = filter_seqs_paired(
                    #   add_data_seqs, self.pgen_dir_heavy, self.pgen_dir_light, **kwargs
                    #)
            except Exception as e:
                raise Exception(e)

            add_data_seqs = np.array(
                [[seq, '', ''] if isinstance(seq, str) else seq for seq in add_data_seqs]
            )
            if len(self.data_seqs) == 0: self.data_seqs = add_data_seqs
            else: self.data_seqs = np.concatenate([self.data_seqs, add_data_seqs])

        if len(add_gen_seqs) > 0:
            logging.info('Adding gen seqs.')
            try:
                if 'Paired' not in type(self).__name__:
                    add_gen_seqs = filter_seqs(add_gen_seqs, self.pgen_dir, **kwargs)
                else:
                    pass
            except Exception as e:
                raise Exception(e)
            add_gen_seqs=np.array(
                [[seq,'',''] if isinstance(seq, str) else seq for seq in add_gen_seqs]
            )
            if len(self.gen_seqs) == 0: self.gen_seqs = add_gen_seqs
            else: self.gen_seqs = np.concatenate([self.gen_seqs, add_gen_seqs])

        if ((len(add_data_seqs) + len(add_features) + len(remove_features) > 0
             or update_seq_features)
             and len(self.features) > 0 and len(self.data_seqs) > 0):
            logging.info('Encode data seqs.')
            self.data_encoding = self.encode_data(self.data_seqs)

        if ((len(add_data_seqs) + len(add_features) + len(remove_features) > 0
             or update_marginals) and len(self.features) > 0):
            if self.data_encoding.shape[0]:
                self.data_marginals = self.compute_marginals(
                    encoding=self.data_encoding, use_flat_distribution=True
                )

        if ((len(add_gen_seqs) + len(add_features) + len(remove_features) > 0
             or update_seq_features)
            and len(self.features) > 0 and len(self.gen_seqs) > 0):
            logging.info('Encode gen seqs.')
            self.gen_encoding = self.encode_data(self.gen_seqs)

        if ((len(add_gen_seqs) + len(add_features) + len(remove_features) > 0
             or update_marginals) and len(self.features) > 0):
            if self.gen_encoding.shape[0]:
                self.gen_marginals = self.compute_marginals(
                    encoding=self.gen_encoding, use_flat_distribution=True
                )
                self.model_marginals = self.compute_marginals(encoding=self.gen_encoding)

    def add_generated_seqs(
        self,
        num_gen_seqs,
        reset_gen_seqs: bool = True,
        add_error: bool = False,
        error_rate: Optional[int] = None,
    ) -> None:
        """Generates MonteCarlo sequences for gen_seqs using OLGA.
        Only generates seqs from a V(D)J model. Requires the OLGA package
        (pip install olga).
        Parameters
        ----------
        num_gen_seqs : int or float
            Number of MonteCarlo sequences to generate and add to the specified
            sequence pool.
        custom_model_folder : str
            Path to a folder specifying a custom IGoR formatted model to be
            used as a generative model. Folder must contain 'model_params.txt'
            and 'model_marginals.txt'
        add_error: bool
            simualate sequencing error: default is false
        error_rate: int
            set custom error rate for sequencing error.
            Default is the one inferred by igor.
        Attributes set
        --------------
        gen_seqs : list
            MonteCarlo sequences drawn from a VDJ recomb model
        gen_seq_features : list
            Features gen_seqs have been projected onto.
        """
        if hasattr(self, 'pgen_dir'):
            logging.info(
                f'Generating {num_gen_seqs} using the pgen model in {self.pgen_dir}.'
            )
        elif hasattr(self, 'pgen_dir_light'):
            logging.info(
                f'Generating {num_gen_seqs} using the light pgen model in '
                f'{self.pgen_dir_light} and the heavy pgen model in '
                f'{self.pgen_dir_heavy}.'
            )

        seqs = self.generate_sequences_pre(
            num_gen_seqs, nucleotide=False, add_error=add_error
        )
        if reset_gen_seqs: self.gen_seqs = []
        self.update_model(add_gen_seqs=seqs)

    def save_model(
        self,
        save_dir: str,
        save_data_seqs: bool = False,
        save_gen_seqs: bool = False,
        force: bool = True
    ) -> None:
        """Saves model parameters and sequences
        Parameters
        ----------
        save_dir : str
            Directory name to save model attributes to.
        attributes_to_save: list
            name of attributes to save
        """
        if os.path.isdir(save_dir):
            if not force:
                if not input(f'The directory {save_dir} already exists. '
                             'Overwrite existing model (y/n)? ').strip().lower() in ['y', 'yes']:
                    print('Exiting...')
                    return
        else:
            os.mkdir(save_dir)

        if save_data_seqs:
            with open(os.path.join(save_dir, 'data_seqs.tsv'), 'w') as data_seqs_file:
                data_seq_energies = self.compute_energy(self.data_encoding)
                data_seq_features = self.encoding_to_feature_strs(self.data_encoding)
                data_seqs_file.write('Sequence;Genes\tLog(Q)\tFeatures\n')
                data_seqs_file.write(
                    '\n'.join(
                        [';'.join(seq) + '\t'
                         + str(-data_seq_energies[i] - np.log(self.Z)) + '\t'
                         + ';'.join(
                             [','.join(features) for features in data_seq_features[i]]
                         )
                         for i, seq in enumerate(self.data_seqs)]
                    )
                )

        if save_gen_seqs:
            with open(os.path.join(save_dir, 'gen_seqs.tsv'), 'w') as gen_seqs_file:
                gen_seq_energies = self.compute_energy(self.gen_encoding)
                gen_seq_features = self.encoding_to_feature_strs(self.gen_encoding)
                gen_seqs_file.write('Sequence;Genes\tLog(Q)\tFeatures\n')
                gen_seqs_file.write(
                    '\n'.join(
                        [';'.join(seq) + '\t'
                         +  str(-gen_seq_energies[i] - np.log(self.Z)) + '\t'
                         + ';'.join(
                             [','.join(features) for features in gen_seq_features[i]]
                         )
                         for i, seq in enumerate(self.gen_seqs)
                        ]
                    )
                )

        with open(os.path.join(save_dir, 'log.txt'), 'w') as L1_file:
            L1_file.write('Z ='+str(self.Z)+'\n')
            L1_file.write('norm_productive ='+str(self.norm_productive)+'\n')
            L1_file.write('min_energy_clip ='+str(self.min_energy_clip)+'\n')
            L1_file.write('max_energy_clip ='+str(self.max_energy_clip)+'\n')
            L1_file.write('likelihood_train,likelihood_test\n')
            for llh_train, llh_test in zip(self.likelihood_train, self.likelihood_test):
                L1_file.write(f'{llh_train},{llh_test}\n')

        if 'Sonia' in type(self).__name__:
            energies = self.model.get_weights()[0].ravel()
            with open(os.path.join(save_dir, 'features.tsv'), 'w') as feature_file:
                feature_file.write('Feature,energy,marginal_data,marginal_model,marginal_gen\n')
                for i, _ in enumerate(self.features):
                    feature_file.write(
                        ';'.join(self.features[i])
                        + ',' + str(energies[i])
                        + ',' + str(self.data_marginals[i])
                        + ',' + str(self.model_marginals[i])
                        + ',' + str(self.gen_marginals[i])
                        + '\n'
                    )
        else:
            with open(os.path.join(save_dir, 'features.tsv'), 'w') as feature_file:
                feature_file.write('Feature,marginal_data,marginal_model,marginal_gen\n')
                for i, _ in enumerate(self.features):
                    feature_file.write(
                        ';'.join(self.features[i])
                        + ',' + str(self.data_marginals[i])
                        + ',' + str(self.model_marginals[i])
                        + ',' + str(self.gen_marginals[i])
                        + '\n'
                    )

        self.model.save(os.path.join(save_dir, 'model.h5'))
        self._save_pgen_model(save_dir)

    def _save_pgen_model(
        self,
        save_dir: str
    ) -> None:
        import shutil
        shutil.copy2(os.path.join(self.pgen_dir, 'model_params.txt'), save_dir)
        shutil.copy2(os.path.join(self.pgen_dir, 'model_marginals.txt'), save_dir)
        shutil.copy2(os.path.join(self.pgen_dir, 'V_gene_CDR3_anchors.csv'), save_dir)
        shutil.copy2(os.path.join(self.pgen_dir, 'J_gene_CDR3_anchors.csv'), save_dir)

    def load_model(
        self,
        ppost_model: str,
        load_seqs: bool = True,
        verbose: bool = True
    ) -> None:
        """Loads model from directory.
        Parameters
        ----------
        load_dir : str
            Directory name to load model attributes from.
        """
        paired = 'Paired' in type(self).__name__
        self.ppost_dir = get_model_dir(ppost_model, paired)

        ppost_files = ('features.tsv', 'log.txt')

        if 'NN' in type(self).__name__:
            ppost_files += ('model.h5', )

        files_in_dir = set(os.listdir(self.ppost_dir))
        missing_files = set(ppost_files) - files_in_dir

        if len(missing_files) > 0:
            missing_files = f'{missing_files}'[1:-1]
            if 'model.h5' in missing_files:
                if 'Paired' in type(self).__name__:
                    pair_msg = 'Paired'
                else:
                    pair_msg = ''
                raise RuntimeError('The model cannot be loaded. The following files '
                                   f'are missing: {missing_files}. Should a Sonia{pair_msg} '
                                   'model be initialized instead?')
            else:
                raise RuntimeError('The model cannot be loaded. The following files '
                                   f'are missing: {missing_files}.')

        feature_file = os.path.join(self.ppost_dir, 'features.tsv')
        model_file = os.path.join(self.ppost_dir, 'model.h5')
        data_seq_file = os.path.join(self.ppost_dir, 'data_seqs.tsv')
        gen_seq_file = os.path.join(self.ppost_dir, 'gen_seqs.tsv')
        log_file = os.path.join(self.ppost_dir, 'log.txt')

        with open(log_file, 'r') as L1_file:
            self.L1_converge_history = []

            # Zeroth line is Z.
            self.Z = float(next(L1_file).strip().partition('=')[-1])

            # First line is productive norm.
            if self.recompute_productive_norm:
                next(L1_file)
            else:
                self.norm_productive = float(next(L1_file).strip().partition('=')[-1])

            # Second line is minimum energy clip.
            try:
                self.min_energy_clip=float(next(L1_file).strip().partition('=')[-1])
            except:
                pass

            # Third line is maximum energy clip.
            try:
                self.max_energy_clip=float(next(L1_file).strip().partition('=')[-1])
            except:
                pass

            for line in L1_file:
                line = line.strip()
                if len(line) == 0:
                    continue
                try:
                    train_val, _, test_val = line.partition(',')
                    self.likelihood_train.append(float(train_val))
                    self.likelihood_test.append(float(test_val))
                except:
                    continue

        self._load_features_and_model(feature_file, model_file, verbose)

        if not load_seqs:
            return

        def seq_loader(
            infile,
            seqs,
        ) -> sparse.csr_array:
            indices = []
            indptr = [0]
            with open(infile, 'r') as fin:
                next(fin)

                for line in fin:
                    split_line = line.split('\t')
                    seqs.append(split_line[0].split(';'))
                    features = split_line[2].strip().split(';')
                    specified_features = []
                    for feature in features:
                        if ',' in feature:
                            feature = tuple(feature.split(','))
                        else:
                            feature = (feature,)
                        if feature in self.feature_dict:
                            specified_features.append(self.feature_dict[feature])
                    indices += specified_features
                    indptr.append(len(specified_features) + indptr[-1])
            data = np.ones(len(indices), dtype=np.int8)
            return sparse.csr_array(
                (data, indices, indptr),
                shape=(len(indptr) - 1, len(self.features))
            )

        if os.path.isfile(data_seq_file):
            self.data_seqs = []
            self.data_encoding = seq_loader(data_seq_file, self.data_seqs)
        elif verbose:
            logging.info('Cannot find data_seqs.tsv  --  no data seqs loaded.')

        if os.path.isfile(gen_seq_file):
            self.gen_seqs = []
            self.gen_encoding = seq_loader(gen_seq_file, self.gen_seqs)
        elif verbose:
            logging.info('Cannot find gen_seqs.tsv  --  no generated seqs loaded.')

    def _load_features_and_model(
        self,
        feature_file: str,
        model_file: str,
        verbose: bool = True
    ) -> None:
        """
        Loads features and model.

        This is set as an internal function to allow daughter classes to load
        models from saved feature energies directly.
        """
        features = []
        data_marginals = []
        model_marginals = []
        gen_marginals = []
        energies = []

        with open(feature_file, 'r') as features_file:
            column_names = next(features_file)
            sonia_or_sonnia = column_names.split(',')[1]
            if sonia_or_sonnia == 'marginal_data': k = 0
            else: k = 1

            for line in features_file:
                splitted = line.strip().split(',')
                features.append(splitted[0].split(';'))
                data_marginals.append(float(splitted[1 + k]))
                model_marginals.append(float(splitted[2 + k]))
                gen_marginals.append(float(splitted[3 + k]))

                if k == 1:
                    energies.append(float(splitted[1]))

        self.features = np.array(features, dtype=object)
        self.data_marginals = np.array(data_marginals)
        self.gen_marginals = np.array(gen_marginals)
        self.model_marginals = np.array(model_marginals)
        self.feature_dict = {tuple(f): i for i, f in enumerate(self.features)}

        if k == 1:
            feature_energies = np.array(energies).reshape(len(self.features), 1)
            self.update_model_structure(initialize=True)
            self.model.set_weights([feature_energies])
            self.model_params = self.model.get_weights()

        else:
            try:
                self.model = keras.models.load_model(
                    model_file,
                    custom_objects={
                        'loss': self._loss, 'likelihood': self._likelihood,
                        'clip': ko.clip,
                    },
                    compile=False
                )
            except Exception as e:
                if 'Unknown layer' in str(e):
                    paired_str = 'Paired' if 'Paired' in type(self).__name__ else ''
                    raise RuntimeError('The loaded model structure is supposed to be for '
                                       f'a SoNNia{paired_str} model, but a Sonia{paired_str} '
                                       'model is trying to be initialized. Try loading '
                                       f'the model using the SoNNia{paired_str} class instead.')
                else:
                    raise e

            if len(self.model.layers) > 3:
                paired_str = 'Paired' if 'Paired' in type(self).__name__ else ''
                raise RuntimeError('The loaded model structure is supposed to be '
                                   f'for a SoNNia{paired_str} model, but a Sonia{paired_str} '
                                   'model is trying to be initialized. Try loading '
                                   f'the model using the SoNNia{paired_str} class instead.')

            self.optimizer = keras.optimizers.RMSprop()
            self.model.compile(
                optimizer=self.optimizer, loss=self._loss,metrics=[self._likelihood]
            )

    def load_pgen_model(
        self
    ) -> None:
        '''
        load olga model.
        '''
        #Load generative model
        (self.genomic_data, self.generative_model,
         self.pgen_model, self.seqgen_model, self.norm_productive,
         self.pgen_dir) = define_pgen_model(
             self.pgen_model, self.recompute_productive_norm, return_pgen_dir=True
         )

        with open(os.path.join(self.pgen_dir, 'model_params.txt'), 'r') as fin:
            sep = 0
            error_rate = ''
            lines = fin.read().splitlines()
            while len(error_rate) < 1:
                error_rate = lines[-1 + sep]
                sep -= 1
            self.error_rate = float(error_rate)

    def generate_sequences_pre(
        self,
        num_seqs: int,
        nucleotide: bool = False,
        error_rate: Optional[int] = None,
        add_error: bool = False,
        seed: Optional[int | np.random.Generator | np.random.BitGenerator | np.random.SeedSequence] = None,
    ) -> NDArray[str]:
        """Generates MonteCarlo sequences for gen_seqs using OLGA in parallel.
        Only generates seqs from a V(D)J model. Requires the OLGA package
        (pip install olga). If you add error_rate, only the aminoacid sequence is modified.
        Parameters
        ----------
        num_seqs : int or float
            Number of MonteCarlo sequences to generate and add to the specified
            sequence pool.

        Returns
        --------------
        seqs : list
            MonteCarlo sequences drawn from a VDJ recomb model
        """
        from sonnia.utils import add_random_error, generate_sequence
        from olga.utils import nt2aa

        if error_rate is None: error_rate = self.error_rate
        else: error_rate = error_rate

        if seed is None:
            rng = self.rng
        else:
            rng = np.random.default_rng(seed)

        if num_seqs > 20000:
            seeds = rng.integers(low=0, high=2**32 - 1, size=num_seqs)
            zipped = zip(itertools.repeat(self.seqgen_model, num_seqs),
                         itertools.repeat(self.genomic_data, num_seqs),
                         seeds,
                         itertools.repeat(add_error, num_seqs),
                         itertools.repeat(error_rate, num_seqs))

            with mp.Pool(processes=self.processes) as pool:
                seqs = pool.starmap(generate_sequence, zipped)
        else:
            seqs = []
            tqdm_desc = 'Generating sequences'
            for i in tqdm(range(int(num_seqs)), position=0, desc=tqdm_desc):
                np.random.seed(rng.integers(0, 2**32 - 1))
                seq = self.seqgen_model.gen_rnd_prod_CDR3(conserved_J_residues='ABCEDFGHIJKLMNOPQRSTUVWXYZ')

                if add_error:
                    err_seq = add_random_error(seq[0], self.error_rate)
                    seq = [err_seq, nt2aa(err_seq), seq[2], seq[3]]

                seq = [seq[1],
                       self.genomic_data.genV[seq[2]][0].partition('*')[0],
                       self.genomic_data.genJ[seq[3]][0].partition('*')[0],
                       seq[0]]

                seqs.append(seq)

        seqs = np.array(seqs)
        if nucleotide:
            return seqs
        else:
            return seqs[:, :-1]

#        if seed is None:
#            seed = self.rng
#
#        seqs = rg.generate_pgen_seqs(
#            self.pgen_dir, num_seqs, seed, processes=self.processes)
#
#        if nucleotide:
#            return seqs
#        return seqs[:, :-1]

    def generate_sequences_post(
        self,
        num_seqs: int = 1,
        upper_bound: float = 10,
        nucleotide: bool = False,
        seed: Optional[int | np.random.Generator | np.random.BitGenerator | np.random.SeedSequence] = None,
        max_chunksize: int = int(2e6),
    ) -> NDArray[str]:
        """
        Generate Monte Carlo sequences from Sonia through rejection sampling.

        Parameters
        ----------
        num_seqs : int, deafult 1
            Number of Monte Carlo sequences to generate and add to the specified
            sequence pool.
        upper_bound : float, default 10
            The value of Q at above which all generated sequences are accepted.
        nucleotide : bool, default False
            Return the CDR3 nulceotide sequences in addition to the CDR3 amino
            acid sequence, V gene, and J gene.
        seed : int or numpy.random.Generator or numpy.random.BitGenerator or numpy.random.SeedSequence, optional
            The seed for random number generation.
        max_chunksize : int, default int(2e6)
            The maximum chunksize for generating sequences. The default is to
            generate int(1.1 * upper_bound * num_seqs) sequences and then perform
            rejection sampling. However, if num_seqs is very large, this can ensue
            in high memory costs. The minimum between max_chunksize and the
            aforementioned default amount is used for chunking.

        Returns
        -------
        seqs : numpy.ndarray of str
            Monte Carlo sequences drawn from a VDJ recomb model that pass selection.
        """
        if seed is None:
            rng = self.rng
        else:
            rng = np.random.default_rng(seed)

        seqs = []

        chunksize = min(
            max_chunksize, int(upper_bound * num_seqs * 1.1)
        )

        while len(seqs) < num_seqs + 1:
            # generate sequences from pre
            seqs_gen = self.generate_sequences_pre(
                num_seqs=chunksize, nucleotide=nucleotide, seed=rng
            )

            # compute features and energies 
            encoding = self.encode_data(seqs_gen)
            energies = self.compute_energy(encoding)

            #do rejection
            rejection_selection = self.rejection_sampling(
                energies, upper_bound, rng
            )

            seqs_post = seqs_gen[rejection_selection]
            if len(seqs) == 0 and len(seqs_post) > 0:
                seqs = seqs_post
            elif len(seqs) > 0:
                seqs = np.concatenate((seqs, seqs_post), axis=0)
        return seqs[:num_seqs]

    def rejection_sampling(
        self,
        energies: NDArray[np.float32],
        upper_bound: float = 10,
        seed: Optional[int | np.random.Generator | np.random.BitGenerator | np.random.SeedSequence] = None,
    ) -> NDArray[bool]:
        """
        Return whether sequences would pass selection using rejection sampling.

        Parameters
        ----------
        energies : numpy.ndarray of numpy.float32
            The energies computed by the model.
        upper_bound : float, default 10
            The value of Q at above which all generated sequences are accepted.
        seed : int or numpy.random.Generator or numpy.random.BitGenerator or numpy.random.SeedSequence, optional
            The seed for random number generation.

        Returns
        -------
        numpy.ndarray of bool
            An array of whether the sequence associated with that energy passed
            selection.
        """
        if seed is None:
            rng = self.rng
        else:
            rng = np.random.default_rng(seed)

        q = np.exp(-energies) / self.Z
        random_samples = rng.uniform(size=len(energies))

        return random_samples < q / upper_bound

    def evaluate_seqs(
        self,
        seqs: Sequence[Sequence[str]] = [],
        include_genes: bool = True
    ) -> Tuple[NDArray[np.float32] | NDArray[np.float64]]:
        """
        Return the selection factors, Pgen, and Ppost of sequences.

        Parameters
        ----------
        seqs : sequence of sequence of str
            Each subsequence is the CDR3 amino acid sequence, V gene, and J gene.

        Returns
        -------
        qs : numpy.ndarray of numpy.float32
            The normalized selection factors computed by the model associated
            with each sequence.
        pgens : numpy.ndarray of numpy.float64
            The probabilities of generating the sequences normalized by the
            probability of generating a productive sequence.
        pposts : numpy.ndarray of numpy.float64
            The product of the normalized selection factors and productive-normalized
            probabilities of generating the given sequences.
        """
        encoding = self.encode_data(seqs)
        energies = self.compute_energy(encoding)
        qs = np.exp(-energies) / self.Z
        pgens = self.compute_all_pgens(seqs, include_genes) / self.norm_productive
        pposts = pgens * qs

        return qs, pgens, pposts

    def evaluate_selection_factors(
        self,
        seqs: Sequence[Sequence[str]] = []
    ) -> NDArray[np.float32]:
        """
        Return the normalized selection factors.

        Parameters
        ----------
        seqs : sequence of sequence of str
            Each subsequence is the CDR3 amino acid sequence, V gene, and J gene.

        Returns
        -------
        numpy.ndarray of numpy.float32
            The normalized selection factors computed by the model associated
            with each sequence.
        """
        encoding = self.encode_data(seqs)
        energies = self.compute_energy(encoding)
        return np.exp(-energies) / self.Z

    def joint_marginals(
        self,
        encoding: Optional[sparse.csr_array] = None,
        seqs: Optional[Sequence[Sequence[str]]] = None,
        features: Optional[Sequence[Tuple[str]]] = None,
        use_flat_distribution: bool = False,
    ) -> None:
        '''Returns joint marginals P(i,j) with i and j features of sonia (l3, aA6, etc..), index of features attribute is preserved.
           Matrix is lower-triangular.
        Parameters
        ----------
        features: list
            custom feature list
        seq_model_features: list
            encoded sequences
        seqs: list
            seqs to encode.
        use_flat_distribution: bool
            for data and generated seqs is True, for model is False (weights with Q)
        Returns
        -------
        joint_marginals: array
            matrix (i,j) of joint marginals
        '''

        if encoding is None and seqs is None:
            raise RuntimeError('Both encoding and seqs cannot be None.')
        if encoding is not None and seqs is not None:
            raise RuntimeError('Both encoding and seqs cannot be given. '
                             'One of them must be None.')
        if encoding is not None and features is not None:
            raise RuntimeError('Both encoding and features cannot be given. '
                               'If features is given, seqs must be given to redo '
                               'the one-hot encoding.')
        if features is not None:
            num_features = len(features)
        else:
            num_features = len(self.features)
            features = self.features

        if seqs is not None:
            encoding = self.encode_data(seqs, features)
        seq_model_features = self.encoding_to_feature_idxs(encoding)

        l = len(features)
        two_points_marginals = np.zeros((l,l))
        n = len(seq_model_features)
        procs = mp.cpu_count()
        sizeSegment = int(n / procs)

        if not use_flat_distribution:
            energies = self.compute_energy(encoding)
            Qs = np.exp(-energies)
        else:
            Qs = np.ones(encoding.shape[0])

        # Overhead of parallel is too long for small amount of sequences.
        if len(seq_model_features) < int(1e5):
            two_points_marginals, Z = partial_joint_marginals(
                (seq_model_features, Qs, np.zeros((l, l)))
            )
            return two_points_marginals / Z

        # Create size segments list
        jobs = []
        for i in range(0, procs-1):
            jobs.append([seq_model_features[i*sizeSegment:(i+1)*sizeSegment],Qs[i*sizeSegment:(i+1)*sizeSegment],np.zeros((l,l))])
        p=mp.Pool(procs)
        pool = p.map(partial_joint_marginals, jobs)
        p.close()
        Z=0
        two_points_marginals=0
        for m,z in pool:
            Z+=z
            two_points_marginals+=m
        return two_points_marginals/Z

    def joint_marginals_independent(
        self,
        marginals: np.ndarray
    ) -> np.ndarray:
        '''Returns independent joint marginals P(i,j)=P(i)*P(j) with i and j features of sonia (l3, aA6, etc..), index of features attribute is preserved.
        Matrix is lower-triangular.
        Parameters
        ----------
        marginals: list
            marginals.
        Returns
        -------
        joint_marginals: array
            matrix (i,j) of joint marginals
        '''
        joint_marginals = np.outer(marginals, marginals)

        # Return the lower triangle of the matrix with zeros along the diagonal,
        # consistent with previous versions of the software.
        joint_marginals = np.tril(joint_marginals)
        np.fill_diagonal(joint_marginals, 0)
        return joint_marginals

    def compute_joint_marginals(
        self
    ) -> None:
        '''Computes joint marginals for all.
        Attributes Set
        -------
        gen_marginals_two: array
            matrix (i,j) of joint marginals for pre-selection distribution
        data_marginals_two: array
            matrix (i,j) of joint marginals for data
        model_marginals_two: array
            matrix (i,j) of joint marginals for post-selection distribution
        gen_marginals_two_independent: array
            matrix (i,j) of independent joint marginals for pre-selection distribution
        data_marginals_two_independent: array
            matrix (i,j) of joint marginals for pre-selection distribution
        model_marginals_two_independent: array
            matrix (i,j) of joint marginals for pre-selection distribution
        '''

        self.gen_marginals_two = self.joint_marginals(
            encoding=self.gen_encoding,
            use_flat_distribution=True
        )
        self.data_marginals_two = self.joint_marginals(
            encoding=self.data_encoding,
            use_flat_distribution=True
        )
        self.model_marginals_two = self.joint_marginals(encoding=self.gen_encoding)
        self.gen_marginals_two_independent = self.joint_marginals_independent(self.gen_marginals)
        self.data_marginals_two_independent = self.joint_marginals_independent(self.data_marginals)
        self.model_marginals_two_independent = self.joint_marginals_independent(self.model_marginals)

    def compute_all_pgens(
        self,
        seqs: Sequence[Sequence[str]],
        include_genes: bool = True
    ) -> np.ndarray:
        '''Compute Pgen of sequences using OLGA in parallel
        Parameters
        ----------
        seqs: list
            list of sequences to evaluate.
        Returns
        -------
        pgens: array
            generation probabilities of the sequences.
        '''
        if include_genes:
            with mp.Pool(processes=self.processes) as pool:
                f = pool.map(compute_pgen_expand, zip(seqs, itertools.repeat(self.pgen_model)))
            return np.array(f)

        with mp.Pool(processes=self.processes) as pool:
            f = pool.map(compute_pgen_expand_novj, zip(seqs, itertools.repeat(self.pgen_model)))
        return np.array(f)

    def entropy(
        self,
        n: int = int(2e4),
        include_genes: bool = True,
        remove_zeros: bool = True
    ) -> float:
        '''Compute Entropy of Model
        Returns
        -------
        entropy: float
            entropy of the model
        '''
        if self.gen_encoding.shape[0] >= int(1e4):
            encoding = self.gen_encoding[:n]
            seqs= self.gen_seqs[:n]
        else:
            raise RuntimeError('At least 10,000 generated sequences must be used '
                               f'for estimating entropy. Only {self.gen_encoding.shape[0]} '
                              'generated sequences are present.')

        energies = self.compute_energy(encoding) # compute energies
        self.gen_Q = np.exp(-energies) / self.Z # compute Q
        self.gen_pgen = self.compute_all_pgens(seqs, include_genes) / self.norm_productive # compute pgen
        sel = self.gen_pgen > 0
        num_zero_pgen = len(sel) - np.count_nonzero(sel)
        if num_zero_pgen > 0:
            logging.info(f'{num_zero_pgen} sequences have zero Pgen, we remove '
                         'them in the evaluation of the entropy')
        self.gen_ppost = self.gen_pgen * self.gen_Q # compute ppost
        self._entropy = -np.mean(self.gen_Q[sel] * np.log2(self.gen_ppost[sel]))
        return self._entropy

    def dkl_post_gen(
        self,
        n: int = int(1e5)
    ) -> float:
        '''Compute D_KL(P_post|P_gen)
        Returns
        -------
        dkl: float
            D_KL(P_post|P_gen)
        '''
        if hasattr(self, 'energies_gen'): # energies gen exist
            if len(self.energies_gen) < 1e4:
                raise RuntimeError(
                    'At least 10,000 generated sequences must be used for estimating '
                    f'DKL(post || gen). Only {len(self.energies_gen)} were used.'
                )
            Q = np.exp(-self.energies_gen) / self.Z
            self.dkl = np.mean(Q * np.log2(Q))
            return self.dkl

        if self.gen_encoding.shape[0] >= int(1e4):
            encoding = self.gen_encoding[:n]
        else:
            raise RuntimeError(
                'At least 10,000 generated sequences must be used for estimating '
                f'entropy. Only {self.gen_encoding.shape[0]} are present.'
            )
        energies = self.compute_energy(encoding) # compute energies
        Q = np.exp(-energies) / self.Z # compute Q
        self.dkl = np.mean(Q * np.log2(Q))
        return self.dkl
