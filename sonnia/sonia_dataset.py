#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2024 Montague, Zachary
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""Script containing the SoniaDataset class for loading data into mini-batches.
"""
from typing import Any, Callable, Dict, List, Optional, Tuple

import keras
import keras.ops as ko
import numpy as np
from numpy.typing import NDArray
import scipy.sparse as sparse

class SoniaDataset(keras.utils.PyDataset):
    """
    A Dataset class for ensuring both gen seqs and data seqs appear in a mini-batch.
    Options allow there to be the original imbalance as well as sampling schemes for
    balancing the two datasets in mini-batches using over- and undersampling.

    Attributes
    ----------
    x : numpy.ndarray of numpy.int8 or scipy.sparse.csr_array
        The dense or sparse one-hot feature encoding.
    y : numpy.ndarray of numpy.int8
        The labels for whether the feature comes from data (0) or gen (1).
    batch_size : int
        How big a mini-batch is.
    shuffle : bool
        Whether the features and labels should be shuffled after each epoch.
    rng : numpy.random.Generator
        A random number generator.
    split_encoding : callable
        A function for SoNNia models for splitting the encoding into separate
        length, amino acid, and gene feature arrays.
    sparse_input : bool
        If the encoding of features is a scipy.sparse.csr_array.
    where_class_0 : numpy.ndarray of numpy.int32
        The indices of data features in x.
    where_class_1 : numpy.ndarray of numpy.int32
        The indices of gen features in x.
    class_0_size : int
        The amount of data seqs.
    class_1_size : int
        The amount of gen seqs.
    bigger_size : int
        The size of the majority class.
    smaller_size : int
        The size of the minority class.
    batch_constraint : int
        The constraint on the number of mini-batches.
    sampling : str
        The type of sampling using to construct mini-batches.
    class_0_batch : int
        The number of class 0 datapoints when using imbalanced sampling.
    class_1_batch : int
        The number of class 1 datapoints when using imbalanced sampling.

    Methods
    -------
    __getitem__(index)
        Return the mini-batch indexed by index.
    __len__()
        Return the number of mini-batches in an epoch.
    on_epoch_end_oversample()
        Update the indices for the next epoch when oversampling the minority class.
    on_epoch_end_undersample()
        Update the indices for the next epoch when undersampling the majority class.
    on_epoch_end()
        How to update the indices (determined by the sampling scheme).
    """
    def __init__(
        self,
        x: NDArray[np.int8],
        y: NDArray[np.int8],
        sampling: str,
        batch_size: int = 512,
        shuffle: bool = True,
        seed: Optional[int | np.random.Generator | np.random.BitGenerator | np.random.SeedSequence] = None,
        split_encoding: Optional[Callable] = None,
        **kwargs: Dict[str, Any],
    ) -> None:
        """
        Initialize a SoniaDataset object.

        Parameters
        ----------
        x : numpy.ndarray of numpy.int8
            The one-hot encoded sequence features.
        y : numpy.ndarray of numpy.int8
            The labels of the data.
        sampling : str
            Options are 'undersample' (undersample the majority class each epoch).
            'oversample' (oversample the minority class each epoch), or 'unbalanced'
            (use the original proportion of data and gen but ensure that both data
            and gen appear in each mini-batch).
        batch_size : int, default 512
            The size of the mini-batch.
        shuffle : bool, default True
            After every epoch, reshuffle the data.
        seed : int or np.random.Generator or np.random.BitGenerator or np.random.SeedSequence, optional
            Sets random seed.
        **kwargs
            Keyword arguments to keras.utils.PyDataset.
        """
        super().__init__(**kwargs)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.rng = np.random.default_rng(seed)
        self.split_encoding = split_encoding

        self.sparse_input = isinstance(x, sparse.csr_array)

        self.x = x
        self.y = y

        self.where_class_0 = ko.where(y == 0)[0].numpy()
        self.where_class_1 = ko.where(y == 1)[0].numpy()

        self.class_0_size = self.where_class_0.shape[0]
        if self.class_0_size == 0:
            raise RuntimeError('No data seqs loaded into the SoniaDataset.')

        self.class_1_size = self.where_class_1.shape[0]
        if self.class_1_size == 0:
            raise RuntimeError('No gen seqs loaded into the SoniaDataset.')

        if self.class_0_size > self.class_1_size:
            self.bigger_size = self.class_0_size
            self.smaller_size = self.class_1_size
            self.majority_class = 0
        else:
            self.bigger_size = self.class_1_size
            self.smaller_size = self.class_0_size
            self.majority_class = 1

        self.sampling = sampling
        if self.sampling == 'undersample':
            self.batch_constraint = self.smaller_size * 2
            self.on_epoch_end = self.on_epoch_end_shuffle
        elif self.sampling == 'oversample':
            self.batch_constraint = self.bigger_size * 2
            self.on_epoch_end = self.on_epoch_end_oversample
        elif self.sampling == 'unbalanced':
            self.batch_constraint = self.class_0_size + self.class_1_size
            self.class_0_batch = int(
                self.class_0_size / (self.class_0_size + self.class_1_size) * self.batch_size
            )

            if self.class_0_batch == 0:
                raise RuntimeError(
                    'The classes are too imbalanced. Not every batch will contain '
                    'a data sequence. One possible way to mitigate this is increasing '
                    'the batch_size.'
                )


            self.class_1_batch = int(
                self.class_1_size / (self.class_0_size + self.class_1_size) * self.batch_size
            )
            if self.class_1_batch == 0:
                raise RuntimeError(
                    'The classes are too imbalanced. Not every batch will contain '
                    'a gen sequence. One possible way to mitigate this is increasing '
                    'the batch_size.'
                )

            self.on_epoch_end = self.on_epoch_end_shuffle
        else:
            raise ValueError(
                'sampling must be \'undersample\', \'oversample\', or \'unbalanced\'.'
            )

        self.on_epoch_end()

    def __getitem__(
        self,
        index: int
    ) -> Tuple[NDArray[np.int8] | List[NDArray[np.int8]], NDArray[np.int8]]:
        """
        Return a mini-batch which is guaranteed to include both gen and data seqs.

        For the undersample stratgey, there are batches indexed beyond the object's
        length which contain only one class. In practice, keras does not index beyond
        the SoniaDataset object's length, so this isn't a problem at the moment for
        ensuring both data and gen seqs appear in a mini-batch.

        Parameters
        ----------
        index : int
            The index of the mini-batch for an epoch.

        Returns
        -------
        x : numpy.ndarray of numpy.int8 or list of numpy.ndarray of numpy.int8
            The features of class 0 and class 1 data.
        y : numpy.ndarray of numpy.int8
            The labels denoting which features come from class 0 or class 1 data.
        """
        if self.sampling != 'unbalanced':
            start_idx = index * self.batch_size // 2
            end_idx = start_idx + self.batch_size // 2
            indices_0 = self.class_0_indices[start_idx:end_idx]
            indices_1 = self.class_1_indices[start_idx:end_idx]
        else:
            start_idx_0 = index * self.class_0_batch
            end_idx_0 = start_idx_0 + self.class_0_batch
            start_idx_1 = index * self.class_1_batch
            end_idx_1 = start_idx_1 + self.class_1_batch
            indices_0 = self.class_0_indices[start_idx_0:end_idx_0]
            indices_1 = self.class_1_indices[start_idx_1:end_idx_1]

        # It is faster to concatenate indices than it is to concatenate arrays
        # of large two-dimensional arrays of features.
        batch_indices = np.concatenate((
            self.where_class_0[indices_0], self.where_class_1[indices_1]
        ))

        if self.sparse_input:
            x = self.x[batch_indices].toarray()
        else:
            x = self.x[batch_indices]

        y = self.y[batch_indices]

        if self.split_encoding is not None:
            x = self.split_encoding(x)
        return x, y

    def __len__(
        self
    ) -> int:
        """
        Return the number of mini-batches per epoch.

        Parameters
        ----------
        None

        Returns
        -------
        int
            The number of mini-batches per epoch. If undersampling the majority
            class, the number of mini-batches is set by the minority class, and
            vice versa for oversampling. If unbalanced sampling, the number of
            mini-batches is set by the total number of data and gen seqs.
        """
        return int(np.ceil(self.batch_constraint / self.batch_size))

    def on_epoch_end_oversample(
        self
    ) -> None:
        """
        Update the indices for the next epoch.

        For oversampling, the minority class is sampled with replacement each epoch
        to match the size of the majority class. Shuffling affects only the indices
        of the majority class.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        if self.majority_class == 1:
            self.class_0_indices = self.rng.choice(
                self.class_0_size, size=self.class_1_size
            )
            self.class_1_indices = np.arange(self.class_1_size)
            if self.shuffle == True:
                self.rng.shuffle(self.class_1_indices)
        else:
            self.class_1_indices = self.rng.choice(
                self.class_1_size, size=self.class_0_size
            )
            self.class_0_indices = np.arange(self.class_0_size)
            if self.shuffle == True:
                self.rng.shuffle(self.class_0_indices)

    def on_epoch_end_shuffle(
        self
    ) -> None:
        """
        Update the indices for the next epoch.

        Shuffling affects both minority and majority classes.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        self.class_0_indices = np.arange(self.class_0_size)
        self.class_1_indices = np.arange(self.class_1_size)
        if self.shuffle == True:
            self.rng.shuffle(self.class_0_indices)
            self.rng.shuffle(self.class_1_indices)
