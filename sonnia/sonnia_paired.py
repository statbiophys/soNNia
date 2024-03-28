#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
@author: Giulio Isacchini
"""
from copy import copy
import logging
import multiprocessing as mp
import os
from typing import Any, Dict, Iterable, List, Optional, Tuple

logging.getLogger('tensorflow').disabled = True

import numpy as np
from tensorflow import keras
import tensorflow.keras.backend as K
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.models import load_model as lm

from sonnia.sonia import GENE_FEATURE_OPTIONS
from sonnia.sonia_paired import SoniaPaired

#Set input = raw_input for python 2
try:
    import __builtin__
    input = getattr(__builtin__, 'raw_input')
except (ImportError, AttributeError):
    pass

class SoNNiaPaired(SoniaPaired):
    def __init__(self,
                 *args: Tuple[Any],
                 gene_features: str = 'indep_vj',
                 across_chain_features: Optional[Iterable[str]] = None,
                 independent_chains: bool = False,
                 min_energy_clip: int = -10,
                 max_energy_clip: int = 10,
                 deep: bool = True,
                 l2_reg: float = 1e-3,
                 **kwargs: Dict[str, Any]
                ) -> None:
        invalid_gene_features = {'vjl', 'none'}
        if deep:
            if gene_features in invalid_gene_features:
                valid_gene_features = f'{GENE_FEATURE_OPTIONS - invalid_gene_features}'[1:-1]
                invalid_gene_features = f'{invalid_gene_features}'[1:-1]
                raise ValueError(f'gene_features = \'{gene_features}\' is an invalid option '
                                 'when using a deep SoNNiaPaired model. Use one of the '
                                 f'following instead: {valid_gene_features}.')
            if across_chain_features is not None:
                raise RuntimeError('across_chain_features must be None '
                                   'when using a deep SonniaPaired model.')

        self.deep = deep
        self.independent_chains = independent_chains
        SoniaPaired.__init__(self, *args, gene_features=gene_features,
                             across_chain_features=across_chain_features,
                             min_energy_clip=min_energy_clip,
                             max_energy_clip=max_energy_clip,
                             l2_reg=l2_reg, **kwargs)

    def update_model_structure(self,
                               output_layer: List = [],
                               input_layer: List = [],
                               initialize: bool = False
                              ) -> bool:
        """ Defines the model structure and compiles it.

        Parameters
        ----------
        structure : Sequential Model Keras
            structure of the model
        initialize: bool
            if True, it initializes to linear model, otherwise it updates to new structure

        """
        initial = np.array([s[0][:3] for s in self.features])
        self.l_length_light = np.count_nonzero(initial == 'l_l')
        self.l_length_heavy = np.count_nonzero(initial == 'l_h')

        self.a_length_light = np.count_nonzero(initial == 'a_l')
        self.a_length_heavy = np.count_nonzero(initial == 'a_h')

        self.vj_length_light = np.count_nonzero((initial == 'v_l') | (initial == 'j_l'))
        self.vj_length_heavy = np.count_nonzero((initial == 'v_h') | (initial == 'j_h'))

        self.lengt_encoding = np.max([len(self.features), 1])

        min_clip = copy(self.min_energy_clip)
        max_clip = copy(self.max_energy_clip)
        l2_reg = copy(self.l2_reg)

        if initialize:
            input_l_light = keras.layers.Input(shape=(self.l_length_light,))
            input_l_heavy = keras.layers.Input(shape=(self.l_length_heavy,))

            input_cdr3_light = keras.layers.Input(shape=(self.max_depth * 2, 20,))
            input_cdr3_heavy = keras.layers.Input(shape=(self.max_depth * 2, 20,))

            input_vj_light = keras.layers.Input(shape=(self.vj_length_light,))
            input_vj_heavy = keras.layers.Input(shape=(self.vj_length_heavy,))
            input_layer = [input_l_light, input_l_heavy,
                           input_cdr3_light, input_cdr3_heavy,
                           input_vj_light, input_vj_heavy]

            if not self.deep:
                l_l = input_l_light
                cdr3_l = keras.layers.Flatten()(input_cdr3_light)
                vj_l = input_vj_light

                l_h = input_l_heavy
                cdr3_h = keras.layers.Flatten()(input_cdr3_heavy)
                vj_h = input_vj_heavy

                merge = keras.layers.Concatenate()([l_l, l_h, cdr3_l, cdr3_h, vj_l, vj_h])
                output_layer = keras.layers.Dense(1, use_bias=False, activation='linear',
                                                  activity_regularizer=keras.regularizers.l2(l2_reg),
                                                  kernel_initializer='zeros')(merge)
            else:
                merge_l = keras.layers.Concatenate()([input_l_light, input_vj_light])
                merge_l = keras.layers.Dense(20, activation='tanh',
                                             kernel_regularizer=keras.regularizers.l2(l2_reg))(merge_l)
                merge_l = keras.layers.BatchNormalization()(merge_l)

                merge_h = keras.layers.Concatenate()([input_l_heavy, input_vj_heavy])
                merge_h = keras.layers.Dense(20, activation='tanh',
                                             kernel_regularizer=keras.regularizers.l2(l2_reg))(merge_h)
                merge_h = keras.layers.BatchNormalization()(merge_h)

                merge = keras.layers.Concatenate()([merge_l, merge_h])
                if not self.independent_chains:
                    merge = keras.layers.Dense(20, activation='tanh',
                                               kernel_regularizer=keras.regularizers.l2(l2_reg))(merge) # joint features

                cdr3_h = EmbedViaMatrix(5)(input_cdr3_heavy)
                cdr3_h = keras.layers.Activation('tanh')(cdr3_h)
                cdr3_h = keras.layers.Flatten()(cdr3_h)
                cdr3_l = EmbedViaMatrix(5)(input_cdr3_light)
                cdr3_l = keras.layers.Activation('tanh')(cdr3_l)
                cdr3_l = keras.layers.Flatten()(cdr3_l)

                final = keras.layers.Concatenate()([merge,cdr3_l, cdr3_h])
                output_layer = keras.layers.Dense(1, use_bias=False, activation='linear',
                                                  activity_regularizer=keras.regularizers.l2(l2_reg),
                                                  kernel_initializer='lecun_normal')(final) #normal glm model

        # Define model
        clipped_out = keras.layers.Lambda(lambda x: K.clip(x,min_clip,max_clip))(output_layer)
        self.model = keras.models.Model(inputs=input_layer, outputs=clipped_out)
        self.optimizer = keras.optimizers.RMSprop()
        self.model.compile(optimizer=self.optimizer, loss=self._loss, metrics=[self._likelihood])
        self.model_params = self.model.get_weights()
        return True

    def _encode_data(self,
                 seq_features: Iterable[Iterable[int]]
                ) -> List[np.ndarray]:
        length_input = len(self.features)
        length_seq_features = len(seq_features)
        data_enc = np.zeros((length_seq_features, length_input), dtype=np.int8)
        for i, seq_feats in enumerate(seq_features): data_enc[i][seq_feats] = 1

        l_length_total = self.l_length_light + self.l_length_heavy
        a_length_total = self.a_length_light + self.a_length_heavy

        encl_l = data_enc[:, :self.l_length_light]
        encl_h = data_enc[:, self.l_length_light:l_length_total]

        enca_l = (data_enc[:, l_length_total:l_length_total + self.a_length_light]
                  .reshape(length_seq_features, 20, self.max_depth * 2)
                  .swapaxes(1, 2))
        enca_h = (data_enc[:, l_length_total+ self.a_length_light:l_length_total + a_length_total]
                  .reshape(length_seq_features, 20, self.max_depth * 2)
                  .swapaxes(1, 2))

        encv_l = data_enc[:, l_length_total + a_length_total:l_length_total + a_length_total + self.vj_length_light]
        encv_h = data_enc[:, l_length_total + a_length_total + self.vj_length_light:]
        return [encl_l, encl_h, enca_l, enca_h, encv_l, encv_h]

    def _loss(self,
              y_true,
              y_pred
             ) -> float:
        """Loss function for keras training"""
        gamma = 1e0
        data = K.sum((-y_pred) * (1. - y_true)) / K.sum(1. - y_true)
        gen = K.log(K.sum(K.exp(-y_pred) * y_true)) - K.log(K.sum(y_true))
        reg = K.exp(gen) - 1.
        return gen - data + gamma * reg * reg

    def _load_features_and_model(self,
                                 feature_file: str,
                                 model_file: str,
                                 verbose: bool = True
                                ) -> None:
        """Loads features and model.
        """
        if feature_file is None and verbose:
            print('No feature file provided --  no features loaded.')
        elif os.path.isfile(feature_file):
            features = []
            data_marginals=[]
            gen_marginals=[]
            model_marginals=[]
            initial = []

            with open(feature_file, 'r') as features_file:
                next(features_file)

                for line in features_file:
                    splitted = line.strip().split(',')
                    features.append(splitted[0].split(';'))
                    initial.append(features[-1][0][:3])
                    data_marginals.append(float(splitted[1]))
                    model_marginals.append(float(splitted[2]))
                    gen_marginals.append(float(splitted[3]))

            self.features = np.array(features, dtype=object)
            self.feature_dict = {tuple(f): i for i, f in enumerate(self.features)}
            self.data_marginals = np.array(data_marginals)
            self.model_marginals = np.array(model_marginals)
            self.gen_marginals = np.array(gen_marginals)

            initial = np.array(initial)

            self.l_length_light = np.count_nonzero(initial == 'l_l')
            self.l_length_heavy = np.count_nonzero(initial == 'l_h')

            self.a_length_light = np.count_nonzero(initial == 'a_l')
            self.a_length_heavy = np.count_nonzero(initial == 'a_h')

            self.vj_length_light = np.count_nonzero((initial == 'v_l') | (initial == 'j_l'))
            self.vj_length_heavy = np.count_nonzero((initial == 'v_h') | (initial == 'j_h'))
        elif verbose:
            print('Cannot find features file or model file --  no features loaded.')

        if model_file is None and verbose:
            print('No model file provided -- no model parameters loaded.')
        elif os.path.isfile(model_file):
            self.model = keras.models.load_model(model_file,
                                                 custom_objects={'loss': self._loss,
                                                                 'likelihood': self._likelihood,
                                                                 'EmbedViaMatrix': EmbedViaMatrix},
                                                 compile=False)
            self.optimizer = keras.optimizers.RMSprop()
            self.model.compile(optimizer=self.optimizer, loss=self._loss,metrics=[self._likelihood])
        elif verbose:
            print('Cannot find model file --  no model parameters loaded.')

class EmbedViaMatrix(keras.layers.Layer):
    """
    This layer defines a (learned) matrix M such that given matrix input X the
    output is XM. The number of columns of M is embedding_dim, and the number
    of rows is set so that X and M can be multiplied.
    If the rows of the input give the coordinates of a series of objects, we
    can think of this layer as giving an embedding of each of the encoded
    objects in a embedding_dim-dimensional space.
    """

    def __init__(self, embedding_dim, **kwargs):
        self.embedding_dim = embedding_dim
        super(EmbedViaMatrix, self).__init__(**kwargs)

    def build(self, input_shape):
        self.kernel = self.add_weight(
            name='kernel', shape=(input_shape[2], self.embedding_dim), initializer='uniform', trainable=True)
        super(EmbedViaMatrix, self).build(input_shape)  # Be sure to call this at the end

    def get_config(self):
        config = super(EmbedViaMatrix,self).get_config().copy()
        config.update({'embedding_dim': self.embedding_dim})
        return config

    def call(self, x):
        return K.dot(x, self.kernel)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1], self.embedding_dim)
