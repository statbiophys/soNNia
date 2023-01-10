#!/usr/bin/env python
# -*- coding: utf-8 -*-

# @author: Giulio Isacchini

import os
import numpy as np
import logging
logging.getLogger('tensorflow').disabled = True
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
from tensorflow.keras.models import load_model as lm
import tensorflow.keras.backend as K
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow import keras
from sonnia.sonia import Sonia
from sonia.utils import gene_to_num_str
import sonia.sonia
from copy import copy
import itertools
import multiprocessing as mp
#Set input = raw_input for python 2
try:
    import __builtin__
    input = getattr(__builtin__, 'raw_input')
except (ImportError, AttributeError):
    pass

class SoNNia(Sonia):
    
    def __init__(self, data_seqs = [], gen_seqs = [], chain_type = 'humanTRB',load_dir = None, custom_pgen_model=None ,
                min_energy_clip = -5, max_energy_clip = 10, seed = None, l2_reg=0.,l1_reg=0.,vj=False,gamma=0.1,objective='BCE',
                max_depth = 25, max_L = 30, include_indep_genes = True, include_joint_genes = False, deep=True,joint_vjl=False,include_aminoacids=True):        
        self.deep=deep
        Sonia.__init__(self, data_seqs=data_seqs, gen_seqs=gen_seqs, chain_type=chain_type,load_dir=load_dir, custom_pgen_model=custom_pgen_model,
                       min_energy_clip = min_energy_clip, max_energy_clip = max_energy_clip, seed = seed,l2_reg=l2_reg,l1_reg=l1_reg,vj=vj,
                       objective=objective,gamma=gamma,include_indep_genes=include_indep_genes,include_joint_genes=include_joint_genes,joint_vjl=joint_vjl,
                       include_aminoacids=include_aminoacids,max_depth=max_depth,max_L=max_L)
        
    def update_model_structure(self,output_layer=[],input_layer=[],initialize=False):
        """ Defines the model structure and compiles it.

        Parameters
        ----------
        structure : Sequential Model Keras
            structure of the model
        initialize: bool
            if True, it initializes to linear model, otherwise it updates to new structure

        """
        if len(self.features)>1:
            initial=np.array([s[0][0] for s in self.features])
        else: initial=np.array(['c','c','c'])
        self.l_length=len(np.arange(len(initial))[initial=='l'])
        self.a_length=len(np.arange(len(initial))[initial=='a'])
        self.vj_length=len(np.arange(len(initial))[np.logical_or(initial=='v',initial=='j')])
        
        length_input=np.max([len(self.features),1])
        min_clip=copy(self.min_energy_clip)
        max_clip=copy(self.max_energy_clip)
        l2_reg=copy(self.l2_reg)
        l1_reg=copy(self.l1_reg)

        max_depth=copy(self.max_depth)
        l_length=copy(self.l_length)
        vj_length=copy(self.vj_length)
        if initialize:
            input_l= keras.layers.Input(shape=(l_length,))
            input_cdr3= keras.layers.Input(shape=(max_depth*2,20,))
            input_vj= keras.layers.Input(shape=(vj_length,))
            input_layer=[input_l,input_cdr3,input_vj]
            
            if not self.deep:
                l=input_l
                cdr3=keras.layers.Flatten()(input_cdr3)
                vj=input_vj
                merge=keras.layers.Concatenate()([l,cdr3,vj])
                output_layer=keras.layers.Dense(1,
                                                use_bias=False,
                                                activation='linear',
                                                kernel_regularizer=keras.regularizers.l1_l2(l2=l2_reg,l1=l1_reg),kernel_initializer='zeros')(merge)
            else:
                #define encodings
                l=keras.layers.Dense(10,
                                     activation='tanh',
                                     kernel_initializer='lecun_normal',
                                     kernel_regularizer=keras.regularizers.l2(l2_reg))(input_l)
                cdr3=EmbedViaMatrix(10)(input_cdr3)
                cdr3=keras.layers.Activation('tanh')(cdr3)
                cdr3=keras.layers.Flatten()(cdr3)
                cdr3=keras.layers.Dense(40,
                                        activation='tanh',
                                        kernel_initializer='lecun_normal',
                                        kernel_regularizer=keras.regularizers.l2(l2_reg))(cdr3)
                vj=keras.layers.Dense(30,
                                      activation='tanh',
                                      kernel_initializer='lecun_normal',
                                      kernel_regularizer=keras.regularizers.l2(l2_reg))(input_vj)
                #merge
                merge=keras.layers.Concatenate()([l,cdr3,vj])
                h=keras.layers.Dense(60,
                                     activation='tanh',
                                     kernel_initializer='lecun_normal', 
                                     kernel_regularizer=keras.regularizers.l2(l2_reg))(merge)
                output_layer=keras.layers.Dense(1,
                                                activation='linear',
                                                use_bias=True,
                                                kernel_initializer='lecun_normal', 
                                                kernel_regularizer=keras.regularizers.l2(l2_reg))(h)

        # Define model
        clipped_out=keras.layers.Lambda(lambda x: K.clip(x,min_clip,max_clip))(output_layer)
        self.model = keras.models.Model(inputs=input_layer, outputs=clipped_out)
        self.optimizer = keras.optimizers.RMSprop()

        if self.objective=='BCE':
            self.model.compile(optimizer=self.optimizer, loss=BinaryCrossentropy(from_logits=True),metrics=[self._likelihood, BinaryCrossentropy(from_logits=True,name='binary_crossentropy')])
        else:
            self.model.compile(optimizer=self.optimizer, loss=self._loss, 
                               metrics=[self._likelihood, BinaryCrossentropy(from_logits=True,name='binary_crossentropy')])
        self.model_params = self.model.get_weights()
        return True 

    def _encode_data(self,seq_features):
        length_input=len(self.features)
        #data=np.array(seq_features,dtype=object)
        data_enc = np.zeros((len(seq_features), length_input), dtype=np.int8)
        for i in range(len(data_enc)): data_enc[i][seq_features[i]] = 1
        enc1,enc2,enc3=[],[],[]
        for x in data_enc:
            enc1.append(x[:self.l_length])
            enc2.append(x[self.l_length:self.l_length+self.a_length].reshape(20,50).T)
            enc3.append(x[self.l_length+self.a_length:])
        return [np.array(enc1),np.array(enc2),np.array(enc3)]
    
    def _load_features_and_model(self, feature_file, model_file, verbose = True):
        """Loads features and model.

        This is set as an internal function to allow daughter classes to load
        models from saved feature energies directly.
        """

        if feature_file is None and verbose:
            print('No feature file provided --  no features loaded.')
        elif os.path.isfile(feature_file):
            features = []
            data_marginals=[]
            gen_marginals=[]
            model_marginals=[]
            with open(feature_file, 'r') as features_file:
                lines = features_file.read().strip().split('\n') #skip header
                sonia_or_sonnia=lines[0].split(',')[1]
                if sonia_or_sonnia=='marginal_data':
                    k=0
                else:
                    k=1
                splitted=[l.split(',') for l in lines[1:]]
                features = np.array([l[0].split(';') for l in splitted],dtype=object)
                data_marginals=[float(l[1+k]) for l in splitted]
                model_marginals=[float(l[2+k]) for l in splitted]
                gen_marginals=[float(l[3+k]) for l in splitted]
            
            self.features = np.array(features)
            self.feature_dict = {tuple(f): i for i, f in enumerate(self.features)}
            self.data_marginals=data_marginals
            self.model_marginals=model_marginals
            self.gen_marginals=gen_marginals
            
            initial=np.array([s[0][0] for s in self.features])
            self.l_length=len(np.arange(len(initial))[initial=='l'])
            self.a_length=len(np.arange(len(initial))[initial=='a'])
            self.vj_length=len(set(np.arange(len(initial))[initial=='v'])|set(np.arange(len(initial))[initial=='j']))
        elif verbose:
            print('Cannot find features file or model file --  no features loaded.')

        if model_file is None and verbose:
            print('No model file provided -- no model parameters loaded.')
        elif os.path.isfile(model_file):
            self.model = keras.models.load_model(model_file, custom_objects={'loss': self._loss,'likelihood': self._likelihood,"EmbedViaMatrix":EmbedViaMatrix}, compile = False)
            self.optimizer = keras.optimizers.RMSprop()
            self.model.compile(optimizer=self.optimizer, loss=self._loss,metrics=[self._likelihood])
        elif verbose:
            print('Cannot find model file --  no model parameters loaded.')

    def set_gauge(self):
        '''
        placeholder for gauges.
        '''
        pass

class EmbedViaMatrix(keras.layers.Layer):
    """
    This layer defines a (learned) matrix M such that given matrix input X the
    output is XM. The number of columns of M is embedding_dim, and the number
    of rows is set so that X and M can be multiplied.
    If the rows of the input give the coordinates of a series of objects, we
    can think of this layer as giving an embedding of each of the encoded
    objects in a embedding_dim-dimensional space.
    Adapted from Vampire package.
    """

    def __init__(self, embedding_dim, **kwargs):
        self.embedding_dim = embedding_dim
        super(EmbedViaMatrix, self).__init__(**kwargs)

    def build(self, input_shape):
        self.kernel = self.add_weight(
            name='kernel', shape=(input_shape[2], self.embedding_dim), initializer='uniform', trainable=True)
        super(EmbedViaMatrix, self).build(input_shape)
        
    def get_config(self):
        config = super(EmbedViaMatrix,self).get_config().copy()
        config.update({'embedding_dim': self.embedding_dim})
        return config

    def call(self, x):
        return K.dot(x, self.kernel)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1], self.embedding_dim)