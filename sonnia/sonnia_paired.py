#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
@author: Giulio Isacchini
"""


import os
import numpy as np
import os
import logging
logging.getLogger('tensorflow').disabled = True
from tensorflow.keras.models import load_model as lm
from tensorflow.keras.callbacks import ModelCheckpoint
import tensorflow.keras.backend as K
from tensorflow import keras
from copy import copy
import itertools
from sonnia.sonia_paired import SoniaPaired
import multiprocessing as mp
#Set input = raw_input for python 2
try:
    import __builtin__
    input = getattr(__builtin__, 'raw_input')
except (ImportError, AttributeError):
    pass

class SoNNiaPaired(SoniaPaired):
    
    def __init__(self, data_seqs = [], gen_seqs = [],chain_type_heavy='human_T_beta',chain_type_light='human_T_alpha',
                 load_dir = None, feature_file = None, data_seq_file = None, gen_seq_file = None, log_file = None, load_seqs = True,
                 max_depth = 25, max_L = 30, include_indep_genes = False, include_joint_genes = True, min_energy_clip = -10, max_energy_clip = 10, seed = None,custom_olga_model_light=None,custom_olga_model_heavy=None,deep=True,l2_reg=1e-3,independent_chains=False):
        self.max_depth = max_depth
        self.deep=deep
        self.include_genes=include_joint_genes or include_indep_genes
        self.independent_chains=independent_chains
        SoniaPaired.__init__(self, data_seqs = data_seqs, gen_seqs = gen_seqs,
                 load_dir = load_dir, feature_file = feature_file, data_seq_file = data_seq_file, gen_seq_file = gen_seq_file, log_file = log_file, load_seqs = load_seqs,chain_type_heavy=chain_type_heavy,chain_type_light=chain_type_light,
                 max_depth = max_depth, max_L = max_L, include_indep_genes = include_indep_genes, include_joint_genes = include_joint_genes, min_energy_clip = min_energy_clip, max_energy_clip = max_energy_clip, seed = seed, custom_olga_model_light=custom_olga_model_light, custom_olga_model_heavy=custom_olga_model_heavy)
        self.l2_reg=l2_reg
        
    def update_model_structure(self,output_layer=[],input_layer=[],initialize=False):
        """ Defines the model structure and compiles it.

        Parameters
        ----------
        structure : Sequential Model Keras
            structure of the model
        initialize: bool
            if True, it initializes to linear model, otherwise it updates to new structure

        """
        initial=np.array([s[0][:3] for s in self.features])
        self.l_length_light=len(np.arange(len(initial))[initial=='l_l'])
        self.l_length_heavy=len(np.arange(len(initial))[initial=='l_h'])

        self.a_length_light=len(np.arange(len(initial))[initial=='a_l'])
        self.a_length_heavy=len(np.arange(len(initial))[initial=='a_h'])

        self.vj_length_light=len(np.arange(len(initial))[np.logical_or(initial=='v_l',initial=='j_l')])
        self.vj_length_heavy=len(np.arange(len(initial))[np.logical_or(initial=='v_h',initial=='j_h')])
        
        self.lengt_encoding=np.max([len(self.features),1])
        
        min_clip=copy(self.min_energy_clip)
        max_clip=copy(self.max_energy_clip)
        l2_reg=copy(self.l2_reg)
        if initialize:
            input_l_light= keras.layers.Input(shape=(self.l_length_light,))
            input_l_heavy= keras.layers.Input(shape=(self.l_length_heavy,))

            input_cdr3_light= keras.layers.Input(shape=(self.max_depth*2,20,))
            input_cdr3_heavy= keras.layers.Input(shape=(self.max_depth*2,20,))
            
            input_vj_light= keras.layers.Input(shape=(self.vj_length_light,))
            input_vj_heavy= keras.layers.Input(shape=(self.vj_length_heavy,))
            input_layer=[input_l_light,input_l_heavy,input_cdr3_light,input_cdr3_heavy,input_vj_light,input_vj_heavy]
            
            if not self.deep:
                l_l=input_l_light
                cdr3_l=keras.layers.Flatten()(input_cdr3_light)
                vj_l=input_vj_light

                l_h=input_l_heavy
                cdr3_h=keras.layers.Flatten()(input_cdr3_heavy)
                vj_h=input_vj_heavy

                merge=keras.layers.Concatenate()([l_l,l_h,cdr3_l,cdr3_h,vj_l,vj_h])
                output_layer=keras.layers.Dense(1,use_bias=False,activation='linear', activity_regularizer=keras.regularizers.l2(l2_reg),kernel_initializer='zeros')(merge)
                
            else:

                merge_l=keras.layers.Concatenate()([input_l_light,input_vj_light])
                merge_l=keras.layers.Dense(20,activation='tanh',kernel_regularizer=keras.regularizers.l2(l2_reg))(merge_l)
                merge_l=keras.layers.BatchNormalization()(merge_l)

                merge_h=keras.layers.Concatenate()([input_l_heavy,input_vj_heavy])
                merge_h=keras.layers.Dense(20,activation='tanh',kernel_regularizer=keras.regularizers.l2(l2_reg))(merge_h)
                merge_h=keras.layers.BatchNormalization()(merge_h)

                merge=keras.layers.Concatenate()([merge_l,merge_h])
                if not self.independent_chains:
                    merge=keras.layers.Dense(20,activation='tanh',kernel_regularizer=keras.regularizers.l2(l2_reg))(merge) # joint features

                cdr3_h=EmbedViaMatrix(5)(input_cdr3_heavy)
                cdr3_h=keras.layers.Activation('tanh')(cdr3_h)
                cdr3_h=keras.layers.Flatten()(cdr3_h)
                cdr3_l=EmbedViaMatrix(5)(input_cdr3_light)
                cdr3_l=keras.layers.Activation('tanh')(cdr3_l)
                cdr3_l=keras.layers.Flatten()(cdr3_l)
        
                final=keras.layers.Concatenate()([merge,cdr3_l,cdr3_h])
                output_layer = keras.layers.Dense(1,use_bias=False,activation='linear',
                                                  activity_regularizer=keras.regularizers.l2(l2_reg),kernel_initializer='lecun_normal')(final) #normal glm model
                
        # Define model
        clipped_out=keras.layers.Lambda(lambda x: K.clip(x,min_clip,max_clip))(output_layer)
        self.model = keras.models.Model(inputs=input_layer, outputs=clipped_out)
        self.optimizer = keras.optimizers.RMSprop()
        self.model.compile(optimizer=self.optimizer, loss=self._loss,metrics=[self._likelihood])
        self.model_params = self.model.get_weights()
        return True 

    def _encode_data(self,seq_features):
        length_input=len(self.features)
        data=np.array(seq_features)
        data_enc = np.zeros((len(data), length_input), dtype=np.int8)
        for i in range(len(data_enc)): data_enc[i][data[i]] = 1
        encl_l,enca_l,encv_l=[],[],[]
        encl_h,enca_h,encv_h=[],[],[]
        l=self.l_length_light+self.l_length_heavy
        a=self.a_length_light+self.a_length_heavy
        for x in data_enc:
            encl_l.append(x[:self.l_length_light])
            encl_h.append(x[self.l_length_light:l])

            enca_l.append(x[l: l+self.a_length_light ].reshape(20,self.max_depth*2).T)
            enca_h.append(x[l+self.a_length_light:l+a].reshape(20,self.max_depth*2).T)

            encv_l.append(x[l+a:l+a+self.vj_length_light])
            encv_h.append(x[l+a+self.vj_length_light:])
        return [np.array(encl_l),np.array(encl_h),np.array(enca_l),np.array(enca_h),np.array(encv_l),np.array(encv_h)]

    def _loss(self, y_true, y_pred):
        """Loss function for keras training"""
        gamma=1e0
        data= K.sum((-y_pred)*(1.-y_true))/K.sum(1.-y_true)
        gen= K.log(K.sum(K.exp(-y_pred)*y_true))-K.log(K.sum(y_true))
        reg= K.exp(gen)-1.
        return gen-data+gamma*reg*reg
    
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
                all_lines = features_file.read().strip().split('\n')[1:] #skip header
                splitted=[l.split(',') for l in all_lines]
                features = np.array([l[0].split(';') for l in splitted])
                data_marginals=[float(l[1]) for l in splitted]
                model_marginals=[float(l[2]) for l in splitted]
                gen_marginals=[float(l[3]) for l in splitted]
            
            self.features = np.array(features)
            self.feature_dict = {tuple(f): i for i, f in enumerate(self.features)}
            self.data_marginals=data_marginals
            self.model_marginals=model_marginals
            self.gen_marginals=gen_marginals
            
            initial=np.array([s[0][:3] for s in self.features])

            self.l_length_light=len(np.arange(len(initial))[initial=='l_l'])
            self.l_length_heavy=len(np.arange(len(initial))[initial=='l_h'])

            self.a_length_light=len(np.arange(len(initial))[initial=='a_l'])
            self.a_length_heavy=len(np.arange(len(initial))[initial=='a_h'])

            self.vj_length_light=len(np.arange(len(initial))[np.logical_or(initial=='v_l',initial=='j_l')])
            self.vj_length_heavy=len(np.arange(len(initial))[np.logical_or(initial=='v_h',initial=='j_h')])
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