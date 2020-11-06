#!/usr/bin/env python
# -*- coding: utf-8 -*-

# @author: Giulio Isacchini

import os
import numpy as np
import logging
logging.getLogger('tensorflow').disabled = True
from tensorflow.keras.models import load_model as lm
from tensorflow.keras.callbacks import ModelCheckpoint
import tensorflow.keras.backend as K
from tensorflow import keras
from sonia.sonia import Sonia
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

class SoNNiaTransfer(Sonia):
    
    def __init__(self, data_seqs = [], gen_seqs = [], chain_type = 'humanTRB',
                 load_dir = None, feature_file = None, data_seq_file = None, gen_seq_file = None, log_file = None, load_seqs = True,
                 max_depth = 25, max_L = 30, include_indep_genes = False, include_joint_genes = True, min_energy_clip = -5, max_energy_clip = 10, seed = None,custom_pgen_model=None ,deep=True, l2_reg=0.,l1_reg=0.,joint_vjl=False,vj=False,batch_norm=True):
        self.batch_norm=batch_norm
        self.max_depth=max_depth
        self.max_L = max_L
        self.deep=deep
        self.include_indep_genes=include_indep_genes
        self.include_joint_genes=include_joint_genes
        self.joint_vjl=joint_vjl
        self.custom_pgen_model=custom_pgen_model
        Sonia.__init__(self, data_seqs=data_seqs, gen_seqs=gen_seqs, chain_type=chain_type, 
                       min_energy_clip = min_energy_clip, max_energy_clip = max_energy_clip, seed = seed,l2_reg=l2_reg,l1_reg=l1_reg,vj=vj)
        
        if any([x is not None for x in [load_dir, feature_file]]):
            self.load_model(load_dir = load_dir, feature_file = feature_file, data_seq_file = data_seq_file, 
                            gen_seq_file = gen_seq_file, log_file = log_file, load_seqs = load_seqs)
        else:
            self.add_features(custom_pgen_model = custom_pgen_model)
        
    def add_features(self, custom_pgen_model=None):
        """Generates a list of feature_lsts for L/R pos model.

        Parameters
        ----------
        custom_pgen_model: string
            path to folder of custom olga model.

        """
        features = []
        import olga.load_model as olga_load_model
        import olga.generation_probability as pgen
        import olga.sequence_generation as seq_gen
        if custom_pgen_model is None:
            main_folder = os.path.join(os.path.dirname(sonia.sonia.__file__), 'default_models', self.chain_type)
        else:
            main_folder = custom_pgen_model
        params_file_name = os.path.join(main_folder,'model_params.txt')
        marginals_file_name = os.path.join(main_folder,'model_marginals.txt')
        V_anchor_pos_file = os.path.join(main_folder,'V_gene_CDR3_anchors.csv')
        J_anchor_pos_file = os.path.join(main_folder,'J_gene_CDR3_anchors.csv')
            
        if self.vj: 
            self.genomic_data = olga_load_model.GenomicDataVJ()
            self.generative_model = olga_load_model.GenerativeModelVJ()
            self.genomic_data.load_igor_genomic_data(params_file_name, V_anchor_pos_file, J_anchor_pos_file)
            self.generative_model.load_and_process_igor_model(marginals_file_name)
            self.seq_gen_model = seq_gen.SequenceGenerationVJ(self.generative_model, self.genomic_data)
            self.pgen_model = pgen.GenerationProbabilityVJ(self.generative_model, self.genomic_data)
        else: 
            self.genomic_data = olga_load_model.GenomicDataVDJ()
            self.generative_model = olga_load_model.GenerativeModelVDJ()
            self.genomic_data.load_igor_genomic_data(params_file_name, V_anchor_pos_file, J_anchor_pos_file)
            self.generative_model.load_and_process_igor_model(marginals_file_name)
            self.seq_gen_model = seq_gen.SequenceGenerationVDJ(self.generative_model, self.genomic_data)
            self.pgen_model = pgen.GenerationProbabilityVDJ(self.generative_model, self.genomic_data)
            
        if self.joint_vjl:
            features += [[v, j, 'l'+str(l)] for v in set([gene_to_num_str(genV[0],'V')for genV in self.genomic_data.genV]) for j in set([gene_to_num_str(genJ[0],'J') for genJ in self.genomic_data.genJ]) for l in range(1, self.max_L + 1)]
        else:
            features += [['l' + str(L)] for L in range(1, self.max_L + 1)]
            
        for aa in self.amino_acids:
            features += [['a' + aa + str(L)] for L in range(self.max_depth)]
            features += [['a' + aa + str(L)] for L in range(-self.max_depth, 0)]  
            
        if self.include_indep_genes:
            features += [[v] for v in set([gene_to_num_str(genV[0],'V') for genV in self.genomic_data.genV])]
            features += [[j] for j in set([gene_to_num_str(genJ[0],'J') for genJ in self.genomic_data.genJ])]
        if self.include_joint_genes:
            features += [[v, j] for v in set([gene_to_num_str(genV[0],'V') for genV in self.genomic_data.genV]) for j in set([gene_to_num_str(genJ[0],'J') for genJ in self.genomic_data.genJ])]

        self.update_model(add_features=features)

    def find_seq_features(self, seq, features = None):
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
            seq_feature_lsts = [['l' + str(len(seq[0]))]]
            seq_feature_lsts += [['a' + aa + str(i)] for i, aa in enumerate(seq[0])]
            seq_feature_lsts += [['a' + aa + str(-1-i)] for i, aa in enumerate(seq[0][::-1])]
            v_genes = [gene for gene in seq[1:] if 'v' in gene.lower()]
            j_genes = [gene for gene in seq[1:] if 'j' in gene.lower()]
            #Allow for just the gene family match
            v_genes += [gene.split('-')[0] for gene in seq[1:] if 'v' in gene.lower()]
            j_genes += [gene.split('-')[0] for gene in seq[1:] if 'j' in gene.lower()]

            try:
                seq_feature_lsts += [[gene_to_num_str(gene,'V')] for gene in v_genes]
                seq_feature_lsts += [[gene_to_num_str(gene,'J')] for gene in j_genes]
                seq_feature_lsts += [[gene_to_num_str(v_gene,'V'), gene_to_num_str(j_gene,'J')] for v_gene in v_genes for j_gene in j_genes]
            except ValueError:
                pass
            seq_features = list(set([self.feature_dict[tuple(f)] for f in seq_feature_lsts if tuple(f) in self.feature_dict]))
        else:
            seq_features = []
            for feature_index,feature_lst in enumerate(features):
                if self.seq_feature_proj(feature_lst, seq):
                    seq_features += [feature_index]

        return seq_features
        
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
            l=keras.layers.Dense(10,
                                     activation='tanh',
                                     kernel_initializer='lecun_normal',
                                     kernel_regularizer=keras.regularizers.l2(l2_reg))(input_l)
            cdr3=EmbedViaMatrix(10)(input_cdr3)
            cdr3=keras.layers.Activation('tanh')(cdr3)
            cdr3=keras.layers.Flatten()(cdr3)
            cdr3=keras.layers.Dense(30,
                                        activation='tanh',
                                        kernel_initializer='lecun_normal',
                                        kernel_regularizer=keras.regularizers.l2(l2_reg))(cdr3)
            vj=keras.layers.Dense(30,
                                      activation='tanh',
                                      kernel_initializer='lecun_normal',
                                      kernel_regularizer=keras.regularizers.l2(l2_reg))(input_vj)
            vj=keras.layers.BatchNormalization()(vj)
            merge=keras.layers.Concatenate()([l,cdr3,vj])
            h=keras.layers.Dense(50,
                                     activation='tanh',
                                     kernel_initializer='lecun_normal', 
                                     kernel_regularizer=keras.regularizers.l2(l2_reg))(merge)
            output_layer2=keras.layers.Dense(1,
                                                activation='linear',
                                                use_bias=False,
                                                kernel_initializer='zeros', 
                                                kernel_regularizer=keras.regularizers.l2(l2_reg))(h)

            cdr31=keras.layers.Flatten()(input_cdr3)
            merge1=keras.layers.Concatenate()([input_l,cdr31,input_vj])
            output_layer1=keras.layers.Dense(1,
                                                use_bias=False,
                                                activation='linear',
                                                kernel_regularizer=keras.regularizers.l1_l2(l2=l2_reg,l1=l1_reg),
                                                kernel_initializer='zeros')(merge1)
        
        output_layer=keras.layers.Add()([output_layer1,output_layer2])
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
        enc1,enc2,enc3=[],[],[]
        for x in data_enc:
            enc1.append(x[:self.l_length])
            enc2.append(x[self.l_length:self.l_length+self.a_length].reshape(20,50).T)
            enc3.append(x[self.l_length+self.a_length:])
        return [np.array(enc1),np.array(enc2),np.array(enc3)]
    
    def infer_selection(self, epochs = 20, batch_size=5000, initialize = True, seed = None,validation_split=0.2, monitor=False,verbose=0,part='all'):
        """Infer model parameters, i.e. energies for each model feature.
        Parameters
        ----------
        epochs : int
            Maximum number of learning epochs
        intialize : bool
            Resets data shuffle
        batch_size : int
            Size of the batches in the inference
        seed : int
            Sets random seed
        Attributes set
        --------------
        model : keras model
            Parameters of the model
        model_marginals : array
            Marginals over the generated sequences, reweighted by the model.
        L1_converge_history : list
            L1 distance between data_marginals and model_marginals at each
            iteration.
        """

        if seed is not None:
            np.random.seed(seed = seed)
        if initialize:
            # prepare data
            self.X = np.array(self.data_seq_features+self.gen_seq_features)
            self.Y = np.concatenate([np.zeros(len(self.data_seq_features)), np.ones(len(self.gen_seq_features))])

            shuffle = np.random.permutation(len(self.X)) # shuffle
            self.X=self.X[shuffle]
            self.Y=self.Y[shuffle]

        if part=='linear':
            for i in range(len(self.model.layers)):
                self.model.layers[i].trainable=False
            self.model.layers[-4].trainable=True
            self.model.compile(optimizer=self.optimizer, loss=self._loss,metrics=[self._likelihood])
        elif part=='deep':
            for i in range(len(self.model.layers)):
                self.model.layers[i].trainable=True
            self.model.layers[-4].trainable=False
            self.model.compile(optimizer=self.optimizer, loss=self._loss,metrics=[self._likelihood])
        elif part=='ll':
            for i in range(len(self.model.layers)):
                self.model.layers[i].trainable=False
            self.model.layers[-3].trainable=True
            self.model.layers[-4].trainable=True
            self.model.compile(optimizer=self.optimizer, loss=self._loss,metrics=[self._likelihood])

        callbacks=[]
        self.learning_history = self.model.fit(self._encode_data(self.X), self.Y, epochs=epochs, batch_size=batch_size,
                                          validation_split=validation_split, verbose=verbose, callbacks=callbacks)
        self.likelihood_train=self.learning_history.history['_likelihood']
        self.likelihood_test=self.learning_history.history['val__likelihood']
        # set Z    
        self.energies_gen=self.compute_energy(self.gen_seq_features)
        self.Z=np.sum(np.exp(-self.energies_gen))/len(self.energies_gen)
        for i in range(len(self.model.layers)): self.model.layers[i].trainable=True
        self.model.compile(optimizer=self.optimizer, loss=self._loss,metrics=[self._likelihood])

        self.update_model(auto_update_marginals=True)


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
            
            initial=np.array([s[0][0] for s in self.features])
            self.l_length=len(np.arange(len(initial))[initial=='l'])
            self.a_length=len(np.arange(len(initial))[initial=='a'])
            self.vj_length=len(np.logical_or(np.arange(len(initial))[initial=='v'],np.arange(len(initial))[initial=='j']))
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