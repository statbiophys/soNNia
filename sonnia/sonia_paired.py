#!/usr/bin/env python
# -*- coding: utf-8 -*-

# @author: Giulio Isacchini

import os
from tensorflow import keras
import tensorflow.keras.backend as K
import multiprocessing as mp
import numpy as np
import olga.load_model as load_model
import olga.sequence_generation as seq_gen
from sonia.utils import gene_to_num_str
from sonia.sonia import Sonia
import sonia.sonia
import olga.generation_probability as generation_probability

class SoniaPaired(Sonia):
    
    def __init__(self, data_seqs = [], gen_seqs = [], chain_type_heavy='human_T_beta',chain_type_light='human_T_alpha', load_dir = None, feature_file = None, data_seq_file = None, gen_seq_file = None, log_file = None, load_seqs = True,
                 max_depth = 25, max_L = 30, include_indep_genes = False, include_joint_genes = True, min_energy_clip = -10, max_energy_clip = 10, seed = None,custom_olga_model_light=None,custom_olga_model_heavy=None,l2_reg=0.):

        Sonia.__init__(self, data_seqs=data_seqs, gen_seqs=gen_seqs, min_energy_clip = min_energy_clip, max_energy_clip = max_energy_clip, seed = seed,l2_reg=l2_reg)        
        self.max_depth = max_depth
        self.max_L = max_L
        self.include_genes=include_joint_genes or include_indep_genes
        self.chain_type_heavy=chain_type_heavy
        self.chain_type_light=chain_type_light
        self.include_indep_genes=include_indep_genes
        self.include_joint_genes=include_joint_genes
        self.initialize_olga_models(custom_olga_model_light = custom_olga_model_light, custom_olga_model_heavy = custom_olga_model_heavy)
        if any([x is not None for x in [load_dir, feature_file]]):
            self.load_model(load_dir = load_dir, feature_file = feature_file, data_seq_file = data_seq_file, gen_seq_file = gen_seq_file, log_file = log_file, load_seqs = load_seqs)
        else:
            self.add_features(include_indep_genes = include_indep_genes, include_joint_genes = include_joint_genes)
            
    def initialize_olga_models(self,custom_olga_model_light=None,custom_olga_model_heavy=None):
        if custom_olga_model_light is None: main_folder_light = os.path.join(os.path.dirname(sonia.sonia.__file__), 'default_models', self.chain_type_light)
        else: main_folder_light=custom_olga_model_light
        if custom_olga_model_heavy is None: main_folder_heavy = os.path.join(os.path.dirname(sonia.sonia.__file__), 'default_models', self.chain_type_heavy)
        else: main_folder_heavy=custom_olga_model_heavy

        marginals_file_name = os.path.join(main_folder_light,'model_marginals.txt')
        params_file_name = os.path.join(main_folder_light,'model_params.txt')
        V_anchor_pos_file = os.path.join(main_folder_light,'V_gene_CDR3_anchors.csv')
        J_anchor_pos_file = os.path.join(main_folder_light,'J_gene_CDR3_anchors.csv')
        self.genomic_data_light = load_model.GenomicDataVJ()
        self.genomic_data_light.load_igor_genomic_data(params_file_name, V_anchor_pos_file, J_anchor_pos_file)
        self.generative_model_light = load_model.GenerativeModelVJ()
        self.generative_model_light.load_and_process_igor_model(marginals_file_name)        
        self.seq_gen_model_light = seq_gen.SequenceGenerationVJ(self.generative_model_light, self.genomic_data_light)
        self.pgen_model_light = generation_probability.GenerationProbabilityVJ(self.generative_model_light, self.genomic_data_light)

        params_file_name = os.path.join(main_folder_heavy,'model_params.txt')
        V_anchor_pos_file = os.path.join(main_folder_heavy,'V_gene_CDR3_anchors.csv')
        J_anchor_pos_file = os.path.join(main_folder_heavy,'J_gene_CDR3_anchors.csv')
        marginals_file_name = os.path.join(main_folder_heavy,'model_marginals.txt')
        self.genomic_data_heavy= load_model.GenomicDataVDJ()
        self.genomic_data_heavy.load_igor_genomic_data(params_file_name, V_anchor_pos_file, J_anchor_pos_file)
        self.generative_model_heavy = load_model.GenerativeModelVDJ()
        self.generative_model_heavy.load_and_process_igor_model(marginals_file_name)    
        self.seq_gen_model_heavy = seq_gen.SequenceGenerationVDJ(self.generative_model_heavy, self.genomic_data_heavy)    
        self.pgen_model_heavy = generation_probability.GenerationProbabilityVDJ(self.generative_model_heavy, self.genomic_data_heavy)
        
            
    def add_features(self, include_indep_genes = False, include_joint_genes = True):
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
            features += [[v, j] for v in set(['v_l' + gene_to_num_str(genV[0],'V')[1:] for genV in self.genomic_data_light.genV]) for j in set(['j_l' + gene_to_num_str(genJ[0],'J')[1:] for genJ in self.genomic_data_light.genJ])]
            features += [[v, j] for v in set(['v_h'  + gene_to_num_str(genV[0],'V')[1:] for genV in self.genomic_data_heavy.genV]) for j in set(['j_h'  + gene_to_num_str(genJ[0],'J')[1:] for genJ in self.genomic_data_heavy.genJ])]
        if self.include_indep_genes: 
            features += [[v] for v in set(['v_l' + gene_to_num_str(genV[0],'V')[1:] for genV in self.genomic_data_light.genV])]
            features += [[j] for j in set(['j_l'  + gene_to_num_str(genJ[0],'J')[1:] for genJ in self.genomic_data_light.genJ])]
            features += [[v] for v in set(['v_h' + gene_to_num_str(genV[0],'V')[1:] for genV in self.genomic_data_heavy.genV])]
            features += [[j] for j in set(['j_h'  + gene_to_num_str(genJ[0],'J')[1:] for genJ in self.genomic_data_heavy.genJ])]
    
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
        #0-2 is heavy 3-5 is light
        if features is None:
            seq_feature_lsts = [['l_h' + str(len(seq[0]))]]
            seq_feature_lsts += [['l_l' + str(len(seq[3]))]]

            seq_feature_lsts += [['a_h' + aa + str(i)] for i, aa in enumerate(seq[0])]
            seq_feature_lsts += [['a_h' + aa + str(-1-i)] for i, aa in enumerate(seq[0][::-1])]

            seq_feature_lsts += [['a_l' + aa + str(i)] for i, aa in enumerate(seq[3])]
            seq_feature_lsts += [['a_l' + aa + str(-1-i)] for i, aa in enumerate(seq[3][::-1])]

            v_genes_heavy = [gene for gene in seq[1:3] if 'v' in gene.lower()]
            j_genes_heavy = [gene for gene in seq[1:3] if 'j' in gene.lower()]
            v_genes_light = [gene for gene in seq[4:] if 'v' in gene.lower()]
            j_genes_light = [gene for gene in seq[4:] if 'j' in gene.lower()]
            
            try:
                seq_feature_lsts += [['v_h' + gene_to_num_str(gene,'V')[1:]] for gene in v_genes_heavy]
                seq_feature_lsts += [['j_h' + gene_to_num_str(gene,'J')[1:]] for gene in j_genes_heavy]
                seq_feature_lsts += [['v_l' + gene_to_num_str(gene,'V')[1:]] for gene in v_genes_light]
                seq_feature_lsts += [['j_l' + gene_to_num_str(gene,'J')[1:]] for gene in j_genes_light]
                seq_feature_lsts += [['v_h' + gene_to_num_str(v_gene,'V')[1:], 'j_h' + gene_to_num_str(j_gene,'J')[1:]] for v_gene in v_genes_heavy for j_gene in j_genes_heavy]
                seq_feature_lsts += [['v_l' + gene_to_num_str(v_gene,'V')[1:], 'j_l' +  gene_to_num_str(j_gene,'J')[1:]] for v_gene in v_genes_light for j_gene in j_genes_light]
            except ValueError:
                pass
            seq_features = list(set([self.feature_dict[tuple(f)] for f in seq_feature_lsts if tuple(f) in self.feature_dict]))
        else:
            seq_features = []
            for feature_index,feature_lst in enumerate(features):
                if self.seq_feature_proj(feature_lst, seq):
                    seq_features += [feature_index]
                
        return seq_features

    def add_generated_seqs(self, num_gen_seqs = 0, reset_gen_seqs = True):
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

        Attributes set
        --------------
        gen_seqs : list
            MonteCarlo sequences drawn from a VDJ recomb model
        gen_seq_features : list
            Features gen_seqs have been projected onto.

        """
        #Load OLGA for seq generation
        #Generate sequences
        seqs_light = [[seq[1], self.genomic_data_light.genV[seq[2]][0].split('*')[0], self.genomic_data_light.genJ[seq[3]][0].split('*')[0]] for seq in [self.seq_gen_model_light.gen_rnd_prod_CDR3(conserved_J_residues='ABCEDFGHIJKLMNOPQRSTUVWXYZ') for _ in range(int(num_gen_seqs))]]
        seqs_heavy = [[seq[1], self.genomic_data_heavy.genV[seq[2]][0].split('*')[0], self.genomic_data_heavy.genJ[seq[3]][0].split('*')[0]] for seq in [self.seq_gen_model_heavy.gen_rnd_prod_CDR3(conserved_J_residues='ABCEDFGHIJKLMNOPQRSTUVWXYZ') for _ in range(int(num_gen_seqs))]]
        seqs = [a+b for a,b in zip(seqs_heavy,seqs_light)]
        if reset_gen_seqs: #reset gen_seqs if needed
            self.gen_seqs = []
        #Add to specified pool(s)
        self.update_model(add_gen_seqs = seqs)