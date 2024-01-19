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
from sonnia.utils import gene_to_num_str
from sonnia.sonia import Sonia
import olga.generation_probability as generation_probability
from tqdm import tqdm
import sonnia.sonnia

class SoniaPaired(Sonia):
    
    def __init__(self, data_seqs = [], gen_seqs = [], chain_type_heavy='human_T_beta',chain_type_light='human_T_alpha', load_dir = None, 
                 max_depth = 25, max_L = 30, include_indep_genes = False, include_joint_genes = True, 
                 min_energy_clip = -10, max_energy_clip = 10, seed = None,
                 custom_pgen_model_light=None,custom_pgen_model_heavy=None,l2_reg=0.,l1_reg=0.):

        self.chain_type_heavy=chain_type_heavy
        self.chain_type_light=chain_type_light
        self.custom_pgen_model_light=custom_pgen_model_light
        self.custom_pgen_model_heavy=custom_pgen_model_heavy

        Sonia.__init__(self, data_seqs=data_seqs, gen_seqs=gen_seqs, min_energy_clip = min_energy_clip, max_energy_clip = max_energy_clip, seed = seed,
                            l2_reg=l2_reg,l1_reg=l1_reg,max_depth=max_depth,max_L=max_L,include_joint_genes=include_joint_genes,include_indep_genes=include_indep_genes)        
        self.define_models(recompute_norm= not (custom_pgen_model_light is None and custom_pgen_model_heavy is None))

    def define_models(self,recompute_norm=False):

        if self.custom_pgen_model_light is None: main_folder_light = os.path.join(os.path.dirname(sonnia.__file__), 'default_models', self.chain_type_light)
        else: main_folder_light=self.custom_pgen_model_light
        if self.custom_pgen_model_heavy is None: main_folder_heavy = os.path.join(os.path.dirname(sonnia.__file__), 'default_models', self.chain_type_heavy)
        else: main_folder_heavy=self.custom_pgen_model_heavy

        marginals_file_name = os.path.join(main_folder_light,'model_marginals.txt')
        params_file_name = os.path.join(main_folder_light,'model_params.txt')
        V_anchor_pos_file = os.path.join(main_folder_light,'V_gene_CDR3_anchors.csv')
        J_anchor_pos_file = os.path.join(main_folder_light,'J_gene_CDR3_anchors.csv')
        self.genomic_data_light = load_model.GenomicDataVJ()
        self.genomic_data_light.load_igor_genomic_data(params_file_name, V_anchor_pos_file, J_anchor_pos_file)
        self.generative_model_light = load_model.GenerativeModelVJ()
        self.generative_model_light.load_and_process_igor_model(marginals_file_name)        
        self.seqgen_model_light = seq_gen.SequenceGenerationVJ(self.generative_model_light, self.genomic_data_light)
        self.pgen_model_light = generation_probability.GenerationProbabilityVJ(self.generative_model_light, self.genomic_data_light)

        params_file_name = os.path.join(main_folder_heavy,'model_params.txt')
        V_anchor_pos_file = os.path.join(main_folder_heavy,'V_gene_CDR3_anchors.csv')
        J_anchor_pos_file = os.path.join(main_folder_heavy,'J_gene_CDR3_anchors.csv')
        marginals_file_name = os.path.join(main_folder_heavy,'model_marginals.txt')
        self.genomic_data_heavy= load_model.GenomicDataVDJ()
        self.genomic_data_heavy.load_igor_genomic_data(params_file_name, V_anchor_pos_file, J_anchor_pos_file)
        self.generative_model_heavy = load_model.GenerativeModelVDJ()
        self.generative_model_heavy.load_and_process_igor_model(marginals_file_name)    
        self.seqgen_model_heavy = seq_gen.SequenceGenerationVDJ(self.generative_model_heavy, self.genomic_data_heavy)    
        self.pgen_model_heavy = generation_probability.GenerationProbabilityVDJ(self.generative_model_heavy, self.genomic_data_heavy)
        
        if recompute_norm:
            self.norm_light= self.pgen_model_light.compute_regex_CDR3_template_pgen('CX{0,}')
            self.norm_heavy= self.pgen_model_heavy.compute_regex_CDR3_template_pgen('CX{0,}')
        else:
            norms={'human_T_beta':0.2442847269027897,'human_T_alpha':0.2847166577727317,'human_B_heavy': 0.1499265655936305, 
                'human_B_lambda':0.29489499727399304, 'human_B_kappa':0.29247125650320943, 'mouse_T_beta':0.2727148540013573,'mouse_T_alpha':0.321870924914448}
            self.norm_light= norms[self.chain_type_light]
            self.norm_heavy=  norms[self.chain_type_heavy]

        self.norm_productive=self.norm_heavy*self.norm_light
            
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
        seq1,invseq=seq[0],seq[0][::-1]
        seq_feature_lsts = [['l_h' + str(len(seq1))]]
        for i in range(len(seq1)):
            seq_feature_lsts += [['a_h' + seq1[i] + str(i)],['a_h' + invseq[i] + str(-1-i)]]
        seq1,invseq=seq[3],seq[3][::-1]
        seq_feature_lsts += [['l_l' + str(len(seq1))]]
        for i in range(len(seq1)):
            seq_feature_lsts += [['a_l' + seq1[i] + str(i)],['a_l' + invseq[i] + str(-1-i)]]
        vh='v_h' + gene_to_num_str(seq[1],'V')[1:]
        jh='j_h' + gene_to_num_str(seq[2],'J')[1:]
        vl='v_l' + gene_to_num_str(seq[4],'V')[1:]
        jl='j_l' + gene_to_num_str(seq[5],'J')[1:]
        seq_feature_lsts += [[vh,jh],[vh],[jh],[vl,jl],[vl],[jl]]
        return list(set([self.feature_dict[tuple(f)] for f in seq_feature_lsts if tuple(f) in self.feature_dict]))

        
    def generate_sequences_pre(self, num_seqs = 1,nucleotide=False):
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
        seqs_generated=[self.seqgen_model_light.gen_rnd_prod_CDR3(conserved_J_residues='ABCEDFGHIJKLMNOPQRSTUVWXYZ') for i in tqdm(range(int(num_seqs)))]
        seqs_light= [[seq[1], self.genomic_data_light.genV[seq[2]][0].split('*')[0], self.genomic_data_light.genJ[seq[3]][0].split('*')[0],seq[0]] for seq in seqs_generated]

        seqs_generated=[self.seqgen_model_heavy.gen_rnd_prod_CDR3(conserved_J_residues='ABCEDFGHIJKLMNOPQRSTUVWXYZ') for i in tqdm(range(int(num_seqs)))]
        seqs_heavy= [[seq[1], self.genomic_data_heavy.genV[seq[2]][0].split('*')[0], self.genomic_data_heavy.genJ[seq[3]][0].split('*')[0],seq[0]] for seq in seqs_generated]

        seqs= [[seqs_heavy[i][0],seqs_heavy[i][1],seqs_heavy[i][2],
                seqs_light[i][0],seqs_light[i][1],seqs_light[i][2],
                seqs_heavy[i][3],seqs_light[i][3]] for i in range(int(num_seqs))]

        if nucleotide: return np.array(seqs)
        else: return np.array(seqs)[:,:-2]
        
    def compute_all_pgens(self,seqs,include_genes=True):
        '''
        Compute Pgen of sequences using OLGA
        '''
        #Load OLGA for seq pgen estimation

        # every process needs to access this vector, for sure there is a smarter way to implement this.
        final_models_heavy= [self.pgen_model_heavy for i in range(len(seqs))]
        final_models_light = [self.pgen_model_light for i in range(len(seqs))]
        if include_genes:
            pool = mp.Pool(processes=self.processes)
            f1=pool.map(compute_pgen_expand_heavy, zip(seqs,final_models_heavy))
            pool.close()
            pool = mp.Pool(processes=self.processes)
            f2=pool.map(compute_pgen_expand_light, zip(seqs,final_models_light))
            pool.close()
            return np.array(f1)*np.array(f2)
        else:
            pool = mp.Pool(processes=self.processes)
            f1=pool.map(compute_pgen_expand_novj_heavy, zip(seqs,final_models_light))
            pool.close()
            pool = mp.Pool(processes=self.processes)
            f2=pool.map(compute_pgen_expand_novj_heavy, zip(seqs,final_models_light))
            pool.close()
            return np.array(f1)*np.array(f2)

    def set_gauge(self):
        '''
        placeholder for gauges.
        '''
        pass

    def _save_pgen_model(self,save_dir):
        import shutil
        try:
            if self.custom_pgen_model_light is None: main_folder = os.path.join(os.path.dirname(sonnia.__file__), 'default_models', self.chain_type_light)
            else: main_folder=self.custom_pgen_model_light
        except:
            main_folder = os.path.join(os.path.dirname(sonnia.__file__), 'default_models', self.chain_type)

        light_chain_dir=os.path.join(save_dir,'light_chain')

        if not os.path.isdir(light_chain_dir):os.mkdir(light_chain_dir)
        shutil.copy2(os.path.join(main_folder,'model_params.txt'),light_chain_dir)
        shutil.copy2(os.path.join(main_folder,'model_marginals.txt'),light_chain_dir)
        shutil.copy2(os.path.join(main_folder,'V_gene_CDR3_anchors.csv'),light_chain_dir)
        shutil.copy2(os.path.join(main_folder,'J_gene_CDR3_anchors.csv'),light_chain_dir)
        try:
            if self.custom_pgen_model_heavy is None: main_folder = os.path.join(os.path.dirname(sonnia.__file__), 'default_models', self.chain_type_heavy)
            else: main_folder=self.custom_pgen_model_heavy
        except:
            main_folder = os.path.join(os.path.dirname(sonnia.__file__), 'default_models', self.chain_type)
        heavy_chain_dir=os.path.join(save_dir,'heavy_chain')
        if not os.path.isdir(heavy_chain_dir):os.mkdir(heavy_chain_dir)
        shutil.copy2(os.path.join(main_folder,'model_params.txt'),heavy_chain_dir)
        shutil.copy2(os.path.join(main_folder,'model_marginals.txt'),heavy_chain_dir)
        shutil.copy2(os.path.join(main_folder,'V_gene_CDR3_anchors.csv'),heavy_chain_dir)
        shutil.copy2(os.path.join(main_folder,'J_gene_CDR3_anchors.csv'),heavy_chain_dir)
    
    def _load_pgen_model(self,load_dir):
        self.custom_pgen_model_light=os.path.join(load_dir,'light_chain')
        self.custom_pgen_model_heavy=os.path.join(load_dir,'heavy_chain')
        self.define_models(recompute_norm=False)

def compute_pgen_expand_light(x):
    return x[1].compute_aa_CDR3_pgen(x[0][3],x[0][4],x[0][5])

def compute_pgen_expand_heavy(x):
    return x[1].compute_aa_CDR3_pgen(x[0][0],x[0][1],x[0][2])

def compute_pgen_expand_novj_light(x):
    return x[1].compute_aa_CDR3_pgen(x[0][3])

def compute_pgen_expand_novj_heavy(x):
    return x[1].compute_aa_CDR3_pgen(x[0][0])