#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function, division

import numpy as np
import os
import multiprocessing as mp
import olga.load_model as olga_load_model
import olga.generation_probability as pgen
from sonia.evaluate_model import EvaluateModel as ev_model

class EvaluateModel(ev_model):
    """Class used to evaluate sequences with the sonia model: Ppost=Q*Pgen


    Attributes
    ----------
    sonia_model: object
        Sonia model. Loaded previously, do not put the path.

    include_genes: bool
        Conditioning on gene usage for pgen/ppost evaluation. Default: True

    processes: int
        Number of processes to use to infer pgen. Default: all.

    custom_olga_model: object
        Optional: already loaded custom generation_probability olga model.

    Methods
    ----------

    evaluate_seqs(seqs=[])
        Returns Q, pgen and ppost of a list of sequences.

    evaluate_selection_factors(seqs=[])
        Returns normalised selection factor Q (Ppost=Q*Pgen) of a list of sequences (faster than evaluate_seqs because it does not compute pgen and ppost)

    """

    def __init__(self,sonia_model=None,include_genes=True,processes=None,custom_olga_model_heavy=None,custom_olga_model_light=None,chain_type_heavy='human_T_beta',chain_type_light='human_T_alpha'):

        if type(sonia_model)==str:
            print('ERROR: you need to pass a Sonia object')
            return
        elif sonia_model is None:
            print('Initialise default sonia model')
            self.sonia_model=SoniaPaired(custom_olga_model_heavy=None,custom_olga_model_light=None,chain_type_heavy='human_T_beta',chain_type_light='human_T_alpha')
        self.sonia_model=sonia_model

        if processes is None: self.processes = mp.cpu_count()
        else: self.processes = processes
            
        self.include_genes=include_genes
        self.genomic_data_light=self.sonia_model.genomic_data_light
        self.pgen_model_light=self.sonia_model.pgen_model_light
        self.genomic_data_heavy=self.sonia_model.genomic_data_heavy
        self.pgen_model_heavy=self.sonia_model.pgen_model_heavy
            
        self.norm_light= self.pgen_model_light.compute_regex_CDR3_template_pgen('CX{0,}')
        self.norm_heavy= self.pgen_model_heavy.compute_regex_CDR3_template_pgen('CX{0,}')
        self.sonia_model.norm_productive=self.norm_heavy*self.norm_light
        
    def compute_all_pgens(self,seqs):
        '''
        Compute Pgen of sequences using OLGA
        '''
        #Load OLGA for seq pgen estimation

        # every process needs to access this vector, for sure there is a smarter way to implement this.
        final_models_heavy= [self.pgen_model_heavy for i in range(len(seqs))]
        final_models_light = [self.pgen_model_light for i in range(len(seqs))]
        if self.include_genes:
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

def compute_pgen_expand_light(x):
    return x[1].compute_aa_CDR3_pgen(x[0][3],x[0][4],x[0][5])

def compute_pgen_expand_heavy(x):
    return x[1].compute_aa_CDR3_pgen(x[0][0],x[0][1],x[0][2])

def compute_pgen_expand_novj_light(x):
    return x[1].compute_aa_CDR3_pgen(x[0][3])

def compute_pgen_expand_novj_heavy(x):
    return x[1].compute_aa_CDR3_pgen(x[0][0])

