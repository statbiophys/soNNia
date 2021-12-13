import numpy as np 
import pandas as pd
from sonnia.sonnia import SoNNia
from sonia.evaluate_model import EvaluateModel
from sonia.plotting import Plotter
from scipy import stats
import matplotlib.pyplot as plt
import re
import tensorflow.keras.backend as K
import tensorflow as tf
import random as rn
import olga.load_model as olga_load_model
import olga.generation_probability as pgen
import os
import olga.sequence_generation as seq_gen
from sonnia.utils import sample_olga

# load olga pgen model and compute normalization for productivity
main_folder='universal_model/'
params_file_name = os.path.join(main_folder,'model_params.txt')
marginals_file_name = os.path.join(main_folder,'model_marginals.txt')
V_anchor_pos_file = os.path.join(main_folder,'V_gene_CDR3_anchors.csv')
J_anchor_pos_file = os.path.join(main_folder,'J_gene_CDR3_anchors.csv')
genomic_data = olga_load_model.GenomicDataVDJ()
genomic_data.load_igor_genomic_data(params_file_name, V_anchor_pos_file, J_anchor_pos_file)
generative_model = olga_load_model.GenerativeModelVDJ()
generative_model.load_and_process_igor_model(marginals_file_name)
pgen_model = pgen.GenerationProbabilityVDJ(generative_model, genomic_data)
norm=pgen_model.compute_regex_CDR3_template_pgen('CX{0,}')


# load generated seqs and training data
gen=list(pd.read_csv('sampled_data/generated_sequences.csv.gz').sample(frac=1).values)
data=list(pd.read_csv('sampled_data/train_data.csv.gz')[['amino_acid','v_gene','j_gene']].sample(frac=1).reset_index(drop=True).values)
print (len(gen),len(data))          

# define parameters of inference
batch_size=int(2e4)
epochs_NN=300

#define and train NN model
qm = SoNNia(data_seqs=data,gen_seqs=gen,custom_pgen_model=main_folder,min_energy_clip=-7,l2_reg=1e-4)
qm.infer_selection(epochs=epochs_NN,batch_size=batch_size,validation_split=0.1,verbose=1)
qm.norm_productive=norm
qm.save_model('selection_models/emerson_NN',attributes_to_save = ['model', 'log'])
pl=Plotter(qm)
pl.plot_model_learning(save_name='selection_models/emerson_NN/training.png')
pl.plot_vjl(save_name='selection_models/emerson_NN/marginals.png')
pl.plot_logQ(save_name='selection_models/emerson_NN/logQ.png')
pl.plot_ratioQ(save_name='selection_models/emerson_NN/ratioQ.png')

#define and train linear model (you don't need too much data for this)
qm = SoNNia(data_seqs=data[:int(1e6)],gen_seqs=gen[:int(2e6)],custom_pgen_model=main_folder,
            deep=False,min_energy_clip=-7,include_indep_genes = False, include_joint_genes = True,
            l2_reg=1e-4,l1_reg=1e-5)
qm.infer_selection(epochs=50,validation_split=0.1,batch_size=batch_size,verbose=1)
qm.norm_productive=norm
qm.save_model('selection_models/emerson_linear',attributes_to_save = ['model', 'log'])
pl=Plotter(qm)
pl.plot_model_learning(save_name='selection_models/emerson_linear/training.png')
pl.plot_vjl(save_name='selection_models/emerson_linear/marginals.png')
pl.plot_logQ(save_name='selection_models/emerson_linear/logQ.png')
pl.plot_ratioQ(save_name='selection_models/emerson_linear/ratioQ.png')