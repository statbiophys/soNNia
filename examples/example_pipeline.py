#!/usr/bin/env python
# coding: utf-8


import os
import sonia
from sonnia.sonnia import SoNNia
from sonia.plotting import Plotter
from sonia.evaluate_model import EvaluateModel
from sonia.sequence_generation import SequenceGeneration
from sonnia.processing import Processing
import numpy as np
import pandas as pd


# # load lists of sequences with gene specification
# this assume data sequences are in semi-colon separated text file, with gene specification
data_seqs = pd.read_csv('data_seqs.csv.gz')

# preprocess data
processor=Processing(chain_type='humanTRB')
filtered=processor.filter_dataframe(data_seqs)

data_seqs=list(filtered.values.astype(np.str))
print(data_seqs[:3])

# initialize model
qm = SoNNia(data_seqs=data_seqs,chain_type='humanTRB')

# add generated sequences (you can add them from file too, more is better.)
qm.add_generated_seqs(int(5e5)) 

#train model
qm.infer_selection(epochs=30,batch_size=int(1e4))


# # do some plotting
plot_sonia=Plotter(qm)
plot_sonia.plot_model_learning()
plot_sonia.plot_vjl()
plot_sonia.plot_logQ()


# # Generate sequences
gn=SequenceGeneration(qm)

#generate from pgen
pre_seqs=gn.generate_sequences_pre(int(1e4))
print(pre_seqs[:3])

#generate from ppost
post_seqs=gn.generate_sequences_post(int(1e4))
print(post_seqs[:3])


# # Evaluate sequences
ev=EvaluateModel(qm)
Q_data,pgen_data,ppost_data=ev.evaluate_seqs(qm.data_seqs[:int(1e4)])
Q_gen,pgen_gen,ppost_gen=ev.evaluate_seqs(pre_seqs)
Q_model,pgen_model,ppost_model=ev.evaluate_seqs(post_seqs)
print(Q_model[:3]),
print(pgen_model[:3])
print(ppost_model[:3])


plot_sonia.plot_prob(data=pgen_data,gen=pgen_gen,model=pgen_model,ptype='P_{pre}')
plot_sonia.plot_prob(ppost_data,ppost_gen,ppost_model,ptype='P_{post}')
plot_sonia.plot_prob(Q_data,Q_gen,Q_model,ptype='Q',bin_min=-4,bin_max=2)


# # some utils inherited from OLGA
# olga classes can be directly inspected in the main SoNNia model

print(qm.seq_gen_model.gen_rnd_prod_CDR3())
print(qm.genomic_data.genJ[1])
print(qm.pgen_model.PinsDJ)

