#!/usr/bin/env python
"""Command line script to infer sonia model.

    Copyright (C) 2020 Isacchini Giulio

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>.

This program will infer a seleciton model
"""

import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

from optparse import OptionParser

import numpy as np
import olga.generation_probability as generation_probability
import olga.load_model as olga_load_model
from tqdm import tqdm

import sonnia.sonnia
from sonnia.sonia import Sonia
from sonnia.sonnia import SoNNia
from sonnia.utils import gene_to_num_str
import pandas as pd
from sonnia.plotting import Plotter

def main():
    """Evaluate sequences."""
    parser = OptionParser(conflict_handler="resolve")
    models=[['humanTRA','human_T_alpha'],
     ['humanTRB','human_T_beta'],
     ['humanIGH','human_B_heavy'],
     ['humanIGK','human_B_kappa'],
     ['humanIGL','human_B_lambda'],
     ['mouseTRA','mouse_T_alpha'],
     ['mouseTRB','mouse_T_beta'],
     ['mouseIGH','mouse_B_heavy']]
    for model in models:
        parser.add_option(
            "--" + model[0],
            "--" + model[1],
            action="store_true",
            dest=model[1],
            default=False,
            help="use default " + model[0] + " model",
        )

    parser.add_option(
        "--custom_model",
        dest="custom_model",
        metavar="PATH/TO/FOLDER/",
        help="specify PATH/TO/FOLDER/ for a custom generative model",
        default=None,
    )

    parser.add_option(
        "--epochs",
        type="int",
        default=100,
        dest="epochs",
        help="number of epochs for inference, default is 30",
    )
    parser.add_option(
        "--batch_size",
        type="int",
        default=5000,
        dest="batch_size",
        help="size of batch for the stochastic gradient descent",
    )
    parser.add_option(
        "--validation_split",
        type="float",
        default=0.1,
        dest="validation_split",
        help="fraction of sequences used for validation.",
    )
    parser.add_option(
        "--gene_features",
        dest="gene_features",
        default=None,
        help="Define gene features. Default is 'joint_vj' for linear model and 'indep_vj' for deep model. Options: 'joint_vj', 'indep_vj', 'v', 'j', 'none', 'vjl'.",
    )

    parser.add_option(
        "--linear",
        action="store_true",
        dest="linear_model",
        default=False,
        help="Join gene features.",
    )

    # input output
    parser.add_option(
        "-i",
        "--infile",
        dest="infile_name",
        metavar="PATH/TO/FILE",
        help="read in CDR3 sequences (and optionally V/J masks) from PATH/TO/FILE",
    )
    parser.add_option(
        "-o",
        "--outfile",
        dest="outfile_name",
        metavar="PATH/TO/FILE",
        help="write CDR3 sequences and pgens to PATH/TO/FILE",
    )
    parser.add_option(
        "-m",
        "--max_number_of_seqs",
        type="int",
        metavar="N",
        dest="max_number_of_seqs",
        help="evaluate for at most N sequences.",
    )
    parser.add_option(
        "-n",
        "--n_gen_seqs",
        type="int",
        metavar="N",
        dest="n_gen_seqs",
        default=0,
        help="sample n sequences from gen distribution.",
    )
    parser.add_option(
        "-g",
        "--infile_gen",
        dest="infile_gen",
        metavar="PATH/TO/FILE",
        help="read generated CDR3 sequences (and optionally V/J masks) from PATH/TO/FILE",
    )
    
    parser.add_option(
        "--no_report",
        "--no_plot_report",
        action="store_false",
        dest="plot_report",
        default=True,
        help="Do not produce report plots of the inferred model.",
    )

    # delimeters
    parser.add_option(
        "-d",
        "--delimiter",
        type="choice",
        dest="delimiter",
        choices=["tab", "space", ",", ";", ":"],
        help="declare infile delimiter. Default is tab for .tsv input files, comma for .csv files, and any whitespace for all others. Choices: 'tab', 'space', ',', ';', ':'",
    )
    
    parser.add_option(
        "--seed",
        type="int",
        metavar="N",
        dest="seed",
        default=None,
        help="set seed for inference",
    )

    (options, args) = parser.parse_args()

    # set seed
    if options.seed is not None:
        import tensorflow as tf

        np.random.seed(options.seed)
        tf.random.set_seed(options.seed)

    if options.gene_features is None:
        if options.linear_model:
            gene_features = "joint_vj"
        else:
            gene_features = "indep_vj"
        joint_genes = True
    else:
        gene_features = options.gene_features
    
    main_folder = os.path.dirname(sonnia.__file__)
    default_model_list=[
            s 
            for s in os.listdir(os.path.join(main_folder, "default_models")) 
            if not '.' in s and len(s.split('_'))==3
        ]
    model_folders=[
            x
            for x in default_model_list
            if getattr(options, x)
        ]
    if options.custom_model is not None:
        model_folders.append(options.custom_model)
    num_models_specified = len(model_folders)
    if num_models_specified == 0:
        print("Need to indicate generative model.")
        print("Exiting...")
        return -1
    elif num_models_specified > 1:
        print("Only specify one model")
        print("Exiting...")
        return -1
    else:
        model_folder=model_folders[0]

    if options.infile_name is not None:
        infile_name = options.infile_name
        if not os.path.isfile(infile_name):
            print("Cannot find input file: " + infile_name)
            print("Exiting...")
            return -1

    if options.outfile_name is not None:
        outfile_name = options.outfile_name

    # Parse delimiter
    junction_column='junction_aa'
    delimiter = options.delimiter
    if delimiter is None:  # Default case
        if options.infile_name is None:
            delimiter = "\t"
        elif ".tsv" in infile_name:  # parse TAB separated value file
            delimiter = "\t"
        elif ".csv" in infile_name:  # parse COMMA separated value file
            delimiter = ","
            junction_column='amino_acid'
    else:
        try:
            delimiter = {"tab": "\t", "space": " ", ",": ",", ";": ";", ":": ":"}[
                delimiter
            ]
        except KeyError:
            pass  # Other string passed as the delimiter.
    
    data_seqs = pd.read_csv(infile_name, delimiter=delimiter)[[junction_column,'v_gene','j_gene']].values
    print('Succesfully loaded ',len(data_seqs),'sequences')
    # define number of gen_seqs:
    gen_seqs = []
    n_gen_seqs = options.n_gen_seqs
    generate_sequences = False
    if options.infile_gen is None:
        generate_sequences = True
        if n_gen_seqs == 0:
            n_gen_seqs = len(data_seqs)#np.max([int(5e5), 3 * len(data_seqs)])
    else:
        gen_seqs = pd.read_csv(options.infile_gen, delimiter=delimiter)[[junction_column,'v_gene','j_gene']].values
    
    # combine sequences.
    print("Initialise Model.")

    # choose sonia model type
    if options.linear_model:
        sonia_model = Sonia(
            data_seqs=data_seqs,
            gen_seqs=gen_seqs,
            pgen_model=model_folder,
            gene_features=gene_features,
        )
    else:
        sonia_model = SoNNia(
            data_seqs=data_seqs,
            gen_seqs=gen_seqs,
            pgen_model=model_folder,
            gene_features=gene_features,
        )

    if generate_sequences:
        sonia_model.add_generated_seqs(n_gen_seqs)

    print("Model initialised. Start inference")
    sonia_model.infer_selection(
        epochs=options.epochs,
        verbose=1,
        batch_size=options.batch_size,
        validation_split=options.validation_split,
    )
    print("Save Model")
    if options.outfile_name is not None:  # OUTFILE SPECIFIED
        name_out = options.outfile_name
    else:
        name_out = "sonnia_model"
    sonia_model.save_model(name_out)

    if options.plot_report:
        pl = Plotter(sonia_model)
        pl.plot_model_learning(os.path.join(name_out, "model_learning.png"))
        pl.plot_vjl(os.path.join(name_out, "marginals.png"))
        pl.plot_logQ(os.path.join(name_out, "log_Q.png"))
        pl.plot_ratioQ(os.path.join(name_out, "Q_ratio.png"))


if __name__ == "__main__":
    main()