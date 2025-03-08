#!/usr/bin/env python
"""Command line script to evaluate sequences.

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

This program will evaluate pgen and ppost of sequences
"""

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import multiprocessing as mp
from optparse import OptionParser
import olga.generation_probability as generation_probability
import olga.load_model as olga_load_model
from tqdm import tqdm
import sonnia.sonnia
from sonnia.sonia import Sonia
from sonnia.sonnia import SoNNia
from sonnia.utils import gene_to_num_str
import pandas as pd

def chunks(lst, n):
    return [lst[i : i + n] for i in range(0, len(lst), n)]


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
        "--recompute_productive_norm",
        "--compute_norm",
        action="store_true",
        dest="recompute_productive_norm",
        default=False,
        help="recompute productive normalization",
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
        default=int(1e8),
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

    (options, args) = parser.parse_args()

    # Check that the model is specified properly
    main_folder = os.path.dirname(sonnia.sonnia.__file__)
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

    recompute_productive_norm = False

    if options.infile_name is not None:
        infile_name = options.infile_name
        if not os.path.isfile(infile_name):
            print("Cannot find input file: " + infile_name)
            print("Exiting...")
            return -1

    if options.outfile_name is not None:
        outfile_name = options.outfile_name
    else:
        outfile_name='evaluated_seqs.tsv'

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

    # choose sonia model type
    print('Initialize soNNia model',model_folder)
    try:
        sonia_model = SoNNia(ppost_model=model_folder)
    except:
        sonia_model = Sonia(ppost_model=model_folder)

    if options.recompute_productive_norm:
        print("Recompute productive normalization.")
        sonia_model.norm_productive = pgen_model.compute_regex_CDR3_template_pgen("CX{0,}")

    print("Load file")
    try:
        data_seqs = pd.read_csv(infile_name, delimiter=delimiter)[[junction_column,'v_gene','j_gene']].values
    except:
        data_seqs = pd.read_csv(infile_name, delimiter=delimiter,header=None).values

    print("Evaluate")
    sonia_model.update_model(add_data_seqs=data_seqs[:options.max_number_of_seqs])
    Q, pgen, ppost = sonia_model.evaluate_seqs(sonia_model.data_seqs)
    df_out=pd.DataFrame(sonia_model.data_seqs,columns=['junction_aa','v_gene','j_gene'])
    df_out['Q']=Q
    df_out['Pgen']=pgen
    df_out['Ppost']=ppost

    print('Save output to',outfile_name)
    df_out.to_csv(outfile_name,sep='\t',index=False)

if __name__ == "__main__":
    main()