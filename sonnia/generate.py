#!/usr/bin/env python
"""Command line script to generate sequences.

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

This program generates sequences
"""

import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

from optparse import OptionParser

from tqdm import tqdm

import sonnia.sonnia
from sonnia.sonia import Sonia
from sonnia.sonnia import SoNNia
import pandas as pd

def chuncks(n, size):
    if n % size:
        return int(n / size) * [size] + [n % size]
    else:
        return int(n / size) * [size]


def main():
    """Generate sequences."""
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
        "--post",
        "--ppost",
        action="store_true",
        dest="ppost",
        default=False,
        help="sample from post selected repertoire",
    )
    parser.add_option(
        "--pre",
        "--pgen",
        action="store_true",
        dest="pgen",
        default=False,
        help="sample from pre selected repertoire ",
    )
    
    parser.add_option(
        "-s",
        "--chunk_size",
        type="int",
        metavar="N",
        dest="chunck_size",
        default=int(1e3),
        help="Number of sequences to generate at each iteration",
    )
    parser.add_option(
        "-r",
        "--rejection_bound",
        type="int",
        metavar="N",
        dest="rejection_bound",
        default=10,
        help="limit above which sequences are always accepted.",
    )

    # input output
    parser.add_option(
        "-o",
        "--outfile",
        dest="outfile_name",
        metavar="PATH/TO/FILE",
        help="write CDR3 sequences to PATH/TO/FILE",
    )
    parser.add_option(
        "-n",
        "--N",
        type="int",
        metavar="N",
        dest="num_seqs_to_generate",
        default=1,
        help="Number of sequences to sample from.",
    )

    (options, args) = parser.parse_args()

    # Check that the model is specified properly
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

    if options.pgen:
        sonia_model = Sonia(pgen_model=model_folder)
    else:
        try:
            sonia_model = SoNNia(ppost_model=model_folder)
        except:
            sonia_model = Sonia(ppost_model=model_folder)

    if options.outfile_name is not None:  # OUTFILE SPECIFIED
        if options.pgen:
            seqs = sonia_model.generate_sequences_pre(
                num_seqs=options.num_seqs_to_generate, nucleotide=True
            )
        elif options.ppost:
            seqs = sonia_model.generate_sequences_post(
                num_seqs=options.num_seqs_to_generate, nucleotide=True, upper_bound=options.rejection_bound
            )
        else:
            print("ERROR: give option between --pre or --post")
            return -1
        pd.DataFrame(seqs,columns=['junction_aa','v_gene','j_gene','junction']).to_csv(options.outfile_name,sep='\t',index=False)
    else:  # print to stdout
        to_generate = chuncks(options.num_seqs_to_generate, options.chunck_size)
        for t in to_generate:
            if options.pgen:
                seqs = sonia_model.generate_sequences_pre(num_seqs=t, nucleotide=True)
            elif options.ppost:
                seqs = sonia_model.generate_sequences_post(
                    num_seqs=t, nucleotide=True, upper_bound=options.rejection_bound
                )
            else:
                print("ERROR: give option between --pre or --post")
                return -1
            print('junction_aa','v_gene','j_gene','junction')
            for seq in seqs:
                print(seq[0], seq[1], seq[2], seq[3])


if __name__ == "__main__":
    main()
