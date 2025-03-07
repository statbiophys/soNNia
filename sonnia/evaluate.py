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

    # specify model
    parser.add_option(
        "--humanTRA",
        "--human_T_alpha",
        action="store_true",
        dest="human_T_alpha",
        default=False,
        help="use default human TRA model (T cell alpha chain)",
    )
    parser.add_option(
        "--humanTRB",
        "--human_T_beta",
        action="store_true",
        dest="human_T_beta",
        default=False,
        help="use default human TRB model (T cell beta chain)",
    )
    parser.add_option(
        "--humanIGH",
        "--human_B_heavy",
        action="store_true",
        dest="human_B_heavy",
        default=False,
        help="use default human IGH model (B cell heavy chain)",
    )
    parser.add_option(
        "--humanIGK",
        "--human_B_kappa",
        action="store_true",
        dest="human_B_kappa",
        default=False,
        help="use default human IGK model (B cell light kappa chain)",
    )
    parser.add_option(
        "--humanIGL",
        "--human_B_lambda",
        action="store_true",
        dest="human_B_lambda",
        default=False,
        help="use default human IGL model (B cell light lambda chain)",
    )
    parser.add_option(
        "--mouseTRA",
        "--mouse_T_alpha",
        action="store_true",
        dest="mouse_T_alpha",
        default=False,
        help="use default mouse TRA model (T cell alpha chain)",
    )
    parser.add_option(
        "--mouseTRB",
        "--mouse_T_beta",
        action="store_true",
        dest="mouse_T_beta",
        default=False,
        help="use default mouse TRB model (T cell beta chain)",
    )
    parser.add_option(
        "--mouseIGH",
        "--mouse_B_heavy",
        action="store_true",
        dest="mouse_B_heavy",
        default=False,
        help="use default mouse IGH model (B cell heavy chain)",
    )

    parser.add_option(
        "--set_custom_model_VDJ",
        dest="vdj_model_folder",
        metavar="PATH/TO/FOLDER/",
        help="specify PATH/TO/FOLDER/ for a custom VDJ generative model",
        default=None,
    )
    parser.add_option(
        "--set_custom_model_VJ",
        dest="vj_model_folder",
        metavar="PATH/TO/FOLDER/",
        help="specify PATH/TO/FOLDER/ for a custom VJ generative model",
        default=None,
    )
    parser.add_option(
        "--ppost",
        "--Ppost",
        action="store_true",
        dest="ppost",
        default=False,
        help="compute Ppost, also computes pgen and Q",
    )
    parser.add_option(
        "--pgen",
        "--Pgen",
        action="store_true",
        dest="pgen",
        default=False,
        help="compute pgen",
    )
    parser.add_option(
        "--Q",
        "--selection_factor",
        action="store_true",
        dest="Q",
        default=False,
        help="compute Q",
    )
    parser.add_option(
        "--recompute_productive_norm",
        "--compute_norm",
        action="store_true",
        dest="recompute_productive_norm",
        default=False,
        help="recompute productive normalization",
    )
    parser.add_option(
        "--skip_off",
        "--skip_empty_off",
        action="store_true",
        dest="skip_empty",
        default=True,
        help="stop skipping empty or blank sequences/lines (if for example you want to keep line index fidelity between the infile and outfile).",
    )
    parser.add_option(
        "-s",
        "--chunk_size",
        type="int",
        metavar="N",
        dest="chunck_size",
        default=mp.cpu_count() * int(5e2),
        help="Number of sequences to evaluate at each iteration",
    )
    parser.add_option(
        "--linear",
        action="store_true",
        dest="linear_model",
        default=False,
        help="Join gene features.",
    )

    # vj genes
    parser.add_option(
        "--v_in",
        "--v_mask_index",
        type="int",
        metavar="INDEX",
        dest="V_mask_index",
        default=1,
        help="specifies V_masks are found in column INDEX in the input file. Default is 1.",
    )
    parser.add_option(
        "--j_in",
        "--j_mask_index",
        type="int",
        metavar="INDEX",
        dest="J_mask_index",
        default=2,
        help="specifies J_masks are found in column INDEX in the input file. Default is 2.",
    )
    parser.add_option(
        "--v_mask",
        type="string",
        dest="V_mask",
        help="specify V usage to condition as arguments.",
    )
    parser.add_option(
        "--j_mask",
        type="string",
        dest="J_mask",
        help="specify J usage to condition as arguments.",
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
        "--seq_in",
        "--seq_index",
        type="int",
        metavar="INDEX",
        dest="seq_in_index",
        default=0,
        help="specifies sequences to be read in are in column INDEX. Default is index 0 (the first column).",
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
        "--lines_to_skip",
        type="int",
        metavar="N",
        dest="lines_to_skip",
        default=0,
        help="skip the first N lines of the file. Default is 0.",
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
        "--raw_delimiter",
        type="str",
        dest="delimiter",
        help="declare infile delimiter as a raw string.",
    )
    parser.add_option(
        "--delimiter_out",
        type="choice",
        dest="delimiter_out",
        choices=["tab", "space", ",", ";", ":"],
        help="declare outfile delimiter. Default is tab for .tsv output files, comma for .csv files, and the infile delimiter for all others. Choices: 'tab', 'space', ',', ';', ':'",
    )
    parser.add_option(
        "--raw_delimiter_out",
        type="str",
        dest="delimiter_out",
        help="declare for the delimiter outfile as a raw string.",
    )
    parser.add_option(
        "--gene_mask_delimiter",
        type="choice",
        dest="gene_mask_delimiter",
        choices=["tab", "space", ",", ";", ":"],
        help="declare gene mask delimiter. Default comma unless infile delimiter is comma, then default is a semicolon. Choices: 'tab', 'space', ',', ';', ':'",
    )
    parser.add_option(
        "--raw_gene_mask_delimiter",
        type="str",
        dest="gene_mask_delimiter",
        help="declare delimiter of gene masks as a raw string.",
    )
    parser.add_option(
        "--comment_delimiter",
        type="str",
        dest="comment_delimiter",
        help="character or string to indicate comment or header lines to skip.",
    )

    (options, args) = parser.parse_args()

    # Check that the model is specified properly
    main_folder = os.path.dirname(sonnia.sonnia.__file__)
    default_model_list=[
            s 
            for s in os.listdir(os.path.join(main_folder, "default_models")) 
            if not '.' in s and len(s.split('_'))==3
        ]
    print(default_model_list)
    model_folders=[
            x
            for x in default_model_list
            if getattr(options, x)
        ]
    if options.vdj_model_folder is not None:
        model_folders.append(options.vdj_model_folder)
    if options.vj_model_folder is not None:
        model_folders.append(options.vj_model_folder)
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
    print(model_folder)
    try:
        sonia_model = SoNNia(ppost_model=model_folder)
    except:
        sonia_model = Sonia(ppost_model=model_folder)

    if options.recompute_productive_norm:
        print("Recompute productive normalization.")
        sonia_model.norm_productive = pgen_model.compute_regex_CDR3_template_pgen("CX{0,}")

    print("Load file")
    data_seqs = pd.read_csv(infile_name, delimiter=delimiter)[[junction_column,'v_gene','j_gene']].values
    
    print("Evaluate")
    sonia_model.update_model(add_data_seqs=data_seqs)
    Q, pgen, ppost = sonia_model.evaluate_seqs(sonia_model.data_seqs)
    df_out=pd.DataFrame(sonia_model.data_seqs,columns=['junction_aa','v_gene','j_gene'])
    df_out['Q']=Q
    df_out['Pgen']=pgen
    df_out['Ppost']=ppost

    print('Save output to',outfile_name)
    df_out.to_csv(outfile_name,sep='\t',index=False)

if __name__ == "__main__":
    main()
