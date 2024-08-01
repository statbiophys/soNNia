from __future__ import annotations
import inspect
import logging
import os
import subprocess
from typing import *

import numpy as np
from numpy.typing import NDArray
import pandas as pd

import olga.generation_probability as generation_probability
import olga.sequence_generation as sequence_generation
import olga.load_model as olga_load_model
import olga.generation_probability as pgen
import olga.sequence_generation as seq_gen
from olga.utils import nt2aa

logging.getLogger().setLevel(logging.INFO)
logging.basicConfig(format='%(asctime)s: %(message)s')

DEFAULT_CHAIN_TYPES = {
    'humanTRA': 'human_T_alpha', 'human_T_alpha': 'human_T_alpha',
    'humanTRB': 'human_T_beta', 'human_T_beta': 'human_T_beta',
    'humanIGH': 'human_B_heavy', 'human_B_heavy': 'human_B_heavy',
    'humanIGK': 'human_B_kappa', 'human_B_kappa': 'human_B_kappa',
    'humanIGL': 'human_B_lambda', 'human_B_lambda': 'human_B_lambda',
    'mouseTRB': 'mouse_T_beta', 'mouse_T_beta': 'mouse_T_beta',
    'mouseTRA': 'mouse_T_alpha', 'mouse_T_alpha': 'mouse_T_alpha'
}
DEFAULT_CHAIN_TYPES_PAIRED = {
    'humanTCR': 'human_T_beta_alpha', 'humanTRAB': 'human_T_beta_alpha',
    'humanTRBA': 'human_T_beta_alpha', 'human_T_beta_alpha': 'human_T_beta_alpha',
    'humanIGHK': 'human_B_heavy_kappa', 'human_B_heavy_kappa': 'human_B_heavy_kappa',
    'human_B_kappa_heavy': 'human_B_heavy_kappa', 'humanIGHL': 'human_B_heavy_lambda',
    'human_B_heavy_lambda': 'human_B_heavy_lambda', 'human_B_lambda_heavy': 'human_B_heavy_lambda'
}

NORM_PRODUCTIVES = {'human_T_beta': 0.2442847269027897,
                    'human_T_alpha': 0.2847166577727317,
                    'human_B_heavy': 0.1499265655936305,
                    'human_B_lambda': 0.29489499727399304,
                    'human_B_kappa': 0.29247125650320943,
                    'mouse_T_beta': 0.2727148540013573,
                    'mouse_T_alpha': 0.321870924914448}

HEAVY_CHAINS = {'TRB', 'TRD', 'IGH'}
LIGHT_CHAINS = {'TRA', 'TRG', 'IGK', 'IGL', 'IGI'}

CSV_READER_PARAMS = inspect.signature(pd.read_csv).parameters.keys()

def run_terminal(
    string: str
):
    return [i.decode("utf-8").split('\n')
            for i in subprocess.Popen(string, shell=True,
                                      stdout=subprocess.PIPE,
                                      stderr=subprocess.PIPE).communicate()]

def get_model_dir(
    model_dir: str,
    paired: bool = False
) -> str:
    """
    Obtain a model directory if it's a default option otherwise check the
    directory exists.

    Parameters
    ----------
    model_dir : str
        The path to the model directory.
    paired : bool, default False
        Bool specifying whether the desired default model is a paired model.

    Returns
    -------
    model_dir : str
        The path to the model directory.
    """
    if paired:
        default_chain_types = DEFAULT_CHAIN_TYPES_PAIRED
    else:
        default_chain_types = DEFAULT_CHAIN_TYPES

    if model_dir in default_chain_types:
        filedir = os.path.dirname(os.path.abspath(__file__))
        model_dir = default_chain_types[model_dir]
        model_dir = os.path.join(filedir, 'default_models', model_dir)
    elif os.path.isdir(model_dir):
        pass
    else:
        options = f'{list(default_chain_types.keys())}'[1:-1]
        paired_str = 'paired ' if paired else ''
        raise ValueError('The model is neither a directory that exists '
                         f'nor one of the default {paired_str}options ({options}). '
                         f'Try using one of the default options for a {paired_str}model '
                         f'or an existing directory containing a {paired_str}model.')
    return model_dir

def define_pgen_model(
    pgen_model: Optional[str] = None,
    compute_norm: bool = True,
    return_pgen_dir: bool = False,
    return_chain: bool = False,
    return_recomb_type: bool = False
):
    pgen_dir = get_model_dir(pgen_model)

    if pgen_model in DEFAULT_CHAIN_TYPES:
        norm_productive = NORM_PRODUCTIVES[DEFAULT_CHAIN_TYPES[pgen_model]]
    else:
        norm_productive = None

    pgen_files = ('model_params.txt', 'model_marginals.txt',
                  'V_gene_CDR3_anchors.csv', 'J_gene_CDR3_anchors.csv')
    files_in_dir = set(os.listdir(pgen_dir))
    missing_files = set(pgen_files) - files_in_dir

    if len(missing_files) > 0:
        missing_files = f'{missing_files}'[1:-1]
        raise RuntimeError('The pgen model cannot be loaded. The following files '
                           f'are missing: {missing_files}.')

    params_file_name = os.path.join(pgen_dir, pgen_files[0])
    marginals_file_name = os.path.join(pgen_dir, pgen_files[1])
    V_anchor_pos_file = os.path.join(pgen_dir, pgen_files[2])
    J_anchor_pos_file = os.path.join(pgen_dir, pgen_files[3])

    chains = []
    for fi, gene in zip((V_anchor_pos_file, J_anchor_pos_file), ('V', 'J')):
        chain_tmp = set()
        with open(fi, 'r') as fin:
            # Skip headers.
            next(fin)

            chain = next(fin).partition(',')[0].partition(gene)[0]
            chain_tmp.add(chain)
        if len(chain_tmp) > 1:
            raise RuntimeError(f'{fi} contains anchors for more than one chain: '
                               f'{chain_tmp}. It should contain only {gene} anchors '
                               'for one chain.')
        chains += list(chain_tmp)

    if chains[0] != chains[1]:
        raise RuntimeError('Either the V and J anchor files do not correspond '
                           'to the same chain or the files are not in the expected '
                           'format (gene,anchor_index,function).')

    if chains[0] in HEAVY_CHAINS:
        recomb_type = 'VDJ'
    elif chains[0] in LIGHT_CHAINS:
        recomb_type = 'VJ'
    else:
        heavy_chain_str = f'{HEAVY_CHAINS}'[1:-1]
        light_chain_str = f'{LIGHT_CHAINS}'[1:-1]
        raise RuntimeError(f'Unrecognized chain: {chains[0]}. Recognized heavy chains: '
                           f'{heavy_chain_str} Recognized light chains: {light_chain_str}.')

    genomic_data = getattr(olga_load_model, f'GenomicData{recomb_type}')()
    genomic_data.load_igor_genomic_data(params_file_name,
                                        V_anchor_pos_file,
                                        J_anchor_pos_file)
    generative_model = getattr(olga_load_model, f'GenerativeModel{recomb_type}')()
    generative_model.load_and_process_igor_model(marginals_file_name)
    pgen_model = getattr(pgen, f'GenerationProbability{recomb_type}')(generative_model, genomic_data)
    seqgen_model = getattr(seq_gen, f'SequenceGeneration{recomb_type}')(generative_model, genomic_data)

    if compute_norm and norm_productive is None:
        logging.info('Recomputing productive norm.')
        norm_productive = pgen_model.compute_regex_CDR3_template_pgen('CX{0,}')

    out_tup = (genomic_data, generative_model, pgen_model, seqgen_model, norm_productive)

    if return_pgen_dir:
        out_tup += (pgen_dir,)
    if return_recomb_type:
        out_tup += (recomb_type,)
    if return_chain:
        out_tup += (chains[0],)

    return out_tup

def filter_seqs(
    seqs: Sequence[Sequence[str]] | pd.DataFrame | str,
    model: str | Sonia | SoNNia,
    seq_col: str = 'amino_acid',
    v_col: str = 'v_gene',
    j_col: str = 'j_gene',
    nt_seq_col: Optional[Union[int, str]] = None,
    abundance_col: Optional[Union[int, str]] = None,
    bounds_check: bool = True,
    cdr3_length_check: bool = True,
    conserved_j_residues: str = 'ABCEDFGHIJKLMNOPQRSTUVWXYZ',
    abundance_threshold: int = 0,
    max_cdr3_length: int = 30,
    deduplicate_nt_recombinations: bool = True,
    return_bools: bool = False,
    verbose: bool = True,
    **kwargs: Dict[str, Any]
) -> NDArray[str]:
    """
    Filter the DataFrame for sequences which are productive and compatible with Sonia.

    Parameters
    ----------
    seqs : iterable of iterable of str or str
        If str, this should point to a csv file. Otherwise, seqs must be
        an iterable of iterable of str.:
    model : str or Sonia object or SoNNia object
        The path to the Sonia model, a default Sonia model keyword, or a Sonia object.
    seq_col : str, default 'amino_acid'
        The label of the column pointing to the CDR3 nucleotide sequences.
    v_col : str, default 'v_gene'
        The label of the column pointing to the V genes.
    j_col : str, default 'j_gene'
        The label of the column pointing to the J genes.
    nt_seq_col : str or int, optional
        The label of the column or index of the column which points to the
        CDR3 nucleotide sequences.
    abundance_col : str or int, optional
        The label of the column or index of the column which points to the
        abundance (amount of reads) for each sequence.
    bounds_check : bool, default True
        Check that the sequence begins with a C and ends with one of the given
        conserved J residues.
    cdr3_length_check : bool, default True
        Check that the sequences are smaller than the max_cdr3_length.
    conserved_j_residues : str, default 'ABCEDFGHIJKLMNOPQRSTUVWXYZ'
        The possible residues that a sequence could end with.
    abundance_threshold : int, default 0
        Clones with abundance at or below this threshold are removed.
    max_cdr3_length : int, default 30
        Sequences with CDR3 length longer than this value are removed.
    deduplicate_nt_recombinations : bool, default True
        Deduplicate sequences by nucleotide recombinations to remove PCR amplification
        bias and clonal expansion effects.
    return_bools : bool, default False
        Return a boolean array of whether the sequences are valid.
    verbose: bool, default True
        Print how many sequences remain after each filter.
    **kwargs : dict of {str : any}
        Keyword arguments to pandas.read_csv.

    Returns
    -------
    np.ndarray
       The filtered sequences or a boolean array of what sequences passed.
    """
    for keyword in kwargs:
        if keyword not in CSV_READER_PARAMS:
            raise TypeError(f'{keyword} is an invalid keyword argument for filter_seqs().')

    def get_functional_genes(
        infile: str
    ) -> Set[str]:
        functional_genes = set()
        functional_markers = {'F', '(F)'}
        gene_type = infile.split('/')[-1].partition('_')[0]
        with open(infile, 'r') as fin:
            for line in fin:
                gene, _, func = line.strip().split(',')
                if func in functional_markers:
                    functional_genes.add(gene_to_num_str(gene, gene_type))
        return functional_genes

    if isinstance(model, str):
        model_dir = get_model_dir(model)
    else:
        model_dir = model.pgen_dir

    v_genes = get_functional_genes(os.path.join(model_dir, 'V_gene_CDR3_anchors.csv'))
    j_genes = get_functional_genes(os.path.join(model_dir, 'J_gene_CDR3_anchors.csv'))

    max_v_length = max(len(gene) for gene in v_genes)
    max_j_length = max(len(gene) for gene in j_genes)

    if isinstance(seqs, str):
        df = pd.read_csv(seqs, **kwargs)
    elif isinstance(seqs, pd.DataFrame):
        df = seqs
    else:
        int_tuple = (int, np.integer)
        df = pd.DataFrame(seqs)

        if seq_col is not None:
            if seq_col == 'amino_acid':
                seq_col = 0
                if verbose:
                    logging.info('Using default index (0) for amino acid CDR3 sequences.')
            elif not isinstance(v_col, int_tuple):
                raise TypeError('Because seqs is an iterable, seq_col must '
                                'be an integer pointing to the column containing '
                                'the amino acid CDR3 sequences.')
            else:
                seq_col = seq_col

        if v_col is not None:
            if v_col == 'v_gene':
                v_col = 1
                if verbose:
                    logging.info('Using default index (1) for V genes.')

            elif not isinstance(v_col, int_tuple):
                raise TypeError('Because seqs is an iterable, v_col must '
                                'be an integer pointing to the column containing '
                                'the V genes.')
            else:
                v_col = v_col

        if j_col is not None:
            if j_col == 'j_gene':
                j_col = 2
                if verbose:
                    logging.info('Using default index (2) for J genes.')
            elif not isinstance(j_col, int_tuple):
                raise TypeError('Because seqs is an iterable, j_col must '
                                'be an integer pointing to the column containing '
                                'the J genes.')
            else:
                j_col = j_col

        if nt_seq_col is not None:
            if not isinstance(nt_seq_col, int_tuple):
                raise TypeError('Because seqs is an iterable, nt_seq_col must '
                                'be an integer pointing to the column containing '
                                'the nucleotide CDR3 sequences.')
        if abundance_col is not None:
            if not isinstance(abundance_col, int_tuple):
                raise TypeError('Because seqs is an iterable, abundance_col must '
                                'be an integer pointing to the column containing '
                                'the abundances.')

    if not df[seq_col].str.contains(r'^[ACDEFGHIKLMNPQRSTVWY~_\*]+$', regex=True, na=False).all():
        raise RuntimeError(f'The seq_col pointed to by {seq_col} does '
                           'not contain strings with only amino acids. '
                           'Is this the correct column for amino acid CDR3 sequences?')

    if verbose:
        logging.info(f'{len(df)} sequences before filtering. Using {model_dir} '
                     'for filtering.')

    bool_arr = np.ones(len(df), dtype=bool)
    num_pass = len(df)

    # Convert genes to num_strs.
    df[v_col] = df[v_col].apply(lambda x: gene_to_num_str(x, 'V'))
    df[j_col] = df[j_col].apply(lambda x: gene_to_num_str(x, 'J'))

    if nt_seq_col is not None:
        if not df[nt_seq_col].str.contains('^[ACGTacgt]+$', regex=True, na=False).all():
            raise RuntimeError(f'The nt_seq_col pointed to by {nt_seq_col} does '
                               'not contain strings with only nucleotide characters. '
                               'Is this the correct column for nucleotide CDR3 sequences?')
        if deduplicate_nt_recombinations:
            bool_arr *= ~df.duplicated([nt_seq_col, v_col, j_col]).values
            if verbose:
                num_pass = np.count_nonzero(bool_arr)
                logging.info(f'{num_pass} sequences remain after deduplicating '
                             'nucleotide recombinations.')

        bool_arr[bool_arr] *= df.loc[bool_arr, nt_seq_col].str.len() % 3 == 0
        if verbose:
            num_pass = np.count_nonzero(bool_arr)
            logging.info(f'{num_pass} sequences remain after removing out-of-frame '
                         'nucleotide sequences.')

    if v_col is not None:
        bool_arr[bool_arr] *= df.loc[bool_arr, v_col].isin(v_genes)
        num_pass = np.count_nonzero(bool_arr)
        if num_pass == 0:
            raise RuntimeError('No data are consistent with the V genes used '
                               f'in the model. Does {v_col} point to the column '
                               'containing V genes? Is the model choice correct '
                               f'for the data? The model used: {model_dir}.')
        if verbose:
            logging.info(f'{num_pass} sequences remain after removing sequences '
                         'with V genes inconsistent with the model.')

    if j_col is not None:
        bool_arr[bool_arr] *= df.loc[bool_arr, j_col].isin(j_genes)
        num_pass = np.count_nonzero(bool_arr)
        if num_pass == 0:
            raise RuntimeError('No data are consistent with the J genes used '
                               f'in the model. Does {j_col} point to the column '
                               'containing J genes? Is the model choice correct '
                               f'for the data? The model used: {model_dir}.')
        if verbose:
            logging.info(f'{num_pass} sequences remain after removing sequences '
                         'with J genes inconsistent with the model.')

    bool_arr[bool_arr] *= ~df.loc[bool_arr, seq_col].str.contains(r'\*|_|~', na=True, regex=True)
    if verbose:
        num_pass = np.count_nonzero(bool_arr)
        logging.info(f'{num_pass} sequences remain after removing data which '
                     'are unproductive amino acid sequences.')

    if bounds_check:
        bound_string = '^C.*' + '$|^C.*'.join(list(conserved_j_residues)) + '$'
        bool_arr[bool_arr] *= df.loc[bool_arr, seq_col].str.contains(bound_string, na=False)
        if verbose:
            num_pass = np.count_nonzero(bool_arr)
            logging.info(f'{num_pass} sequences remain after removing sequences '
                         f'that do not begin with a \'C\' or end in a {list(conserved_j_residues)}.')

    if cdr3_length_check:
        bool_arr[bool_arr] *= df.loc[bool_arr, seq_col].str.len() <= max_cdr3_length
        if verbose:
            num_pass = np.count_nonzero(bool_arr)
            logging.info(f'{num_pass} sequences remain after removing sequences '
                         f'with CDR3 length larger than {max_cdr3_length}.')

    if abundance_col is not None:
        bool_arr[bool_arr] *= df.loc[bool_arr, abundance_col] > abundance_threshold
        if verbose:
            num_pass = np.count_nonzero(bool_arr)
            logging.info(f'{num_pass} sequences remain after removing sequences '
                         f'with abundance <= {abundance_threshold}.')

    num_pass = np.count_nonzero(bool_arr)
    if verbose:
        logging.info(f'{num_pass} sequences remain. Filtering completed.')

    if num_pass == 0:
        verbose_str = ''
        if not verbose:
            verbose_str = ('Rerun with verbose = True for more details on how many '
                           'sequences remained after passing through the filters.')
        raise RuntimeError('No sequences passed all the filters. Is something wrong '
                           'with the data? Are any columns given incorrectly? '
                           f'seq_col: {seq_col}, v_col: {v_col}, j_col: {j_col}, '
                           f'nt_seq_col: {nt_seq_col}, abundance_col: {abundance_col}. '
                           +  verbose_str)

    if return_bools:
        return bool_arr

    str_size = max(max_v_length, max_j_length, max_cdr3_length)
    res = np.zeros((num_pass, 3), dtype=f'<U{str_size}')

    res[:, 0] = df.loc[bool_arr, seq_col]
    if v_col is not None:
        res[:, 1] = df.loc[bool_arr, v_col]
    if j_col is not None:
        res[:, 2] = df.loc[bool_arr, j_col]

    return res

def sample_olga(num_gen_seqs=1,custom_model_folder=None,vj=False,chain_type='human_T_beta'):
    (genomic_data, generative_model,
     _, seq_model) = define_pgen_model(custom_model_folder, chain_type, vj)

    #Generate sequences
    return [[seq[1],
             genomic_data.genV[seq[2]][0].split('*')[0],
             genomic_data.genJ[seq[3]][0].split('*')[0]]
            for seq in [sg_model.gen_rnd_prod_CDR3(conserved_J_residues='ABCEDFGHIJKLMNOPQRSTUVWXYZ')
                        for _ in range(int(num_gen_seqs))]]

def correct_olga_heavy(qm):
    # this corrects only the V gene distribution.
    # P(D,J)= P(D|J)P(J)
    old_pv=qm.generative_model.PV
    old_names=qm.pgen_model.V_allele_names
    old_ns=[gene_to_num_str(v,'V') for v in old_names]

    #P(allele|V)
    p_allele_given_v={}
    for v in set(old_ns):
        indices=np.arange(len(old_ns))[np.array(old_ns)==v]
        alleles_probs=np.nan_to_num(old_pv[indices]/np.sum(old_pv[indices]),1)
        for i,j in enumerate(alleles_probs):
            p_allele_given_v[str(i)+'_'+str(indices[i])+'_' +v]=j
    weigths=np.zeros(len(old_names))
    for v in list(p_allele_given_v): weigths[int(v.split('_')[1])]=p_allele_given_v[v]

    # compute probabilities data
    v_data,count_data=np.unique(np.array(qm.data_seqs)[:,1],return_counts=True)
    freq_data=count_data/np.sum(count_data)
    v_data_ns=[gene_to_num_str(v,'V') for v in v_data]
    dict_data=dict(zip(v_data_ns,freq_data))

    new_pv=np.zeros(len(old_ns))
    for i,v in enumerate(old_ns):
        if v in v_data_ns: new_pv[i]=weigths[i]*dict_data[v]

    qm.generative_model.PV=np.array(new_pv)
    qm.seq_gen_model = sequence_generation.SequenceGenerationVDJ(qm.generative_model, qm.genomic_data)
    qm.pgen_model = generation_probability.GenerationProbabilityVDJ(qm.generative_model, qm.genomic_data)

def correct_olga_light(qm):

    old_pvj=qm.generative_model.PVJ
    old_names_v=qm.pgen_model.V_allele_names
    old_names_j=qm.pgen_model.J_allele_names
    old_ns_v=[gene_to_num_str(v,'V') for v in old_names_v]
    old_ns_j=[gene_to_num_str(v,'J') for v in old_names_j]

    #P(alleleV|VJ)
    p_allele_given_v={}
    for v in set(old_ns_v):
        indices=np.arange(len(old_ns_v))[np.array(old_ns_v)==v]
        new_probs=old_pvj[indices].sum(axis=-1)
        alleles_probs=np.nan_to_num(new_probs/np.sum(new_probs),1)
        for i,j in enumerate(alleles_probs):
            p_allele_given_v[str(i)+'_'+str(indices[i])+'_' +v]=j

    weigths_v={}
    for v in list(p_allele_given_v): weigths_v[int(v.split('_')[1])]=p_allele_given_v[v]

    #P(alleleJ|VJ)
    p_allele_given_j={}
    for v in set(old_ns_j):
        indices=np.arange(len(old_ns_j))[np.array(old_ns_j)==v]
        new_probs=old_pvj.T[indices].sum(axis=-1)
        alleles_probs=np.nan_to_num(new_probs/np.sum(new_probs),1)
        for i,j in enumerate(alleles_probs):
            p_allele_given_j[str(i)+'_'+str(indices[i])+'_' +v]=j
    weigths_j={}
    for v in list(p_allele_given_j): weigths_j[int(v.split('_')[1])]=p_allele_given_j[v]

    # compute probabilities data
    VJ_pair= [gene_to_num_str(v,'V')+','+gene_to_num_str(j,'J') for v,j in np.array(qm.data_seqs)[:,1:]]
    vj_data,count_data=np.unique(VJ_pair,return_counts=True)
    freq_data=count_data/np.sum(count_data)
    dict_data=dict(zip(vj_data,freq_data))

    new_pvj=np.zeros(np.array(old_pvj).shape)
    for i,v in enumerate(old_ns_v):
        for k,j in enumerate(old_ns_j):
            if v+','+j in vj_data: new_pvj[i,k]=dict_data[v+','+j]*weigths_v[i]*weigths_j[k]
    new_pvj=new_pvj/np.sum(new_pvj)

    qm.generative_model.PVJ=np.array(new_pvj)
    qm.seq_gen_model = sequence_generation.SequenceGenerationVJ(qm.generative_model, qm.genomic_data)
    qm.pgen_model = generation_probability.GenerationProbabilityVJ(qm.generative_model, qm.genomic_data)


def correct_olga_paired(qm):

    #####################
    ### heavy chain #####
    #####################

    old_pv=qm.generative_model_heavy.PV
    old_names=qm.pgen_model_heavy.V_allele_names
    old_ns=[gene_to_num_str(v,'V') for v in old_names]

    #P(allele|V)
    p_allele_given_v={}
    for v in set(old_ns):
        indices=np.arange(len(old_ns))[np.array(old_ns)==v]
        alleles_probs=np.nan_to_num(old_pv[indices]/np.sum(old_pv[indices]),1)
        for i,j in enumerate(alleles_probs):
            p_allele_given_v[str(i)+'_'+str(indices[i])+'_' +v]=j
    weigths=np.zeros(len(old_names))
    for v in list(p_allele_given_v): weigths[int(v.split('_')[1])]=p_allele_given_v[v]

    # compute probabilities data
    v_data,count_data=np.unique(np.array(qm.data_seqs)[:,1],return_counts=True)
    freq_data=count_data/np.sum(count_data)
    v_data_ns=[gene_to_num_str(v,'V') for v in v_data]
    dict_data=dict(zip(v_data_ns,freq_data))

    new_pv=np.zeros(len(old_ns))
    for i,v in enumerate(old_ns):
        if v in v_data_ns: new_pv[i]=weigths[i]*dict_data[v]

    qm.generative_model_heavy.PV=np.array(new_pv)
    qm.seq_gen_model_heavy = sequence_generation.SequenceGenerationVDJ(qm.generative_model_heavy, qm.genomic_data_heavy)
    qm.pgen_model_heavy = generation_probability.GenerationProbabilityVDJ(qm.generative_model_heavy, qm.genomic_data_heavy)

    #####################
    ### light chain #####
    #####################

    old_pvj=qm.generative_model_light.PVJ
    old_names_v=qm.pgen_model_light.V_allele_names
    old_names_j=qm.pgen_model_light.J_allele_names
    old_ns_v=[gene_to_num_str(v,'V') for v in old_names_v]
    old_ns_j=[gene_to_num_str(v,'J') for v in old_names_j]

    #P(alleleV|VJ)
    p_allele_given_v={}
    for v in set(old_ns_v):
        indices=np.arange(len(old_ns_v))[np.array(old_ns_v)==v]
        new_probs=old_pvj[indices].sum(axis=-1)
        alleles_probs=np.nan_to_num(new_probs/np.sum(new_probs),1)
        for i,j in enumerate(alleles_probs):
            p_allele_given_v[str(i)+'_'+str(indices[i])+'_' +v]=j

    weigths_v={}
    for v in list(p_allele_given_v): weigths_v[int(v.split('_')[1])]=p_allele_given_v[v]

    #P(alleleJ|VJ)
    p_allele_given_j={}
    for v in set(old_ns_j):
        indices=np.arange(len(old_ns_j))[np.array(old_ns_j)==v]
        new_probs=old_pvj.T[indices].sum(axis=-1)
        alleles_probs=np.nan_to_num(new_probs/np.sum(new_probs),1)
        for i,j in enumerate(alleles_probs):
            p_allele_given_j[str(i)+'_'+str(indices[i])+'_' +v]=j
    weigths_j={}
    for v in list(p_allele_given_j): weigths_j[int(v.split('_')[1])]=p_allele_given_j[v]

    # compute probabilities data
    VJ_pair= [gene_to_num_str(v,'V')+','+gene_to_num_str(j,'J') for v,j in np.array(qm.data_seqs)[:,4:]]
    vj_data,count_data=np.unique(VJ_pair,return_counts=True)
    freq_data=count_data/np.sum(count_data)
    dict_data=dict(zip(vj_data,freq_data))

    new_pvj=np.zeros(np.array(old_pvj).shape)
    for i,v in enumerate(old_ns_v):
        for k,j in enumerate(old_ns_j):
            if v+','+j in vj_data: new_pvj[i,k]=dict_data[v+','+j]*weigths_v[i]*weigths_j[k]
    new_pvj=new_pvj/np.sum(new_pvj)
    qm.generative_model_light.PVJ=np.array(new_pvj)
    qm.seq_gen_model_light = sequence_generation.SequenceGenerationVJ(qm.generative_model_light, qm.genomic_data_light)
    qm.pgen_model_light = generation_probability.GenerationProbabilityVJ(qm.generative_model_light, qm.genomic_data_light)

def add_random_error(nt, p):
    """ Take a nucleotide seq then simulate a sequencing
    error on it. Explicitely, each nucleotide has a probability p
    of being randomly modified. Adapted from Thomas Dupic.
    @ Arguments:
    * nt: amino-acid sequence
    * p: the error rate
    """
    rand = np.random.choice(["A", "T", "G", "C"], len(nt))
    return "".join([(a, r)[np.random.random() < p] for a, r in zip(nt, rand)])

def gene_to_num_str(
    gene_name: str,
    gene_type: str
) -> str:
    """
    Strip excess gene name info to number string.

    Parameters
    ----------
    gene_name : str
        Gene or allele name
    gene_type : char
        Genomic cassette type. (i.e. V, D, or J)

    Returns
    -------
    num_str : str
        Reduced gene or allele name with leading zeros and excess
        characters removed.
    """
    gene_name = gene_name.partition('*')[0].lower()
    gene_type = gene_type.lower()
    pre_hyphen, hyphen, post_hyphen = gene_name.partition(gene_type)[-1].partition('-')
    suffix = (pre_hyphen.lstrip('0')
              + hyphen + post_hyphen.lstrip('0')
             ).replace('/', '').replace('-1', '')
    return gene_type + suffix

def compute_pgen_expand(x):
    # compute pgen conditioned on gene usage
    return x[1].compute_aa_CDR3_pgen(x[0][0],x[0][1],x[0][2])

def compute_pgen_expand_novj(x):
    # compute pgen unconditioned on gene usage
    return x[1].compute_aa_CDR3_pgen(x[0][0])

def generate_sequence(seqgen_model: Union[sequence_generation.SequenceGenerationVJ,
                                          sequence_generation.SequenceGenerationVDJ],
                      genomic_data: Union[olga_load_model.GenomicDataVJ,
                                          olga_load_model.GenomicDataVDJ],
                      seed: Optional[np.uint64] = None,
                      add_error: bool = False,
                      error_rate: Optional[np.float64] = None
                     ) -> Tuple[str]:
    """
    Generate a sequence using OLGA.

    Parameters
    ----------
    seqgen_model : olga.sequence_generation.SequenceGenerationVJ or olga.sequence_generation.SequenceGenerationVDJ
        The OLGA sequence generation class for generating sequences.
    genomic_data : olga.load_model.GenomicDataVJ or olga.load_model.GenomicDataVDJ
        The OLGA genomic data class used for parsing gene choices.
    seed : np.uint64, optional
        The seed used for random generation.
    add_error : bool, default False
        Whether error should be added to the CDR3 sequence.
    error_rate : np.float64, optional
        The error rate.

    Returns
    -------
    tuple of str
        The generated CDR3 amino acid sequence, V gene, J gene, and CDR3 nucleotide sequence.
    """
    np.random.seed(seed)
    seq = seqgen_model.gen_rnd_prod_CDR3(conserved_J_residues='ABCEDFGHIJKLMNOPQRSTUVWXYZ')
    if add_error:
        err_seq = add_random_error(seq[0], error_rate)
        seq = [err_seq, nt2aa(err_seq), seq[2], seq[3]]

    return (seq[1], genomic_data.genV[seq[2]][0].partition('*')[0],
            genomic_data.genJ[seq[3]][0].partition('*')[0],seq[0])

def generate_paired_sequence(seqgen_model_light: sequence_generation.SequenceGenerationVJ,
                             seqgen_model_heavy: sequence_generation.SequenceGenerationVDJ,
                             genomic_data_light: olga_load_model.GenomicDataVJ,
                             genomic_data_heavy: olga_load_model.GenomicDataVDJ,
                             seed: Optional[np.uint64] = None,
                             add_error: bool = False,
                             error_rate_light: Optional[np.float64] = None,
                             error_rate_heavy: Optional[np.float64] = None
                            ) -> Tuple[str]:
    """
    Generate a paired sequence using OLGA.

    Parameters
    ----------
    seqgen_model_light : olga.sequence_generation.SequenceGenerationVJ
        The OLGA sequence generation class for generating light chain sequences.
    seqgen_model_heavy : olga.sequence_generation.SequenceGenerationVDJ
        The OLGA sequence generation class for generating heavy chain sequences.
    genomic_data_light : olga.load_model.GenomicDataVJ
        The OLGA genomic data class used for parsing light chain gene choices.
    genomic_data_heavy : olga.load_model.GenomicDataVJ or olga.load_model.GenomicDataVDJ
        The OLGA genomic data class used for parsing heavy chain gene choices.
    seed : np.uint64, optional
        The seed used for random generation.
    add_error : bool, default False
        Whether error should be added to the CDR3 sequence.
    error_rate_light : np.float64, optional
        The error rate for the light chain.
    error_rate_heavy : np.float64, optional
        The error rate for the heavy chain.

    Returns
    -------
    tuple of str
        The generated heavy CDR3 amino acid sequence, heavy V gene, heavy J gene,
        light CDR3 amino acid sequence, light V gene, light J gene, heavy CDR3
        nucleotide sequence, and light CDR3 nucleotide sequence.
    """
    np.random.seed(seed)
    seq_light = seqgen_model_light.gen_rnd_prod_CDR3(conserved_J_residues='ABCEDFGHIJKLMNOPQRSTUVWXYZ')
    seq_heavy = seqgen_model_heavy.gen_rnd_prod_CDR3(conserved_J_residues='ABCEDFGHIJKLMNOPQRSTUVWXYZ')

    if add_error:
        err_seq = add_random_error(seq_light[0], error_rate_light)
        seq_light = [err_seq, nt2aa(err_seq), seq_light[2], seq_light[3]]
        err_seq = add_random_error(seq_heavy[0], error_rate_heavy)
        seq_heavy = [err_seq, nt2aa(err_seq), seq_heavy[2], seq_heavy[3]]

    return (seq_heavy[1],
            genomic_data_heavy.genV[seq_heavy[2]][0].split('*')[0],
            genomic_data_heavy.genJ[seq_heavy[3]][0].split('*')[0],
            seq_light[1],
            genomic_data_light.genV[seq_light[2]][0].split('*')[0],
            genomic_data_light.genJ[seq_light[3]][0].split('*')[0],
            seq_heavy[0], seq_light[0])

def partial_joint_marginals(args):
    # compute joint marginals on subset of seqs.
    features = args[0]
    Qs = args[1]
    marginals = args[2]
    Z = 0

    for seq_features, Q in zip(features, Qs):
        pairs = np.stack(np.meshgrid(seq_features, seq_features), -1).reshape(-1, 2)
        marginals[pairs[:, 0], pairs[:, 1]] += Q
        Z += Q
        # NOTE No performance difference using bool instead of np.tril and np.fill_diagonal.
        #tril_bool = pairs[:, 0] > pairs[:, 1]
        #marginals[pairs[:, 0][tril_bool], pairs[:, 1][tril_bool]] += Q

    # Consistent with previous versions, return only the lower triangle with
    # diagonal containing 0.
    marginals = np.tril(marginals)
    np.fill_diagonal(marginals, 0)
    return [marginals, Z]

def parallel_function(x):
    return x[0](x[1])
