import os
import subprocess
from typing import Optional

import numpy as np

import olga.generation_probability as generation_probability
import olga.sequence_generation as sequence_generation
import olga.load_model as olga_load_model
import olga.generation_probability as pgen
import olga.sequence_generation as seq_gen

DEFAULT_CHAIN_TYPES = {'humanTRA': 'human_T_alpha', 'human_T_alpha': 'human_T_alpha',
                       'humanTRB': 'human_T_beta', 'human_T_beta': 'human_T_beta',
                       'humanIGH': 'human_B_heavy', 'human_B_heavy': 'human_B_heavy',
                       'humanIGK': 'human_B_kappa', 'human_B_kappa': 'human_B_kappa',
                       'humanIGL': 'human_B_lambda', 'human_B_lambda': 'human_B_lambda',
                       'mouseTRB': 'mouse_T_beta', 'mouse_T_beta': 'mouse_T_beta',
                       'mouseTRA': 'mouse_T_alpha','mouse_T_alpha':'mouse_T_alpha'}
NORM_PRODUCTIVES = {'human_T_beta': 0.2442847269027897,
                    'human_T_alpha': 0.2847166577727317,
                    'human_B_heavy': 0.1499265655936305,
                    'human_B_lambda': 0.29489499727399304,
                    'human_B_kappa': 0.29247125650320943,
                    'mouse_T_beta': 0.2727148540013573,
                    'mouse_T_alpha': 0.321870924914448}

HEAVY_CHAINS = {'TRB', 'TRD', 'IGH'}
LIGHT_CHAINS = {'TRA', 'TRG', 'IGK', 'IGL', 'IGI'}


def run_terminal(string):
    return [i.decode("utf-8").split('\n')
            for i in subprocess.Popen(string, shell=True,
                                      stdout=subprocess.PIPE,
                                      stderr=subprocess.PIPE).communicate()]

def define_pgen_model(pgen_model: Optional[str] = None,
                      compute_norm: bool = True
                     ):
    if pgen_model in DEFAULT_CHAIN_TYPES:
        filedir = os.path.dirname(os.path.abspath(__file__))
        pgen_model = DEFAULT_CHAIN_TYPES[pgen_model]
        pgen_dir = os.path.join(filedir, 'default_models', pgen_model)
        norm_productive = NORM_PRODUCTIVES[pgen_model]
    elif os.path.isdir(pgen_model):
        pgen_dir = pgen_model
        norm_productive = None
    else:
        options = f'{list(DEFAULT_CHAIN_TYPES.keys())[::2]}'[1:-1]
        raise ValueError('pgen_model is neither a directory that exists '
                         f'nor one of the default options ({options}). '
                         'Try using one of the default options or an existing '
                         'directory containing the pgen model.')

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
        with open(fi, 'r') as fin:
            # Skip headers.
            next(fin)

            chain = next(fin).partition(',')[0].partition(gene)[0]
            chains.append(chain)
    if chains[0] != chains[1]:
        raise RuntimeError('Either the V and J anchor files do not correspond '
                           'to the same chain or the files are not in the expected '
                           'format (gene,anchor_index,function).')

    if chains[0] in HEAVY_CHAINS:
        model_str = 'VDJ'
    elif chains[0] in LIGHT_CHAINS:
        model_str = 'VJ'
    else:
        heavy_chain_str = f'{HEAVY_CHAINS}'[1:-1]
        light_chain_str = f'{LIGHT_CHAINS}'[1:-1]
        raise RuntimeError(f'Unrecognized chain: {chains[0]}. Recognized heavy chains: '
                           f'{heavy_chain_str} Recognized light chains: {light_chain_str}.')

    genomic_data = getattr(olga_load_model, f'GenomicData{model_str}')()
    genomic_data.load_igor_genomic_data(params_file_name,
                                        V_anchor_pos_file,
                                        J_anchor_pos_file)
    generative_model = getattr(olga_load_model, f'GenerativeModel{model_str}')()
    generative_model.load_and_process_igor_model(marginals_file_name)
    pgen_model = getattr(pgen, f'GenerationProbability{model_str}')(generative_model, genomic_data)
    seqgen_model = getattr(seq_gen, f'SequenceGeneration{model_str}')(generative_model, genomic_data)

    if compute_norm and norm_productive is None:
        norm_productive = pgen_model.compute_regex_CDR3_template_pgen('CX{0,}')

    return (genomic_data, generative_model, pgen_model, seqgen_model,
            norm_productive, pgen_dir, model_str, chains[0])

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

def gene_to_num_str(gene_name: str,
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
    return gene_type + (pre_hyphen.lstrip('0') + hyphen + post_hyphen.lstrip('0')).replace('/', '')

def compute_pgen_expand(x):
    # compute pgen conditioned on gene usage
    return x[1].compute_aa_CDR3_pgen(x[0][0],x[0][1],x[0][2])

def compute_pgen_expand_novj(x):
    # compute pgen unconditioned on gene usage
    return x[1].compute_aa_CDR3_pgen(x[0][0])

def generate_sequence(x):
    seq_gen_model=x[0]
    genomic_data=x[1]
    np.random.seed(x[2])
    seq=seq_gen_model.gen_rnd_prod_CDR3(conserved_J_residues='ABCEDFGHIJKLMNOPQRSTUVWXYZ')
    return [seq[1], genomic_data.genV[seq[2]][0].split('*')[0], genomic_data.genJ[seq[3]][0].split('*')[0],seq[0]]

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
