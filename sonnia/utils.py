from sonia.utils import gene_to_num_str
import olga.generation_probability as generation_probability
import olga.sequence_generation as sequence_generation
import olga.load_model as load_model
import olga.generation_probability as pgen
import olga.sequence_generation as seq_gen
import os
import itertools
import subprocess
import numpy as np
import sonia.sonia_leftpos_rightpos
def run_terminal(string):
    return [i.decode("utf-8").split('\n') for i in subprocess.Popen(string, shell=True, stdout=subprocess.PIPE,stderr = subprocess.PIPE).communicate()]

def sample_olga(num_gen_seqs=1,custom_model_folder=None,vj=False,chain_type='human_T_beta'):
    
    #Load generative model
    if custom_model_folder is None:
        main_folder = os.path.join(os.path.dirname(sonia.sonia_leftpos_rightpos.__file__), 'default_models', chain_type)
    else:
        main_folder = custom_model_folder

    params_file_name = os.path.join(main_folder,'model_params.txt')
    marginals_file_name = os.path.join(main_folder,'model_marginals.txt')
    V_anchor_pos_file = os.path.join(main_folder,'V_gene_CDR3_anchors.csv')
    J_anchor_pos_file = os.path.join(main_folder,'J_gene_CDR3_anchors.csv')

    if not os.path.isfile(params_file_name) or not os.path.isfile(marginals_file_name):
        print('Cannot find specified custom generative model files: ' + '\n' + params_file_name + '\n' + marginals_file_name)
        print('Exiting sequence generation...')
        return None
    if not os.path.isfile(V_anchor_pos_file):
        V_anchor_pos_file = os.path.join(os.path.dirname(olga_load_model.__file__), 'default_models', self.chain_type, 'V_gene_CDR3_anchors.csv')
    if not os.path.isfile(J_anchor_pos_file):
        J_anchor_pos_file = os.path.join(os.path.dirname(olga_load_model.__file__), 'default_models', self.chain_type, 'J_gene_CDR3_anchors.csv')

    if VJ:
        genomic_data = load_model.GenomicDataVJ()
        genomic_data.load_igor_genomic_data(params_file_name, V_anchor_pos_file, J_anchor_pos_file)
        generative_model = load_model.GenerativeModelVJ()
        generative_model.load_and_process_igor_model(marginals_file_name)
        sg_model = seq_gen.SequenceGenerationVJ(generative_model, genomic_data)
    else:
        genomic_data = load_model.GenomicDataVDJ()
        genomic_data.load_igor_genomic_data(params_file_name, V_anchor_pos_file, J_anchor_pos_file)
        generative_model = load_model.GenerativeModelVDJ()
        generative_model.load_and_process_igor_model(marginals_file_name)
        sg_model = seq_gen.SequenceGenerationVDJ(generative_model, genomic_data)

    #Generate sequences
    return [[seq[1], genomic_data.genV[seq[2]][0].split('*')[0], genomic_data.genJ[seq[3]][0].split('*')[0]] for seq in [sg_model.gen_rnd_prod_CDR3(conserved_J_residues='ABCEDFGHIJKLMNOPQRSTUVWXYZ') for _ in range(int(num_gen_seqs))]]

def correct_olga_heavy(qm):

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
    VJ_pair= [gene_to_num_str(v,'V')+','+gene_to_num_str(j,'J') for v,j in np.array(qm.data_seqs)[:,4:]]
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

