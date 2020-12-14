import numpy as np
import pandas as pd
import random
import subprocess
import olga.load_model as load_model
import olga.generation_probability as pgen
import olga.sequence_generation as seq_gen
import os
import sonia.sonia_leftpos_rightpos
from sonia.utils import gene_to_num_str

translate_header = {
    'sequenceStatus': 'frame_type',
    'aminoAcid': 'amino_acid',
    'vGeneName': 'v_gene',
    'jGeneName': 'j_gene',
    'rearrangement':'rearrangement',
    'aminoAcid': 'amino_acid',
    'junction_aa': 'amino_acid',
    'CDR3.amino.acid.sequence':'amino_acid',
    'CDR3.nucleotide.sequence':'nucleotide',
    'bestVGene':'v_gene',
    'bestJGene':'j_gene',
    'Read.count':'read_count',
    'reads':'read_count'}

default_chain_types = { 'humanTRA': 'human_T_alpha', 'human_T_alpha': 'human_T_alpha', 
                        'humanTRB': 'human_T_beta', 'human_T_beta': 'human_T_beta', 
                        'humanIGH': 'human_B_heavy', 'human_B_heavy': 'human_B_heavy', 
                        'humanIGK': 'human_B_kappa', 'human_B_kappa': 'human_B_kappa', 
                        'humanIGL': 'human_B_lambda', 'human_B_lambda': 'human_B_lambda', 
                        'mouseTRB': 'mouse_T_beta', 'mouse_T_beta': 'mouse_T_beta'}

class Processing(object):
    
    def __init__(self,read_thresh=None,custom_model_folder=None,chain_type='human_T_beta',drop_unproductive=True,max_length=30,vj=False):
        
        self.read_thresh=read_thresh
        self.custom_model_folder=custom_model_folder
        self.max_length=max_length
        self.chain_type=default_chain_types[chain_type]
        if chain_type not in default_chain_types.keys():
            print('Unrecognized chain_type (not a default OLGA model).'
                  ' Please specify one of the following options: humanTRA, humanTRB, humanIGH, humanIGK, humanIGL or mouseTRB.')
            return None
        self.read_thresh=read_thresh
        self.chain_type = default_chain_types[chain_type]
        self.vj=vj
        if self.chain_type in ['human_T_alpha','human_B_kappa','human_B_lambda']: self.vj=True    
        self.define_models()
    
    def filter_dataframe(self, dataframe=[],apply_selection=True,recreate_full=False,conserved_J_residues='ABCEDFGHIJKLMNOPQRSTUVWXYZ'):
        '''
        Implement filtering pipeline
        '''
        # initilize
        self.df=dataframe.copy()
        self.rename_cols()
        initial_columns=self.df.columns.values
        self.df['selection']=True
        #filter
        self.select_good_genes()
        self.select_productive()
        self.select_bounding_cdr3(conserved_J_residues=conserved_J_residues)
        self.select_cdr3_length()
        if not self.read_thresh is None: self.select_read_count()

        #finalize
        if apply_selection: 
            self.df=self.df.loc[self.df.selection].reset_index(drop=True)[initial_columns]
        if recreate_full:
            self.df['full_sequence']=self.df.apply(lambda x: self.recreate_full_sequence(x['nucleotide'],x['v_gene'],x['j_gene']), axis=1)    
            
        return self.df

    def select_read_count(self):
        '''
        Select read count above threshold
        '''
        self.df['selection_read']=self.df.read_count>self.read_thresh
        print ('low reads:',np.sum(np.logical_not(self.df['selection_read'].values[self.df['selection'].values])))
        self.df['selection']=np.logical_and(self.df['selection'].values,self.df['selection_read'])        
            
    def select_good_genes(self):   
        '''
        Translates genes to olga format. Drops unrecongised genes and pseudogenes.
        '''
        self.df['v_gene']=self.df.v_gene.apply(self.rename_V)
        self.df['j_gene']=self.df.j_gene.apply(self.rename_J)
        selv=self.df['v_gene']=='none' 
        selj=self.df['j_gene']=='none' 
        bad_genes = np.logical_or(self.df['j_gene'].apply(lambda x: x in self.bad_js),
                                          self.df['v_gene'].apply(lambda x: x in self.bad_vs))
        recognised_genes=np.logical_or(selv,selj)
        self.df['selection_genes']=np.logical_not(np.logical_or(bad_genes,recognised_genes))
        print ('bad genes:',np.sum(np.logical_not(self.df['selection_genes'].values[self.df['selection'].values])))
        self.df['selection']=np.logical_and(self.df['selection'].values,self.df['selection_genes'].values)
        
    def select_productive(self):
        '''
        looks for *,_,nan,~ in the cdr3.
        '''
        bad_cdr3=self.df.amino_acid.str.contains('\*|_|~',na=True,regex=True)
        self.df['selection_productive']=np.logical_not(bad_cdr3)
        print ('unproductive:',np.sum(np.logical_not(self.df['selection_productive'].values[self.df['selection'].values])))
        self.df['selection']=np.logical_and(self.df['selection'].values,self.df['selection_productive'].values)

        
    def select_bounding_cdr3(self,conserved_J_residues='ABCEDFGHIJKLMNOPQRSTUVWXYZ'):
        '''
        Correct cdr3 bound: C.F or C.YV for everyone expect IGH (C.W)
        '''
        
        string='^C.*'+'$|^C.*'.join(list(conserved_J_residues))+'$'
        self.df['select_bound']=self.df['amino_acid'].str.contains(string,na=False)
        #self.df['select_bound']=self.df['amino_acid'].str.contains('^C.*F$|^C.*YV$|^C.*W$',na=False)
        print ('wrong bounds:',np.sum(np.logical_not(self.df['select_bound'].values[self.df['selection'].values])))
        self.df['selection']=np.logical_and(self.df['selection'].values,self.df['select_bound'].values)
    
    def select_cdr3_length(self):
        '''
        select cdr3 length smaller than max_length
        '''
        self.df['selection_length']=self.df.amino_acid.fillna('nan').apply(len)<self.max_length
        print ('long cdr3s:',np.sum(np.logical_not(self.df['selection_length'].values[self.df['selection'].values])))
        self.df['selection']=np.logical_and(self.df['selection'].values,self.df['selection_length'].values)
        
    def define_models(self):
        '''
        load olga models.
        '''
        #Load generative model
        if self.custom_model_folder is not None:
            main_folder = self.custom_model_folder
        else:
            main_folder=os.path.join(os.path.dirname(sonia.sonia_leftpos_rightpos.__file__),'default_models',self.chain_type)

        params_file_name = os.path.join(main_folder,'model_params.txt')
        marginals_file_name = os.path.join(main_folder,'model_marginals.txt')
        V_anchor_pos_file = os.path.join(main_folder,'V_gene_CDR3_anchors.csv')
        J_anchor_pos_file = os.path.join(main_folder,'J_gene_CDR3_anchors.csv')
        
        if not self.vj:
            self.genomic_data = load_model.GenomicDataVDJ()
            self.genomic_data.load_igor_genomic_data(params_file_name, V_anchor_pos_file, J_anchor_pos_file)
            self.generative_model = load_model.GenerativeModelVDJ()
            self.generative_model.load_and_process_igor_model(marginals_file_name)
            self.pgen_model = pgen.GenerationProbabilityVDJ(self.generative_model, self.genomic_data)
            self.seq_gen_model = seq_gen.SequenceGenerationVDJ(self.generative_model, self.genomic_data)
        else:
            self.genomic_data = load_model.GenomicDataVJ()
            self.genomic_data.load_igor_genomic_data(params_file_name, V_anchor_pos_file, J_anchor_pos_file)
            self.generative_model = load_model.GenerativeModelVJ()
            self.generative_model.load_and_process_igor_model(marginals_file_name)
            self.pgen_model = pgen.GenerationProbabilityVJ(self.generative_model, self.genomic_data)
            self.seq_gen_model = seq_gen.SequenceGenerationVJ(self.generative_model, self.genomic_data)
            
        # check for pseudogenes
        file=pd.read_csv(V_anchor_pos_file)
        file=file.loc[file.function!='F']
        allele=file.gene.apply(lambda x: x.split('*')[1])
        sel=np.logical_or(allele=='01',allele=='00')                       
        file=file.loc[sel]
        self.bad_vs=file.gene.apply(lambda x: x.split('*')[0]).values
        
        file=pd.read_csv(J_anchor_pos_file)
        file=file.loc[file.function!='F']
        allele=file.gene.apply(lambda x: x.split('*')[1])
        sel=np.logical_or(allele=='01',allele=='00')      
        file=file.loc[sel]
        self.bad_js=file.gene.apply(lambda x: x.split('*')[0]).values

    def rename_cols(self):
        columns=self.df.columns.values
        new_cols=[]
        for col in columns:
            try: new_cols.append(translate_header[col])
            except: new_cols.append(col)
        self.df.columns=new_cols
        
    def rename_V(self,gene):
        try:
            gene_split=gene_to_num_str(gene,'V')
            index= self.pgen_model.V_mask_mapping[gene_split][0]
            return self.pgen_model.V_allele_names[index].split('*')[0]
        except:
            return 'none'
        
    def rename_J(self,gene):
        try:
            gene_split=gene_to_num_str(gene,'J')
            index= self.pgen_model.J_mask_mapping[gene_split][0]
            return self.pgen_model.J_allele_names[index].split('*')[0]
        except:
            return 'none'
        
    def recreate_full_sequence(self,ntcdr3, V, J):
        """ Recreate full sequences from the nucleotide CDR3 and
        the V and J. Adapted from Thomas Dupic.
        
        @Arguments:
        * ntcdr3: nucleotide CDR3 sequence (with the two extremities
        * Vname: name of the V gene (as precise as possible)
        * Jname: name of the J gene (as precise as possible)
        * genomic_data: GenomicDataV(D)J object
        
        @Return:
        full sequence in nucleotides
        """
        chain=self.genomic_data.genV[0][0][:3].lower()
        try:
            V=self.pgen_model.V_mask_mapping[gene_to_num_str(V,'V')][0]
            J=self.pgen_model.J_mask_mapping[gene_to_num_str(J,'J')][0]
        except:
            return 'fail'
        fullV_gene = self.genomic_data.genV[V][2]
        endV = -len(self.genomic_data.genV[V][1])
        begin = fullV_gene[:endV]
        fullJ_gene = self.genomic_data.genJ[J][2]
        beginJ = len(self.genomic_data.genJ[J][1])
        end = fullJ_gene[beginJ:]
        ntseq = begin + ntcdr3 + end
        return ntseq