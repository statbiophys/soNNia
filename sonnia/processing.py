import os
from typing import Optional, Set

import numpy as np
import pandas as pd

from sonnia.utils import define_pgen_model, gene_to_num_str

class Processing(object):
    def __init__(self,
                 pgen_model: str,
                 max_length: int = 30,
                 read_thresh: Optional[int] = None,
                 verbose: bool = True
                ) -> None:
        self.read_thresh = read_thresh
        self.pgen_model = pgen_model
        self.max_length = max_length
        self.verbose = verbose
        self.load_pgen_model()

    def filter_dataframe(self,
                         dataframe: pd.DataFrame,
                         seq_col: str = 'amino_acid',
                         v_col: str = 'v_gene',
                         j_col: str = 'j_gene',
                         read_col: str = 'read_count',
                         nt_seq_col: str = 'nucleotide',
                         apply_selection: bool = True,
                         recreate_full: bool = False,
                         conserved_J_residues: str = 'ABCEDFGHIJKLMNOPQRSTUVWXYZ'
                        ) -> pd.DataFrame:
        '''
        Implement filtering pipeline
        '''
        # initialize
        self.df = dataframe.copy()
        initial_columns = list(self.df)

        self.df['selection'] = True
        self.seq_col = seq_col
        self.v_col = v_col
        self.j_col = j_col
        self.read_col = read_col
        self.nt_seq_col = nt_seq_col

        #filter
        self.select_good_genes()
        self.select_productive()
        self.select_bounding_cdr3(conserved_J_residues=conserved_J_residues)
        self.select_cdr3_length()
        if self.read_thresh is not None: self.select_read_count()

        #finalize
        if apply_selection:
            self.df = self.df.loc[self.df.selection].reset_index(drop=True)[initial_columns]
        if recreate_full:
            self.df['full_sequence'] = self.df.apply(
                lambda x: self.recreate_full_sequence(x[self.nt_seq_col], x[self.v_col],
                                                      x[self.j_col]), axis=1)
        return self.df

    def select_read_count(self
                         ) -> None:
        '''
        Select read count above threshold
        '''
        self.df['selection_read'] = self.df[self.read_col] > self.read_thresh
        if self.verbose:
            num_bad = np.count_nonzero(np.logical_not(self.df['selection_read'].values[self.df['selection'].values]))
            print(f'low reads: {num_bad}')
        self.df['selection'] = np.logical_and(self.df['selection'].values,
                                              self.df['selection_read'])

    def select_good_genes(self
                         ) -> None:
        '''
        Drops unrecongised genes and pseudogenes.
        '''
        
        v_sel=self.df[self.v_col].apply(lambda x: gene_to_num_str(x,'V') in self.good_vs)
        j_sel=self.df[self.j_col].apply(lambda x: gene_to_num_str(x,'J') in self.good_js)
        self.df['selection_genes']=np.logical_and(v_sel,j_sel)

        '''
        v_genes = set(self.df[self.v_col])
        j_genes = set(self.df[self.j_col])

        bad_v = list(v_genes - self.good_vs)
        bad_j = list(j_genes - self.good_js)

        self.df['selection_genes'] = True
        self.df = self.df.set_index(self.v_col)
        self.df.loc[bad_v, 'selection_genes'] = False
        self.df = self.df.reset_index().set_index(self.j_col)
        self.df.loc[bad_j, 'selection_genes'] = False
        self.df = self.df.reset_index()
        '''

        if self.verbose:
            num_bad = np.count_nonzero(np.logical_not(self.df['selection_genes'].values[self.df['selection'].values]))
            print(f'bad genes: {num_bad}')

        self.df['selection'] = np.logical_and(self.df['selection'].values,
                                              self.df['selection_genes'].values)

    def select_productive(self
                         ) -> None:
        '''
        looks for *,_,nan,~ in the cdr3.
        '''
        bad_cdr3 = self.df[self.seq_col].str.contains('\*|_|~',na=True,regex=True)
        self.df['selection_productive'] = np.logical_not(bad_cdr3)
        if self.verbose:
            num_bad = np.count_nonzero(np.logical_not(self.df['selection_productive'].values[self.df['selection'].values]))
            print(f'unproductive: {num_bad}')
        self.df['selection'] = np.logical_and(self.df['selection'].values,
                                              self.df['selection_productive'].values)

    def select_bounding_cdr3(self,
                             conserved_J_residues: str = 'ABCEDFGHIJKLMNOPQRSTUVWXYZ'
                            ) -> None:
        '''
        Correct cdr3 bound: C.F or C.YV for everyone expect IGH (C.W)
        '''
        string = '^C.*'+'$|^C.*'.join(list(conserved_J_residues)) + '$'
        self.df['select_bound'] = self.df[self.seq_col].str.contains(string, na=False)
        #self.df['select_bound']=self.df['amino_acid'].str.contains('^C.*F$|^C.*YV$|^C.*W$',na=False)
        if self.verbose:
            num_bad = np.count_nonzero(np.logical_not(self.df['select_bound'].values[self.df['selection'].values]))
            print(f'wrong bounds: {num_bad}')
        self.df['selection'] = np.logical_and(self.df['selection'].values,
                                              self.df['select_bound'].values)

    def select_cdr3_length(self
                          ) -> None:
        '''
        select cdr3 length smaller than max_length
        '''
        self.df['selection_length'] = self.df.amino_acid.fillna('nan').apply(len) < self.max_length
        if self.verbose:
            num_bad = np.count_nonzero(np.logical_not(self.df['selection_length'].values[self.df['selection'].values]))
            print(f'long cdr3s: {num_bad}')
        self.df['selection'] = np.logical_and(self.df['selection'].values,
                                              self.df['selection_length'].values)

    def load_pgen_model(self
                        ) -> None:
        '''
        load olga models.
        '''
        (self.genomic_data, _, self.pgen_model, _,
         _, pgen_dir) = define_pgen_model(self.pgen_model, compute_norm=False,return_pgen_dir=True)

        def get_functional_genes(fin: str
                                ) -> Set[str]:
            df = pd.read_csv(fin)
            df = df.loc[df['function'] == 'F']
            return df['gene'].apply(lambda x: gene_to_num_str(x,fin.split('/')[-1][0]))

        self.good_vs = set(get_functional_genes(os.path.join(pgen_dir, 'V_gene_CDR3_anchors.csv')))
        self.good_js = set(get_functional_genes(os.path.join(pgen_dir, 'J_gene_CDR3_anchors.csv')))

    def recreate_full_sequence(self,
                               ntcdr3: str,
                               V: str,
                               J: str
                              ) -> str:
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
        chain = self.genomic_data.genV[0][0][:3].lower()
        try:
            V = self.pgen_model.V_mask_mapping[gene_to_num_str(V,'V')][0]
            J = self.pgen_model.J_mask_mapping[gene_to_num_str(J,'J')][0]
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
