import numpy as np
import os
import olga.load_model as olga_load_model
import olga.sequence_generation as seq_gen
from sonnia.sonia_paired import SoniaPaired
class SequenceGeneration(object):

    """Class used to evaluate sequences with the sonia model

    Attributes
    ----------
    sonia_model: object 
        Required. Sonia model: only object accepted.

    custom_olga_model: object 
        Optional: already loaded custom olga sequence_generation object

    custom_genomic_data: object
        Optional: already loaded custom olga genomic_data object

    Methods
    ----------

    generate_sequences_pre(num_seqs = 1)
        Generate sequences using olga
    
    generate_sequences_post(num_seqs,upper_bound=10)
        Generate sequences using olga and perform rejection selection.
    
    rejection_sampling(upper_bound=10,energies=None)
        Returns acceptance from rejection sampling of a list of energies.
        By default uses the generated sequences within the sonia model.

    """

    def __init__(self,sonia_model=None, custom_olga_model_heavy=None,custom_olga_model_light=None,chain_type_heavy='human_T_beta',chain_type_light='human_T_alpha'):

        if type(sonia_model)==str:
            print('ERROR: you need to pass a sonia object')
        elif sonia_model is None: 
            print('Initialise default sonia model')
            self.sonia_model=SoniaPaired(custom_olga_model_heavy=None,custom_olga_model_light=None,chain_type_heavy='human_T_beta',chain_type_light='human_T_alpha')
        else: self.sonia_model=sonia_model # sonia model passed as an argument
        
        # some shortcuts
        self.genomic_data_light=self.sonia_model.genomic_data_light
        self.seq_gen_model_light=self.sonia_model.seq_gen_model_light
        self.genomic_data_heavy=self.sonia_model.genomic_data_heavy
        self.seq_gen_model_heavy=self.sonia_model.seq_gen_model_heavy

            
    def generate_sequences_pre(self, num_seqs = 1):
        """Generates MonteCarlo sequences for gen_seqs using OLGA.
        Only generates seqs from a V(D)J model. Requires the OLGA package
        (pip install olga).
        Parameters
        ----------
        num_seqs : int or float
            Number of MonteCarlo sequences to generate and add to the specified
            sequence pool.
        custom_model_folder : str
            Path to a folder specifying a custom IGoR formatted model to be
            used as a generative model. Folder must contain 'model_params.txt'
            and 'model_marginals.txt'
        Returns
        --------------
        seqs : list
            MonteCarlo sequences drawn from a VDJ recomb model
        """
        #Generate sequences
        #seqs_generated=generate_all_seqs(int(num_seqs),sg_model) # parallel version
        seqs_light = [[seq[1], self.genomic_data_light.genV[seq[2]][0].split('*')[0], self.genomic_data_light.genJ[seq[3]][0].split('*')[0]] for seq in [self.seq_gen_model_light.gen_rnd_prod_CDR3(conserved_J_residues='ABCEDFGHIJKLMNOPQRSTUVWXYZ') for _ in range(int(num_seqs))]]
        seqs_heavy = [[seq[1], self.genomic_data_heavy.genV[seq[2]][0].split('*')[0], self.genomic_data_heavy.genJ[seq[3]][0].split('*')[0]] for seq in [self.seq_gen_model_heavy.gen_rnd_prod_CDR3(conserved_J_residues='ABCEDFGHIJKLMNOPQRSTUVWXYZ') for _ in range(int(num_seqs))]]
        return [a+b for a,b in zip(seqs_heavy,seqs_light)]
    
    def generate_sequences_post(self,num_seqs,upper_bound=10):
        """Generates MonteCarlo sequences from Sonia through rejection sampling.

        Parameters
        ----------
        num_seqs : int or float
            Number of MonteCarlo sequences to generate and add to the specified
            sequence pool.
        upper_bound: int
            accept all above the threshold. Relates to the percentage of 
            sequences that pass selection.

        Returns
        --------------
        seqs : list
            MonteCarlo sequences drawn from a VDJ recomb model that pass selection.

        """
        seqs_post=[['a','b','c','d','e','f']] # initialize

        while len(seqs_post)<num_seqs:

            # generate sequences from pre
            seqs=self.generate_sequences_pre(num_seqs = int(1.1*upper_bound*num_seqs))

            # compute features and energies 
            seq_features = [self.sonia_model.find_seq_features(seq) for seq in seqs]
            energies = self.sonia_model.compute_energy(seq_features)

            #do rejection
            rejection_selection=self.rejection_sampling(upper_bound=upper_bound,energies=energies)
            print('acceptance frequency: ',np.sum(rejection_selection)/float(len(rejection_selection)))
            seqs_post=np.concatenate([seqs_post,np.array(seqs)[rejection_selection]])

        return seqs_post[1:num_seqs+1]

    def rejection_sampling(self,upper_bound=10,energies=None):

        ''' Returns acceptance from rejection sampling of a list of seqs.
        By default uses the generated sequences within the sonia model.
        
        Parameters
        ----------
        upper_bound : int or float
            accept all above the threshold. Relates to the percentage of 
            sequences that pass selection

        Returns
        -------
        rejection selection: array of bool
            acceptance of each sequence.
        
        '''

        if energies is None:  
            try:
                energies=self.energies_gen
            except:
                energies=self.sonia_model.compute_energy(self.sonia_model.gen_seq_features)
        random_samples=np.random.uniform(size=len(energies)) # sample from uniform distribution
        return random_samples < np.exp(-energies)/self.sonia_model.Z/float(upper_bound)