#!/usr/bin/env python
import itertools

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.cluster.hierarchy as hc
import scipy.spatial as sp
import seaborn as sns
from sklearn.metrics import auc, roc_curve
from scipy.stats import pearsonr
from sonnia.plotting import Plotter
from sonnia import Sonia,SoNNia
from tqdm import tqdm
from sonnia.utils import filter_seqs
class Compare:
    """
    A class to compare repertoires using various statistical and machine learning methods.
    
    Attributes
    ----------
    pgen_model : object
        Pgen model, default is the same as chain_type.
    data : list
        List of data file paths.
    datasets : list
        List of datasets loaded from the data file paths.
    labels : list
        List of labels for the datasets.
    pairs : itertools.combinations
        Combinations of dataset indices for pairwise self.
    selection_models : list
        List of selection models inferred from the datasets.
    selection : list
        List of selection factors.
    qs_data : list
        List of evaluated selection factors for data.
    qs_gen : list
        List of evaluated selection factors for generated sequences.
    dist_matrix : np.ndarray
        Distance matrix for the datasets.
    differential_qs : list
        List of differential selection factors between datasets.
    
    Methods
    -------
    infer_models():
        Infers selection models for the datasets.
    JSD(i, j):
        Computes the Jensen-Shannon Divergence between two datasets.
    evaluate(max_n=int(2e4), upper_limit=10):
        Evaluates the selection models and computes the distance matrix.
    likelihood(p, Q4, Q8):
        Computes the likelihood given parameters p, Q4, and Q8.
    histogram(value, binning, density=True, c="k", linewidth=1, label=None, alpha=1):
        Plots a histogram of the given values.
    compute_roc(k, ax):
        Computes the ROC curve for the k-th pair of datasets.
    plot_dist_matrix(vmax=None):
        Plots the distance matrix as a heatmap with hierarchical clustering.
    plot_report_inference(qm, save_fig=None):
        Plots the report of the inference process.
    plot_q_distributions(save_fig=None):
        Plots the distributions of the selection factors.
    plot_roc_curves(upper=False, save_fig=None):
        Plots the ROC curves for the pairwise selfs of datasets.
    save_dist_matrix(file_path):
        Saves the distance matrix to a CSV file.
    """

    def __init__(
        self,
        data=[],
        pgen_model=None,
        gen_seqs=None,
        linear: bool = True,
        ):
        """
        Initialize the compare_repertoires class.
        
        Parameters
        ----------
        data (list): A list of file paths to CSV files containing the data to be compared.
        pgen_model (optional): A model used for generating sequences. Default is None.
        gen_seqs (optional): Pre-generated sequences. If not provided, sequences will be
                             generated using the pgen_model.
        
        Attributes
        ----------
        pgen_model: The model used for generating sequences.
        data (list): A list containing the string "$P_{gen}$" followed by the file paths
                     provided in the data parameter.
        datasets (list): A list of datasets, where the first element is the generated
                         sequences and the rest are the filtered datasets from
                         the provided CSV files.
        labels (list): A list of labels for the datasets, where the first element is
                       "$P_{gen}$" and the rest are derived from the file names
                       of the provided CSV files.
        pairs: An iterator of combinations of dataset indices, used for comparing
               pairs of datasets.
        """
        self.pgen_model = pgen_model
        self.linear=linear
        if gen_seqs is not None:
            gen = gen_seqs
        else:
            qm = Sonia(pgen_model=self.pgen_model)
            gen_seqs = qm.generate_sequences_pre(int(1e6))

        self.data = data
        self.len_data=len(data)+1
        self.datasets = []
        for d in data:
            dataset = pd.read_csv(d).sample(frac=1)[["amino_acid", "v_gene", "j_gene"]]
            self.datasets.append(list(filter_seqs(dataset, qm).values))
        self.data = [r"$P_{gen}$"] + self.data
        self.labels = [r"$P_{gen}$"] + [
            " ".join(d.split("/")[-1].split(".")[0].split("_")) for d in data
        ]
        self.datasets = [gen] + self.datasets
        self.pairs = itertools.combinations(np.arange(len(self.datasets)), 2)

    def define_models(self, infer: bool = True) -> None:
        """
        Define and infers selection models for the datasets.

        This method initializes a Sonia model with the provided pgen_model and sets its weights to zero.
        It then iterates over the datasets, inferring selection models for each dataset
        after the first one. The inferred models are stored in the selection_models attribute.

        Attributes
        ----------
            self.selection_models (list): A list to store the inferred selection models.
        """  # noqa: D205
        if self.linear:
            model=Sonia
        else:
            model=SoNNia
        qm = model(pgen_model=self.pgen_model,data_seqs=self.datasets[0])
        len_feats = len(qm.features)
        qm.model.set_weights([np.zeros((len_feats, 1))])

        self.selection_models = [qm]

        for d in self.datasets[1:]:
            if infer:
                qm = model(data_seqs=d, gen_seqs=self.datasets[0], pgen_model=self.pgen_model)
                qm.infer_selection(epochs=50, batch_size=int(1e4))
            else:
                qm = model(data_seqs=d, pgen_model=self.pgen_model)
            self.selection_models.append(qm)

    def JSD(self, i:int, j:int) -> float:
        """
        Calculate the Jensen-Shannon Divergence (JSD).

        The JSD is a method of measuring the similarity between two probability
        distributions. It is a symmetric version of the Kullback-Leibler divergence.

        Parameters
        ----------
        i (int): Index of the first probability distribution.
        j (int): Index of the second probability distribution.

        Returns
        -------
        float: The Jensen-Shannon Divergence between the two distributions.
        """
        part1 = (
            np.mean(
                np.log2(
                    self.qs_gen[i][self.selection[i]]
                    / (
                        self.qs_gen[i][self.selection[i]]
                        + self.qs_gen[j][self.selection[i]]
                    )
                )
            )
            / 2
        )
        part2 = (
            np.mean(
                np.log2(
                    self.qs_gen[j][self.selection[j]]
                    / (
                        self.qs_gen[i][self.selection[j]]
                        + self.qs_gen[j][self.selection[j]]
                    )
                )
            )
            / 2
        )
        return 1 + part1 + part2

    def evaluate_raw_encodings(self)-> None:
        """
        Evaluate raw repertoire encodings and compute a distance matrix.

        This method processes the repertoire encodings from the selection models,
        computes the marginals using a flat distribution, and stores the encodings
        in a DataFrame. It then calculates a distance matrix based on the Pearson
        correlation coefficient between the encodings.

        Attributes
        ----------
        - self.dist_matrix: Distance matrix computed from the repertoire encodings.
        """
        self.define_models(infer=False)
        reps_encoding=[]
        for sonia_model in self.selection_models:
            encoding = sonia_model.compute_marginals(encoding=sonia_model.data_encoding, use_flat_distribution=True)
            reps_encoding.append(encoding)
        feats=['_'.join(x) for x in sonia_model.features]
        self.all_dfs=pd.DataFrame(np.array(reps_encoding),columns=feats)

        self.dist_matrix=np.zeros((self.len_data,self.len_data))
        for i in tqdm(range(self.len_data)):
            dfi=self.all_dfs.iloc[i].values
            sel=dfi>1e-6 # select non zero marginals, hacky, possibly not needed it
            for j in range(self.len_data):
                dfj=self.all_dfs.iloc[j].values
                self.dist_matrix[i,j]=np.sqrt(1-(pearsonr(dfi[sel],dfj[sel])[0])**2)
        self.dist_matrix=(self.dist_matrix+self.dist_matrix.T)/2
        self.dist_matrix=self.dist_matrix-np.diag(np.diag(self.dist_matrix))

    def evaluate(self, max_n:int = int(2e4), upper_limit: int = 10)-> None:
        """
        Evaluate selection models and compute distance matrix and differential selection factors.

        Parameters
        ----------
        max_n (int): Maximum number of samples to consider from each dataset.
                     Default is 20000.
        upper_limit (int): Multiplier for the upper limit of samples
                           from the first dataset. Default is 10.

        Attributes
        ----------
        selection (list): List of boolean arrays indicating selected samples.
        qs_data (list): List of selection factors for the data.
        qs_gen (list): List of selection factors for the generated data.
        dist_matrix (ndarray): Matrix of Jensen-Shannon Divergence values between
                               selection models.
        differential_qs (list): List of differential selection factors between pairs
                                of datasets.
        pairs (iterator): Iterator over combinations of dataset indices.
        """
        self.define_models(infer=True)
        self.selection = []
        self.qs_data = []
        self.qs_gen = []
        for i in range(len(self.selection_models)):
            q_data = self.selection_models[i].evaluate_selection_factors(
                self.datasets[i][:max_n]
            )
            q_gen = self.selection_models[i].evaluate_selection_factors(
                self.datasets[0][: max_n * upper_limit]
            )
            self.selection.append(np.random.uniform(size=len(q_gen)) < q_gen / 10.0)
            self.qs_data.append(q_data)
            self.qs_gen.append(q_gen)

        self.dist_matrix = np.zeros(
            (len(self.selection_models), len(self.selection_models))
        )
        self.differential_qs = []
        self.pairs = itertools.combinations(np.arange(len(self.datasets)), 2)

        for i, j in self.pairs:
            JS = self.JSD(i, j)
            self.dist_matrix[i, j] = JS
            self.dist_matrix[j, i] = JS
            q_ij = self.selection_models[i].evaluate_selection_factors(
                self.datasets[j][:max_n]
            )
            q_ji = self.selection_models[j].evaluate_selection_factors(
                self.datasets[i][:max_n]
            )
            self.differential_qs.append([self.qs_data[i], q_ij, q_ji, self.qs_data[j]])

    def likelihood(self, p:float, Q4:np.array, Q8:np.array) -> float:
        """
        Calculate the likelihood of a mixture model.

        Parameters
        ----------
        p (float): The mixing proportion between Q4 and Q8.
        Q4 (numpy.ndarray): The first component of the mixture model.
        Q8 (numpy.ndarray): The second component of the mixture model.

        Returns
        -------
        float: The mean log-likelihood of the mixture model.
        """
        return np.mean(np.log(p * Q4 + (1.0 - p) * Q8))

    def histogram(
        self, value, binning, density=True, c="k", linewidth=1, label=None, alpha=1
    ):
        """
        Plots a histogram of the given values.

        Parameters
        ----------
        value (array-like): The data to be histogrammed.
        binning (int or sequence): If an integer is given, it defines the number of equal-width bins in the given range. 
                                   If a sequence is given, it defines the bin edges, including the rightmost edge.
        density (bool, optional): If True, the first element of the return tuple will be the counts normalized to form a probability density, i.e., the area (or integral) under the histogram will sum to 1. Default is True.
        c (str, optional): The color of the histogram plot. Default is 'k' (black).
        linewidth (float, optional): The width of the lines of the histogram plot. Default is 1.
        label (str, optional): The label for the histogram plot. Default is None.
        alpha (float, optional): The alpha blending value, between 0 (transparent) and 1 (opaque). Default is 1.

        """
        counts, bins = np.histogram(value, binning, density=True)
        plt.plot(bins[:-1], counts, alpha=alpha, label=label, linewidth=linewidth, c=c)

    def compute_roc(self, k, ax):
        """
        Compute the Receiver Operating Characteristic (ROC) curve and Area Under the Curve (AUC) for the given data.

        Parameters
        ----------
        k (int): Index to access specific elements in the differential_qs attribute.
        ax (matplotlib.axes.Axes): The axes on which to plot the ROC curve.

        Returns
        -------
        float: The computed AUC value.
        """
        ratio_i = np.log10(self.differential_qs[k][0] + 1e-30) - np.log10(
            self.differential_qs[k][2] + 1e-30
        )
        ratio_j = np.log10(self.differential_qs[k][1] + 1e-30) - np.log10(
            self.differential_qs[k][3] + 1e-30
        )
        data = np.concatenate([ratio_i.astype(float), ratio_j.astype(float)])
        true = np.zeros(len(ratio_i.astype(float)) + len(ratio_j.astype(float)))
        true[: len(ratio_i.astype(float))] = 1
        fpr, tpr, _ = roc_curve(true, data)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, color="k", lw=2, alpha=1, label="AUC = %0.2f)" % roc_auc)
        ax.annotate("AUC = %0.2f" % roc_auc, xy=(0.25, 0.1), fontsize=20)
        return roc_auc

    def plot_dist_matrix(self, vmax=None)-> None:
        """
        Plots a distance matrix as a clustered heatmap with dendrograms.

        Parameters
        ----------
        vmax (float, optional): The maximum value for the colormap. If None, the maximum
                                value in the distance matrix is used.
        """
        if vmax is None:
            vmax = np.max(self.dist_matrix)
        linkage = hc.linkage(
            sp.distance.squareform(self.dist_matrix),
            method="average",
            optimal_ordering=True,
        )
        my_df = pd.DataFrame(self.dist_matrix, columns=self.labels)
        my_df.index = self.labels
        g = sns.clustermap(
            my_df,
            cbar_kws={"label": "distance"},
            figsize=(8, 8),
            row_linkage=linkage,
            col_linkage=linkage,
            vmax=vmax,
        )
        plt.setp(g.ax_heatmap.yaxis.get_majorticklabels(), rotation=0)
        plt.setp(g.ax_heatmap.xaxis.get_majorticklabels(), rotation=90)
        g.cax.figure.axes[-1].yaxis.label.set_size(30)
        cax = plt.gcf().axes[-1]
        cax.tick_params(labelsize=30)
        for a in g.ax_row_dendrogram.collections:
            a.set_linewidth(3)
        for a in g.ax_col_dendrogram.collections:
            a.set_linewidth(3)
        g.ax_row_dendrogram.set_visible(False)
        g.ax_heatmap.set_xticklabels(g.ax_heatmap.get_xmajorticklabels(), fontsize=30)
        g.ax_heatmap.set_yticklabels(g.ax_heatmap.get_ymajorticklabels(), fontsize=30)
        g.cax.yaxis.set_ticks_position("left")
        g.cax.yaxis.set_label_position("left")
        dendro_box = g.ax_row_dendrogram.get_position()
        dendro_box.x0 = (dendro_box.x0 + 2 * dendro_box.x1) / 3 - 0.01
        dendro_box.x1 = dendro_box.x1 - 0.01
        g.cax.set_position(dendro_box)
        plt.tight_layout()

    def plot_report_inference(self, qm, save_fig=None):
        """
        Generates and saves plots for model learning, VJL, and logQ.

        Parameters
        ----------
        qm : object
            The model or data to be plotted.
        save_fig : str, optional
            The file path to save the figures. If None, the figures will not be saved.
        """
        pl = Plotter(qm)
        pl.plot_model_learning(save_name=save_fig)
        pl.plot_vjl(save_name=save_fig)
        pl.plot_logQ(save_name=save_fig)

    def plot_q_distributions(self, save_fig=None):
        n = len(self.selection_models) - 1
        fig = plt.figure(figsize=(4 * n, 4), dpi=200)
        binning = np.linspace(-8, 5, 50)
        for i in range(n):
            plt.subplot(1, n, 1 + i)
            plt.title(self.labels[i + 1], fontsize=20)
            self.histogram(np.log(self.qs_data[i + 1]), binning=binning, c="r")
            self.histogram(np.log(self.qs_gen[i + 1]), binning=binning, c="k")
            plt.ylim([0, 0.6])
            if i > 5:
                plt.xlabel(r"$\log Q $", fontsize=30)
            if i == 0:
                plt.ylabel(" density ", fontsize=30)
        plt.tight_layout()
        if save_fig is None:
            plt.show()
        else:
            fig.savefig(save_fig)

    def plot_roc_curves(self, upper=False, save_fig=None) -> None:
        """
        Plots ROC curves for the models in the selection.

        Parameters
        ----------
        upper (bool): If True, plots the ROC curves in the upper triangle of the subplot grid.
                      If False, plots the ROC curves in the lower triangle of the subplot grid.
                      Default is False.
        save_fig (str or None): If a string is provided, saves the figure to the specified file path.
                                If None, displays the figure. Default is None.
        """
        n = len(self.selection_models)
        self.pairs = itertools.combinations(np.arange(len(self.datasets)), 2)

        fig = plt.figure(figsize=(32, 32))
        for k, (i, j) in enumerate(self.pairs):
            if upper:
                ax = fig.add_subplot(n, n, 1 + i * n + j)
            else:
                ax = fig.add_subplot(n, n, 1 + j * n + i)

            auc_ = self.compute_roc(k, ax)
            plt.plot([0, 1], [0, 1])
            plt.xlim([0, 1])
            plt.ylim([0, 1])
            title = self.labels[j]
            label = self.labels[i]
            if upper:
                if i == 0:
                    plt.title(title, fontsize=25)
                if j - i == 1:
                    plt.ylabel(label, fontsize=25)
            else:
                if i == 0:
                    plt.ylabel(title, fontsize=25)
                if j - i == 1:
                    plt.title(label, fontsize=25)
            plt.xticks([], [])
            plt.yticks([], [])
        if save_fig is None:
            plt.show()
        else:
            fig.savefig(save_fig)

    def save_dist_matrix(self, file_path):
        """
        Saves the distance matrix to a CSV file.

        Parameters
        ----------
        file_path : str
            The path to the file where the distance matrix will be saved.
        """
        df = pd.DataFrame(self.dist_matrix, index=self.labels, columns=self.labels)
        df.to_csv(file_path)
