#!/usr/bin/env python
import itertools

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.cluster.hierarchy as hc
import scipy.spatial as sp
import seaborn as sns
from sklearn.metrics import auc, roc_curve

from sonnia.plotting import Plotter
from sonnia.processing import Processing
from sonnia.sonia import Sonia


class Compare:
    def __init__(
        self,
        data=[],
        chain_type="human_T_beta",
        pgen_model=None,
        gen_seqs=None,
        vj=False,
    ):
        self.chain_type = chain_type
        self.custom_pgen_model = pgen_model
        if pgen_model is None:
            self.pgen_model = chain_type
        else:
            self.pgen_model = pgen_model
        if gen_seqs is not None:
            gen = gen_seqs
        else:
            qm = SoNNia(pgen_model=self.pgen_model)
            gen_seqs = qm.generate_sequences_pre(int(1e6))
        self.processor = Processing(pgen_model=self.pgen_model)

        self.data = data
        self.datasets = []
        for d in data:
            dataset = pd.read_csv(d).sample(frac=1)[["amino_acid", "v_gene", "j_gene"]]
            self.datasets.append(list(self.processor.filter_dataframe(dataset).values))
        self.data = [r"$P_{gen}$"] + self.data
        self.labels = [r"$P_{gen}$"] + [
            " ".join(d.split("/")[-1].split(".")[0].split("_")) for d in data
        ]
        self.datasets = [gen] + self.datasets
        self.pairs = itertools.combinations(np.arange(len(self.datasets)), 2)

    def infer_models(self):
        qm = Sonia(pgen_model=self.pgen_model)
        len_feats = len(qm.features)
        qm.model.set_weights([np.zeros((len_feats, 1))])

        self.selection_models = [qm]

        for d in self.datasets[1:]:
            qm = Sonia(
                data_seqs=d, gen_seqs=self.datasets[0], pgen_model=self.pgen_model
            )
            qm.infer_selection(epochs=50, batch_size=int(1e4))
            self.selection_models.append(qm)

    def JSD(self, i, j):
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

    def evaluate(self, max_n=int(2e4), upper_limit=10):
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

    def likelihood(self, p, Q4, Q8):
        return np.mean(np.log(p * Q4 + (1.0 - p) * Q8))

    def histogram(
        self, value, binning, density=True, c="k", linewidth=1, label=None, alpha=1
    ):
        counts, bins = np.histogram(value, binning, density=True)
        plt.plot(bins[:-1], counts, alpha=alpha, label=label, linewidth=linewidth, c=c)

    def compute_roc(self, k, ax):
        ratio_i = np.log10(self.differential_qs[k][0] + 1e-30) - np.log10(
            self.differential_qs[k][2] + 1e-30
        )
        ratio_j = np.log10(self.differential_qs[k][1] + 1e-30) - np.log10(
            self.differential_qs[k][3] + 1e-30
        )
        data = np.concatenate([ratio_i.astype(np.float), ratio_j.astype(np.float)])
        true = np.zeros(len(ratio_i.astype(np.float)) + len(ratio_j.astype(np.float)))
        true[: len(ratio_i.astype(np.float))] = 1
        fpr, tpr, _ = roc_curve(true, data)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, color="k", lw=2, alpha=1, label="AUC = %0.2f)" % roc_auc)
        ax.annotate("AUC = %0.2f" % roc_auc, xy=(0.25, 0.1), fontsize=20)
        return roc_auc

    def plot_dist_matrix(self, vmax=None):
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
            cbar_kws={"label": "bits"},
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

    def plot_roc_curves(self, upper=False, save_fig=None):
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
