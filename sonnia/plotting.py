#!/usr/bin/env python
# -*- coding: utf-8 -*-
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
"""
import os
from typing import Optional, Type

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sonnia.sonia import Sonia
from sonnia.sonnia import SoNNia
from sonnia.sonia_paired import SoniaPaired
from sonnia.sonnia_paired import SoNNiaPaired

class Plotter(object):
    """Class used to plot stuff


    Attributes
    ----------
    sonia_model: object
        Sonia model. No path.

    Methods
    ----------
    plot_model_learning(save_name = None)
        Plots L1 convergence curve and marginal scatter.

    plot_pgen(pgen_data=[],pgen_gen=[],pgen_model=[],n_bins=100)
        Histogram plot of pgen. You need to evalute them first.

    plot_ppost(ppost_data=[],ppost_gen=[],pppst_model=[],n_bins=100)
        Histogram plot of ppost. You need to evalute them first.

    plot_vjl(save_name = None)
        Plots marginals of V gene, J gene and cdr3 length

    plot_logQ(save_name=None)
        Plots logQ of data and generated sequences

    plot_ratioQ(self,save_name=None)
        Plots the ratio of P(Q) in data and pre-selected pool. Useful for model validation.
    """

    def __init__(
        self,
        sonia_model: Type[Sonia]
    ):
        correct_type = False
        for sonia_type in (Sonia, SoNNia, SoniaPaired, SoNNiaPaired):
            correct_type += isinstance(sonia_model, sonia_type)
        if not correct_type:
            raise RuntimeError(
                'Plotter must be initialized with a Sonia, SoNNia, SoniaPaired '
                'or SoNNiaPaired object.'
            )

        self.sonia_model = sonia_model

    def plot_prob(self, data=[],gen=[],model=[],n_bins=30,save_name=None,bin_min=-20,bin_max=-5,ptype='P_{pre}',figsize=(6,4)):
        '''Histogram plot of Pgen

        Parameters
        ----------
        n_bins: int
            number of bins of the histogram

        '''
        fig=plt.figure(figsize=figsize)

        binning_=np.linspace(bin_min,bin_max,n_bins)
        k,l=np.histogram(np.nan_to_num(np.log10(np.array(data)+1e-300)),binning_,density=True)
        plt.plot(l[:-1],k,label='data',linewidth=2)
        k,l=np.histogram(np.nan_to_num(np.log10(np.array(gen)+1e-300)),binning_,density=True)
        plt.plot(l[:-1],k,label='pre-sel',linewidth=2)
        k,l=np.histogram(np.nan_to_num(np.log10(np.array(model)+1e-300)),binning_,density=True)
        plt.plot(l[:-1],k,label='post-sel',linewidth=2)

        plt.xlabel('$log_{10}'+ptype+'$',fontsize=20)
        plt.ylabel('density',fontsize=20)
        plt.legend()
        fig.tight_layout()

        if save_name is not None:
            fig.savefig(save_name)
        plt.show()

    def plot_model_learning(
        self,
        save_name: Optional[str] = None
    ) -> None:
        """
        Plot L1 convergence curve and marginal scatter.

        Parameters
        ----------
        save_name : str or None
            File name to save output figure. If None (default) does not save.

        Returns
        -------
        None
        """

        if len(self.sonia_model.data_seqs):
            num_data_seqs = len(self.sonia_model.data_seqs)
            min_for_plot = 10**(-np.ceil(np.log10(num_data_seqs)))
        else:
            min_nonzero_marginal = np.min(self.sonia_model.data_marginals[self.sonia_model.data_marginals != 0])
            min_for_plot = 10**(np.floor(np.log10(min_nonzero_marginal)))

        fig = plt.figure(figsize =(9, 4))

        ax_marg = fig.add_subplot(121)

        ax_marg.loglog(self.sonia_model.data_marginals, self.sonia_model.gen_marginals, 'r.', alpha=0.2, markersize=1)
        ax_marg.loglog(self.sonia_model.data_marginals, self.sonia_model.model_marginals, 'b.', alpha=0.2, markersize=1)
        ax_marg.loglog([],[], 'r.', label='Raw marginals')
        ax_marg.loglog([],[], 'b.', label='Model adjusted marginals')

        ax_marg.loglog([min_for_plot, 2], [min_for_plot, 2], 'k--', linewidth=0.5)
        ax_marg.set_xlim([min_for_plot, 1])
        ax_marg.set_ylim([min_for_plot, 1])

        ax_marg.set_xlabel('Marginals over data', fontsize=13)
        ax_marg.set_ylabel('Marginals over generated sequences', fontsize=13)
        ax_marg.legend(loc=2, fontsize=10)
        ax_marg.set_title('Marginal Scatter', fontsize=15)
        ax_marg.tick_params(labelsize=12)

        ax_llh = fig.add_subplot(122)
        ax_llh.set_title('Likelihood', fontsize=15)
        ax_llh.plot(self.sonia_model.likelihood_train, label='train', c='k')
        ax_llh.plot(self.sonia_model.likelihood_test, label='validation', c='r')
        ax_llh.legend(fontsize=10)
        ax_llh.set_xlabel('Iteration', fontsize=13)
        ax_llh.set_ylabel('Likelihood', fontsize=13)
        ax_llh.tick_params(labelsize=12)

        fig.tight_layout()

        if save_name is not None:
            fig.savefig(save_name)
        else: plt.show()

    def plot_vjl(
        self,
        save_name: Optional[str] = None
    ) -> None:
        """
        Plot marginals of V gene, J gene and cdr3 length

        Parameters
        ----------
        save_name : str, optional
            File name to save output figure. If None (default) does not save.

        Returns
        -------
        None
        """
        df = pd.DataFrame(
            {'feature': self.sonia_model.features.ravel(),
             'model_marginal': self.sonia_model.model_marginals,
             'gen_marginal': self.sonia_model.gen_marginals,
             'data_marginal': self.sonia_model.data_marginals,
            }
        ).explode(
            'feature'
        ).groupby(
            ['feature']
        )[['model_marginal', 'gen_marginal', 'data_marginal']].sum()

        labels = ('POST marginals', 'DATA marginals', 'GEN marginals')
        marg_types = ('model', 'data', 'gen')

        fig = plt.figure(figsize=(16,4))
        ax_cdr3 = fig.add_subplot(121)
        if df.index.str.contains('_h').any():
            for chain_type in ('h', 'l'):
                df_l = df[df.index.str[:3] == f'l_{chain_type}'].assign(
                    position=lambda x: x.index.str[3:].astype(int)
                ).sort_values(
                    'position'
                )
                for idx, (label, marg_type) in enumerate(zip(labels, marg_types)):
                    if chain_type == 'h':
                        ax_cdr3.plot(
                            df_l['position'].values,
                            df_l[f'{marg_type}_marginal'].values,
                            label=f'heavy {label}', alpha=0.9
                        )
                    else:
                        ax_cdr3.plot(
                            df_l['position'].values,
                            df_l[f'{marg_type}_marginal'].values,
                            label=f'light {label}', alpha=0.9, color=f'C{idx + 3}'
                        )
        else:
            df_l = df[df.index.str[0] == 'l'].assign(
                position=lambda x: x.index.str[1:].astype(int)
            ).sort_values(
                'position'
            )
            for label, marg_type in zip(labels, marg_types):
                ax_cdr3.plot(
                    df_l['position'].values,
                    df_l[f'{marg_type}_marginal'].values,
                    label=label, alpha=0.9
                )

        ax_cdr3.tick_params(rotation=90, axis='x')
        ax_cdr3.grid()
        ax_cdr3.legend(framealpha=1, fontsize=12)
        ax_cdr3.set_title('CDR3 LENGTH DISTRIBUTION',fontsize=20)

        ax_j = plt.subplot(122)
        df_j = df[df.index.str[0] == 'j'].sort_values(
            'model_marginal', ascending=False
        )
        for label, marg_type in zip(labels, marg_types):
            ax_j.scatter(
                df_j.index.values, df_j[f'{marg_type}_marginal'].values,
                label=label, alpha=0.9, edgecolor='k', linewidth=0.75,
                zorder=10
            )
        ax_j.tick_params(rotation=90, axis='x')
        ax_j.grid()
        ax_j.legend(framealpha=1, fontsize=12)
        ax_j.set_title('J USAGE DISTRIBUTION',fontsize=20)
        ax_j.set_xlim(-1, len(df_j))
        fig.tight_layout()
        if save_name is not None:
            fig.savefig(save_name.split('.')[0]+'_jl.'+save_name.split('.')[1])

        fig, ax_v = plt.subplots(figsize=(16,4))
        df_v = df[df.index.str[0] == 'v'].sort_values(
            'model_marginal', ascending=False
        )
        for label, marg_type in zip(labels, marg_types):
            ax_v.scatter(
                df_v.index.values, df_v[f'{marg_type}_marginal'].values,
                label=label, alpha=0.9, edgecolor='k', linewidth=0.75,
                zorder=10
            )
        ax_v.tick_params(rotation=90, axis='x')
        ax_v.set_xlim(-1, len(df_v))
        ax_v.grid()
        ax_v.legend(framealpha=1, fontsize=12)
        ax_v.set_title('V USAGE DISTRIBUTION',fontsize=20)
        fig.tight_layout()
        if save_name is not None:
            fig.savefig(save_name.split('.')[0]+'_v.'+save_name.split('.')[1])
        else: plt.show()

    def plot_logQ(
        self,
        save_name: Optional[str] =None
    ) -> None:
        """
        Plot logQ of data and generated sequences

        Parameters
        ----------
        save_name : str, optional
            File name to save output figure. If None (default) does not save.

        Returns
        -------
        None
        """
        try:
            self.sonia_model.energies_gen
            self.sonia_model.energies_data
        except:
            self.sonia_model.energies_gen = (
                self.sonia_model.compute_energy(self.sonia_model.gen_encoding)
                + np.log(self.sonia_model.Z)
            )
            self.sonia_model.energies_data = (
                self.sonia_model.compute_energy(self.sonia_model.data_encoding)
                + np.log(self.sonia_model.Z)
            )

        fig = plt.figure(figsize=(8,4))

        bins = np.linspace(
            -self.sonia_model.max_energy_clip - 1,
            -self.sonia_model.min_energy_clip + 1,
            100
        )
        bin_centers = (bins[:-1] + bins[1:]) / 2

        hist_gen, _ = np.histogram(
            -self.sonia_model.energies_gen, bins, density=True
        )
        hist_data, _ =np.histogram(
            -self.sonia_model.energies_data, bins, density=True
        )

        plt.plot(bin_centers, hist_gen, label='generated')
        plt.plot(bin_centers, hist_data, label='data')
        plt.ylabel('density', fontsize=20)
        plt.xlabel('log Q', fontsize=20)
        plt.legend(fontsize=20)
        plt.tight_layout()

        if save_name is not None:
            fig.savefig(save_name)
        else: plt.show()

    def plot_ratioQ(
        self,
        save_name: Optional[str] = None
    ) -> None:
        """
        Plot the ratio of the marginals in data and pre-selected pool against
        the energies learned by the model.

        Useful for model validation.

        Parameters
        ----------
        save_name : str, optional
            File name to save output figure. If None (default) does not save.

        Returns
        -------
        None
        """
        self.sonia_model.energies_gen = (
            self.sonia_model.compute_energy(self.sonia_model.gen_encoding)
            + np.log(self.sonia_model.Z)
        )
        self.sonia_model.energies_data = (
            self.sonia_model.compute_energy(self.sonia_model.data_encoding)
            + np.log(self.sonia_model.Z)
        )

        fig = plt.figure(figsize=(8,8))
        bins = np.logspace(-11, 5, 300)
        bin_centers = (bins[:-1] + bins[1:]) / 2
        a, _ = np.histogram(
            np.exp(-self.sonia_model.energies_gen), bins, density=True
        )
        c, _ = np.histogram(
            np.exp(-self.sonia_model.energies_data), bins, density=True
        )

        ratio = np.zeros(shape=len(c)) + np.nan
        ratio = np.divide(c, a, where=a > 0, out=ratio)
        ratio[(c > 0) & (a == 0)] = 1e9

        plt.plot([-1, 1000], [-1, 1000], c='k')
        plt.xlim([0.001, 200])
        plt.ylim([0.001, 200])
        plt.plot(bin_centers, ratio, c='r', linewidth=3, alpha=0.9)
        plt.xscale('log')
        plt.yscale('log')
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)
        plt.xlabel('Q', fontsize=30)
        plt.ylabel('$P_{data}(Q)/P_{pre}(Q)$', fontsize=30)
        plt.grid()

        plt.tight_layout()
        if save_name is not None:
            fig.savefig(save_name)
        else: plt.show()
