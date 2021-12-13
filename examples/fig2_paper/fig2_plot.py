import numpy as np 
import pandas as pd
import matplotlib
matplotlib.use('agg')
from sonia.evaluate_model import EvaluateModel
from sonnia.sonnia import SoNNia
from scipy import stats
import matplotlib.pyplot as plt
plt.style.use('seaborn-white')
import olga.load_model as olga_load_model
import olga.generation_probability as generation_probability
import os
from matplotlib.lines import Line2D
from matplotlib import ticker

#load test data
df=pd.read_csv('sampled_data/test_data.csv.gz')

#evaluated ppost/pgen
q,pgen,ppost=df.q.values,df.pgen.values,df.ppost.values
q_deep,pgen_deep,ppost_deep=df.q_deep.values,df.pgen_deep.values,df.ppost_deep.values

# get rid of pgen=0 seqs and singletons and doubletons
sel=np.logical_and(pgen!=0,df.freq.values>2)
print(np.sum(sel)/len(sel), '% of accepted seqs')

to_compare=['pgen','sonia_linear','sonia_post']
for j,vec in enumerate([pgen,ppost,ppost_deep]):
    r2=[]
    dkls=[]
    for i in range(5):
        subsample=np.random.randint(0,len(pgen),int(len(pgen)/5.))
        selection=sel[subsample]
        log_freq=df.log_freq.values[subsample]
        vec_=vec[subsample]
        r2.append(stats.linregress(log_freq[selection],np.log(vec_[selection]))[2]**2)
        dkls.append(np.mean(log_freq[selection]-np.log(vec_[selection]))/np.log(2))
    print ('R^2',to_compare[j],np.mean(r2),np.std(r2)/np.sqrt(5))
    print ('Dkl',to_compare[j],np.mean(dkls),np.std(dkls)/np.sqrt(5))

#plot everything
def density_scatter( x , y, ax = None, sort = True, bins = 20, **kwargs )   :
    """
    Scatter plot colored by 2d histogram
    """
    if ax is None :
        fig , ax = plt.subplots()
    bins = [100, 500] # number of bins
    # histogram the data
    hh, locx, locy = np.histogram2d(x, y, bins=bins)
    hh=hh#/hh.max()
    z = np.array([hh[np.argmax(a<=locx[1:]),np.argmax(b<=locy[1:])] for a,b in zip(x,y)])
    idx = z.argsort()
    x2, y2, z2 = x[idx], y[idx], z[idx]
    map_reversed = matplotlib.cm.get_cmap('viridis_r')
    s=ax.scatter(x2, y2, c=z2, cmap=map_reversed,s=10,alpha=1.,rasterized=True)
    tick_locator = ticker.MaxNLocator(nbins=4)
    cb=plt.colorbar(s,ax=ax,ticks=tick_locator)
    cb.ax.tick_params(labelsize=20)
    return ax

fig=plt.figure(figsize=(15,6),dpi=200)
ax1=plt.subplot(121)
ax1.set_xlabel('$\log_{10}P_{data}$ ',fontsize=30)
ax1.set_ylabel('$\log_{10}P_{post}$ ',fontsize=30)
ax1.set_ylim([-11,-3])
ax1.set_xlim([-7.5,-3.5])
ax1.locator_params(nbins=4)
ax1.plot([-20,0],[-20,0],c='k')
legend_elements = [Line2D([0], [0], marker='o', color='w', label='LINEAR',
                          markerfacecolor='w', markersize=0)]
ax1.legend(handles=legend_elements,fontsize=20,handletextpad=-2.0)
ax1.set_xticklabels([-8,-7,-6,-5,-4],fontsize=20)
ax1.set_yticklabels([-12,-10,-8,-6,-4],fontsize=20)
density_scatter(df.log_freq.values[sel]/np.log(10),np.log10(np.array(ppost)[sel]),bins = [10,50],ax=ax1)
ax2=plt.subplot(122)
ax2.set_xlabel('$\log_{10}P_{data}$ ',fontsize=30)
ax2.set_ylabel('$\log_{10}P_{post}$ ',fontsize=30)
ax2.set_ylim([-11,-3])
ax2.set_xlim([-7.5,-3.5])
ax2.locator_params(nbins=4)
ax2.plot([-20,0],[-20,0],c='k')
ax2.set_xticklabels([-8,-7,-6,-5,-4],fontsize=20)
ax2.set_yticklabels([-12,-10,-8,-6,-4],fontsize=20)
legend_elements = [Line2D([0], [0], marker='o', color='w', label='DEEP',
                          markerfacecolor='w', markersize=0)]
ax2.legend(handles=legend_elements,fontsize=20,handletextpad=-2.0)
density_scatter(df.log_freq.values[sel]/np.log(10),np.log10(np.array(ppost_deep)[sel]),bins = [10,50],ax=ax2)
plt.tight_layout()
plt.savefig("fig2.pdf")