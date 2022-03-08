<img src="docs/logo1.jpg" width="150" />


SoNNia is a python 3.6 software which extends the functionality of the [SONIA](https://github.com/statbiophys/SONIA) package.  It  expands  the  choice  of selection  models that can be inferred. Non linear single-chain models and (non-)linear paired-chain models are included in the package. The pre-processing pipeline implemented in the corresponding paper is also included as a separate class. Finally the likelihood ratio classifier and a linear logistic classifier for functional annotation are also included and can be directly applied to T- and B-cell receptor repertoire datasets.
![image](docs/summary_fig.png)


## Version
Latest released version: 0.1.2

## Installation
SoNNia is a python /3.6 software. It is available on PyPI and can be downloaded and installed through pip:

 ```pip install sonnia```.

SoNNia is also available on [GitHub](https://github.com/statbiophys/sonnia). The command line entry points can be installed by using the setup.py script:

 ```python setup.py install```.
 
Sometimes pip fails to install the dependencies correctly. Thus, if you get any error try first to install the dependencies separately:
 ```
pip install tensorflow
pip install matplotlib
pip install olga
pip install sonia 
 ```

## References
Isacchini G, Walczak AM, Mora T, Nourmohammad A, Deep generative selection models of T and B cell receptor repertoires with soNNia, (2021) PNAS, https://www.pnas.org/content/118/14/e2023141118.short

## SoNNia modules in a Python script
In order to incorporate the core algorithm into an analysis pipeline (or to write your own script wrappers) all that is needed is to import the modules. Each module defines some classes that only a few methods get called on.

The modules are:

| Module name                                    | Classes                                          |    
|------------------------------------------------|--------------------------------------------------|
| evaluate_model_paired.py                       | EvaluateModel                                    |
| sequence_generation_paired.py                  | SequenceGeneration                               |
| sonia_paired.py                                | SoniaPaired                                      |
| sonnia_paired.py                               | SoNNiaPaired                                     |
| sonnia_transfer.py                             | SoNNiaTransfer                                   |
| sonnia.py                                      | SoNNia                                           |
| utils.py                                       | N/A (contains util functions)                    |
| processing.py                                  | Processing                                       |
| classifiers.py                                 | Linear, SoniaRatio                               |

The classes EvaluateModel, SequenceGeneration, SoniaPaired, SoNNiaPaired, SoNNiaTransfer and SoNNia have similar behaviour to the ones defined in the [SONIA](https://github.com/statbiophys/SONIA) package and we refer to its documentation for the usage.
As an example, the basic import and initialization of the single-chain SoniaLeftposRightpos model
```
from sonia.sonia_leftpos_rightpos import SoniaLeftposRightpos
qm=SoniaLeftposRightpos()
```
translates into the deep version as 
```
from sonnia.sonnia import SoNNia
qm=SoNNia()
```
translates into the linear paired-chain (i.e. alpha-beta for TCRs) version as 

```
from sonnia.sonia_paired import SoniaPaired
qm=SoniaPaired()
```
translates into the deep paired (i.e. alpha-beta for TCRs) version as 

```
from sonnia.sonnia_paired import SoNNiaPaired
qm=SoNNiaPaired()

```
SoNNia keeps all the functionalities of SONIA. As an example you can infer a linear SONIA model with SoNNia using the following definition of the model:

```
from sonnia.sonnia import SoNNia
qm=SoNNia(deep=False, include_joint_genes=True, include_indep_genes=False)

```
 The above commands specifies that the network architecture must be linear and that the V and J genes should form joint features.

In the examples folder there is a python notebook  (or alternatively the example_pipeline script) which shows the main properties of the software. The fig2_paper folder contains all scripts and explanations needed to reproduce figure 2 of the soNNia paper.

## Command line console scripts

There are three command line console scripts (the scripts can still be called as executables if SoNNia is not installed):
1. ```sonnia-evaluate```
  * evaluates Ppost, Pgen or selection factors of sequences according to a generative V(D)J model and selection model.
2. ```sonnia-generate```
  * generates CDR3 sequences, before (like olga) or after selection
3. ```sonnia-infer```
  * infers a selection model with respect to a generative V(D)J model

For any of them you can execute with the -h or --help flags to get the options.

###
We offer a quick demonstration of the console scripts. This will show how to generate and evaluate sequences and infer a selection model using the default generation model for human TCR beta chains that ships with the SONIA software. In order to run the commands below you need to download the examples folder. 

1. ```$ sonnia-infer --humanTRB -i examples/data_seqs.txt -d ';' -m 10000```
  * This reads in the full file example_seqs.txt, infers a selection model and saves to the folder sel_model


2. ```$ sonnia-generate --set_custom_model_VDJ examples/sonnia_model --post -n 100```
  * Generate 100 human TRB CDR3 sequences from the post-selection repertoire and print to stdout along with the V and J genes used to generate them.
3. ```$ sonnia-evaluate --set_custom_model_VDJ examples/sonnia_model -i examples/data_seqs.txt --ppost -m 100 -d ';' ```
  * This computes Ppost,Pgen and Q of the first 100 seqs in the data_seqs file.  

# Notes about CDR3 sequence definition and Dataset size

This code is quite flexible, however it does demand a very consistent definition of CDR3 sequences.

CHECK THE DEFINITION OF THE CDR3 REGION OF THE SEQUENCES YOU INPUT. This will likely be the most often problem that occurs.

The default models/genomic data are set up to define the CDR3 region from the conserved cysteine C (INCLUSIVE) in the V region to the conserved F or W (INCLUSIVE) in the J. This corresponds to positions X and X according to IMGT. 

Neural Network models suffer from overfitting issues in the low data regime. While the use of appropriate regularization could reduce the risk of overfitting, it is recommended to use the linear SONIA model for datasets with fewer than 100 000 receptor sequences.


## Contact

Any issues or questions should be addressed to [us](mailto:giulioisac@gmail.com).

## License

Free use of soNNia is granted under the terms of the GNU General Public License version 3 (GPLv3).
