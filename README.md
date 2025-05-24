<img src="docs/logo.jpg" width="150" />

# SoNNia
SoNNia is a Python software package that extends the functionality of [SONIA](https://github.com/statbiophys/SONIA). It expands the range of selection models that can be inferred, including both non-linear single-chain models and linear/non-linear paired-chain models.
SoNNia takes as input CDR3 amino acid sequences, with (or without) V and J genes assignment. Its output is selection factors that can be used to calculate the probability of observing any sequence after selection. The inference is based on maximizing the likelihood of observing a selected data sample given a representative pre-selected sample. This method was first used in Elhanati et al (2014) to study thymic selection. Generally, the pre-selected sample can be generated internally using the OLGA software, but SoNNia also allows it to be supplied externally, in the same way the data sample is provided.

The package provides both command line tools and a Python API for easy integration into analysis pipelines. It ships with several pre-trained models for common use cases like human and mouse TCR and BCR chains. Custom models can also be trained on new datasets.

![image](docs/summary_fig.png)

## Documentation
Extensive documentation can be found [here](https://sonnia.readthedocs.io/en/latest/index.html).

## Installation
SoNNia is a python software. It is available on PyPI and can be downloaded and installed through pip:

 ```pip install sonnia```

For mac user on new metal devices, make sure to install additional dependencies (i.e. tensorflow-metal) to make tensorflow work with the GPU. CPU version is also available and, given the small size of the models, it should be sufficient for most use cases.
SoNNia is also available on [GitHub](https://github.com/statbiophys/sonnia). The command line entry points can be installed by using the setup.py script:

 ```pip install .```

## References
Isacchini G, Walczak AM, Mora T, Nourmohammad A, Deep generative selection models of T and B cell receptor repertoires with soNNia, (2021) PNAS, https://www.pnas.org/content/118/14/e2023141118.short

## Important Notes
### Dataset size requirements
- For datasets < 100,000 clones: Use linear SONIA model
- For larger datasets: Can use non-linear SoNNia models

Using neural networks on small datasets risks overfitting due to the large number of parameters that need to be learned. The neural network may learn noise patterns in the training data rather than true underlying relationships. This is why we recommend using the simpler linear SONIA model for datasets with fewer than 100,000 clones, as it has fewer parameters and is less prone to overfitting. The non-linear SoNNia models are better suited for larger datasets where there is enough data to reliably learn complex patterns.

### CDR3 sequence definition
This code is quite flexible, however it does demand a very consistent definition of CDR3 (junction) sequences.

CHECK THE DEFINITION OF THE CDR3 REGION OF THE SEQUENCES YOU INPUT. This will likely be the most often problem that occurs.

The default models/genomic data are set up to define the CDR3 region (i.e. the junction) from the conserved cysteine C (INCLUSIVE) in the V region to the conserved F or W (INCLUSIVE) in the J.

## Command line console scripts

There are three command line console scripts (the scripts can still be called as executables if SoNNia is not installed):
1. ```sonnia evaluate```
  * evaluates Ppost, Pgen or selection factors of sequences according to a generative V(D)J model and selection model.
2. ```sonnia generate```
  * generates CDR3 sequences, before (like olga) or after selection
3. ```sonnia infer```
  * infers a selection model with respect to a generative V(D)J model

For any of them you can execute with the -h or --help flags to get the options.

### Demonstration of the console scripts
We offer a quick demonstration of the console scripts. This will show how to generate and evaluate sequences and infer a selection model using the default generation model for human TCR beta chains that ships with the SONIA software. In order to run the commands below you need to download the examples folder. 

1. ```$ sonnia infer --model humanTRB -i examples/data_seqs.csv.gz```
  * This reads in the full file example_seqs.csv.gz, infers a selection model and saves to the folder sel_model
2. ```$ sonnia generate --model examples/sonnia_model --post -n 100```
  * Generate 100 human TRB CDR3 (junction) sequences from the post-selection repertoire and print to stdout along with the V and J genes used to generate them.
3. ```$ sonnia evaluate --model examples/sonnia_model -i examples/data_seqs.csv.gz --ppost ```
  * This computes Ppost,Pgen and Q of the first 100 seqs in the data_seqs file.  

## Single Chain Model Types Table
 Model Type | Description | Chain Type |
|------------|-------------|------------|
| humanTRA | Human T cell alpha | VJ |
| humanTRB | Human T cell beta | VDJ |
| humanIGH | Human B cell heavy | VDJ |
| humanIGK | Human B cell kappa | VJ |
| humanIGL | Human B cell lambda | VJ |
| mouseTRB | Mouse T cell beta | VDJ |
| mouseTRA | Mouse T cell alpha | VJ |
| mouseIGH | Mouse B cell heavy | VDJ |
| mouseIGK | Mouse B cell kappa | VJ |
| mouseIGL | Mouse B cell lambda | VJ |

## Python API 
In order to incorporate the core algorithm into an analysis pipeline (or to write your own script wrappers) all that is needed is to import the modules. Each module defines some classes that only a few methods get called on.

The modules are:

| Module name                                    | Classes                                          |    
|------------------------------------------------|--------------------------------------------------|
| sonia_paired.py                                | SoniaPaired                                      |
| sonnia_paired.py                               | SoNNiaPaired                                     |
| sonnia.py                                      | SoNNia                                           |
| sonia.py                                       | Sonia                                            |
| utils.py                                       | N/A (contains util functions)                    |
| classifiers.py                                 | Linear, SoniaRatio                               |

The classes SoniaPaired, SoNNiaPaired, and SoNNia have similar behaviour to the ones defined in the [SONIA](https://github.com/statbiophys/SONIA) package.
As an example, the basic import and initialization of the single-chain SoniaLeftposRightpos model
```
# linear sonia model
from sonia.sonia_leftpos_rightpos import SoniaLeftposRightpos
qm=SoniaLeftposRightpos()

# deep sonia model
from sonnia.sonnia import SoNNia
qm=SoNNia()

# linear paired-chain model (i.e. alpha-beta for TCRs)
from sonnia.sonia_paired import SoniaPaired
qm=SoniaPaired()

# deep paired-chain model (i.e. alpha-beta for TCRs)
from sonnia.sonnia_paired import SoNNiaPaired
qm=SoNNiaPaired()

# linear single-chain model (sonia equivalent)
from sonnia.sonia import Sonia
qm=Sonia()
```

In the examples folder there is a python notebook (or alternatively the example_pipeline script) which shows the main properties of the software, including:

- Loading and preprocessing sequence data
- Training selection models on single-chain data
- Evaluating selection probabilities
- Generating synthetic sequences
- Visualizing selection factors
- Comparing different models

The fig2_paper folder contains all scripts and explanations needed to reproduce figure 2 of the soNNia paper, which demonstrates:

- Training deep and linear models on TCR data
- Comparing selection patterns between models
- Analyzing position-specific amino acid selection
- Evaluating model performance and generalization

The fig4_paper folder contains all scripts and explanations needed to reproduce figure 4 of the soNNia paper, which demonstrates:
- infer model on different datasets
- compare selection patterns between models
- plot distance matrix between datasets 

## Contact

Any issues or questions should be addressed to [us](mailto:giulioisac@gmail.com).

## License

Free use of soNNia is granted under the terms of the GNU General Public License version 3 (GPLv3).
