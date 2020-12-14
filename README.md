<img src="docs/logo1.jpg" width="150" />


SoNNia is a python 3.6 software which extends the functionality of the [SONIA](https://github.com/statbiophys/SONIA) package.  It  expands  the  choice  of selection  models that can be inferred. Non linear single-chain models and (non-)linear paired-chain models are included in the package. The pre-processing pipeline implemented in the corresponding paper is also included as a separate class. Finally the likelihood ratio classifier and a linear logistic classifier for functional annotation are also included and can be directly applied to T- and B-cell receptor repertoire datasets.
![image](docs/summary_fig.png)


## Version
Latest released version: 0.0.1

## Installation
SONIA is a python /3.6 software. The package can be installed by using the setup.py script:

 ```python setup.py install```.
 
Sometimes pip fails to install the dependencies correctly. Thus, if you get any error try first to install the dependencies separately:
 ```
pip install tensorflow
pip install matplotlib
pip install olga
pip install sonia 
 ```

## References
Isacchini G, Walczak AM, Mora T, Nourmohammad A, Deep generative selection models of T and B cell receptor repertoires with soNNia, (2020) bioRxiv, https://www.biorxiv.org/content/10.1101/2020.11.05.370346v1

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

In the examples folder there is a python notebook  (or alternatively the example_pipeline script) which shows the main properties of the software.

## Contact

Any issues or questions should be addressed to [us](mailto:giulioisac@gmail.com).

## License

Free use of soNNia is granted under the terms of the GNU General Public License version 3 (GPLv3).