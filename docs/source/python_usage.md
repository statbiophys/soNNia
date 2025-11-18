## SoNNia modules in a Python script

In order to incorporate the core algorithm into an analysis pipeline (or to write your own script wrappers) all that is needed is to import the modules. Each module defines some classes that only a few methods get called on.

The modules are:

| Module name                                    | Classes                                          |    
|------------------------------------------------|--------------------------------------------------|
| sonia.py                                       | Sonia                                            |
| sonnia.py                                      | SoNNia                                           |
| sonia_paired.py                                | SoniaPaired                                      |
| sonnia_paired.py                               | SoNNiaPaired                                     |
| classifiers.py                                 | Linear, SoniaRatio                               |
| compare_repertoires.py                         | Compare                                          |
| plotting.py                                    | Plotter                                          |
| utils.py                                       | N/A (contains util functions)                    |

The classes SoniaPaired, SoNNiaPaired, and SoNNia have similar behaviour to the ones defined in the `SONIA <https://github.com/statbiophys/SONIA>`_ package.

### Basic Usage Examples

**Linear single-chain model (equivalent to SONIA):**

```python
from sonnia.sonia import Sonia
qm = Sonia()
```

**Deep non-linear single-chain model:**

```python
from sonnia.sonnia import SoNNia
qm = SoNNia()
```

**Linear paired-chain model (e.g., alpha-beta for TCRs or heavy-light for BCRs):**

```python
from sonnia.sonia_paired import SoniaPaired
qm = SoniaPaired()
```

**Deep non-linear paired-chain model:**

```python
from sonnia.sonnia_paired import SoNNiaPaired
qm = SoNNiaPaired()
```

### Additional Utility Classes

**Plotting and visualization:**

```python
from sonnia.plotting import Plotter
pl = Plotter(sonia_model)
pl.plot_model_learning("learning_curve.png")
pl.plot_vjl("marginals.png")
```

**Repertoire comparison:**

```python
from sonnia.compare_repertoires import Compare
comparator = Compare(pgen_model="humanTRB", data=["data1.csv", "data2.csv"])
comparator.infer_models()
comparator.compute_distances()
```