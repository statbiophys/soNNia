## SoNNia modules in a Python script

In order to incorporate the core algorithm into an analysis pipeline (or to write your own script wrappers) all that is needed is to import the modules. Each module defines some classes that only a few methods get called on.

The modules are:

| Module name                                    | Classes                                          |    
|------------------------------------------------|--------------------------------------------------|
| sonia.py                                       | Sonia                                            |
| sonnia.py                                      | SoNNia                                           |
| sonia_paired.py                                | SoniaPaired                                      |
| sonnia_paired.py                               | SoNNiaPaired                                     |
| processing.py                                  | Processing                                       |
| classifiers.py                                 | Linear, SoniaRatio                               |
| compare_repertoires.py                         | Compare                                          |
| utils.py                                       | N/A (contains util functions)                    |

The classes SoniaPaired, SoNNiaPaired, and SoNNia have similar behaviour to the ones defined in the [SONIA](https://github.com/statbiophys/SONIA) package.

As an example, the basic import and initialization of the single-chain SoniaLeftposRightpos model

```
from sonia.sonia_leftpos_rightpos import SoniaLeftposRightpos
qm=SoniaLeftposRightpos()
```

translates here into

```
from sonnia.sonia import Sonia
qm=Sonia()
``` 

Import of the deep version becomes 

```
from sonnia.sonnia import SoNNia
qm=SoNNia()
```

while for linear paired-chain (i.e. alpha-beta for TCRs) is 

```
from sonnia.sonia_paired import SoniaPaired
qm=SoniaPaired()
```

and the deep paired (i.e. alpha-beta for TCRs) version is 

```
from sonnia.sonnia_paired import SoNNiaPaired
qm=SoNNiaPaired()
```