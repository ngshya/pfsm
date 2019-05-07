# Python Fast Strings Matching

This package provides a fast way to perform strings matching tasks. 

## Installing pfsm

```bash
pip install pfsm
```

## Example

```python
from pfsm import *
import numpy as np

def stringsVsStrings(array_1, array_2):
    l = len(array_1)
    obj = Strings(np.concatenate((np.array(array_1), np.array(array_2))))
    obj.processStrings()
    obj.tfidfM()
    return cosimtop(obj.M[0:l,], obj.M[l:, ].transpose(), 3, 0.8)

M = stringsVsStrings(["Company X", "Bank XXYYZZ"], ["Bank XYZ", "Sample Corporation", "X Company"]).todense()
```

```python
from pfsm import *

def getIdsFromStrings(strings):
    obj_strings = Strings(strings)
    obj_strings.processStrings()
    obj_strings.tfidfM()
    obj_strings.groupByKMeans()
    obj_strings.getEntityID()
    return obj_strings.groups

dtf_strings_ids = getIdsFromStrings(["Company X", "Bank XXYYZZ", "Bank XYZ", "Sample Corporation", "X Company"])
```