Title: mypy 檢查不出的 pandas 型別，也許你可以試試 pyre-check
Date: 2021-01-02 12:57
Category: Tech
Tags:
Slug: pyre-check
Authors: Lee-W
Status: draft


```python
# example.py
import numpy as np
import pandas as pd


def pd_func() -> int:
    return pd.DataFrame()


def np_func() -> int:
    return np.array([1, 2, 3])
```



```sh
mypy example.py
```

```txt
example.py:2: error: Skipping analyzing 'numpy': found module but no type hints or library stubs
example.py:2: note: See https://mypy.readthedocs.io/en/latest/running_mypy.html#missing-imports
example.py:3: error: Skipping analyzing 'pandas': found module but no type hints or library stubs
Found 2 errors in 1 file (checked 1 source file)
```

```txt
Success: no issues found in 1 source file
```

* [data-science-types](https://github.com/predictive-analytics-lab/data-science-types)
    * [ENH: add type stubs from numpy-stubs #16515](https://github.com/numpy/numpy/pull/16515)
    * [adding static type checking with mypy #14468](https://github.com/pandas-dev/pandas/issues/14468)


* [pytype](https://github.com/google/pytype): Google
* [pyright](https://github.com/Microsoft/pyright): Microsoft
* [pyre-check](https://pyre-check.org/): Facebook

可以參考 [Introduce several Python type checking tools](https://developpaper.com/introduce-several-python-type-checking-tools/)

https://numpy.org/devdocs/release/1.20.0-notes.html#numpy-is-now-typed
