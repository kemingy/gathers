# Gathers Python

## Installation

```bash
pip install gathers
```

## Usage

```python
from gathers import kmeans_fit
import numpy as np


data = np.array([[1, 2], [1, 4], [1, 0],
                 [4, 2], [4, 4], [4, 0]])
centroids = kmeans_fit(data, 2)
```
