# Writeprints Features Extractor

This python package can help extract features from a text document based on the paper [Writeprints: A Stylometric Approach to
Identity-Level Identification and Similarity
Detection in Cyberspace.](https://www.scss.tcd.ie/Khurshid.Ahmad/Research/Sentiments/K_Teams_Buchraest/a7-abbasi.pdf)

Code was adopted from the [Extended-Writeprints](https://github.com/asad1996172/Extended-Writeprints) repository.

## Installation from PyPi
To install from PyPi, run following command:
```
pip install writeprints
```
## Installation from Source
To manually install from the github repository, clone the repository, go into the directory and run:
```
pip install ./
```

## Usage
To extract features from a single text document contained in a python string:
```python
from writeprints.text_processor import Processor
processor = Processor (flatten = False) # Flatten will split vectorized features into individual featurs
features = processor.extract(string)
```

To extract features from a pandas data frame in which a column named "text" contains the required text documents:

```python
from writeprints.text_processor import Processor
processor = Processor (flatten = False) # Flatten will split vectorized features into individual featurs
features = processor.extract_df(df)
```

License
----

MIT
