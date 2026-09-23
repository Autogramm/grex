# **Grex**: Automatic **Gr**ammar **Ex**traction

This is the maintained repository of Grex, a tool for the automatic extraction of grammatical rules from treebanks using machine learning.

This is the branch has the **WIP library version of Grex**.

- Restructured as an installable library.
- Code moved from flat scripts to a proper src/grex package (pip install -e . via pyproject.toml, replacing requirements.txt).
- Rule-extraction methods (DTreeExtractor, LassoExtractor) now share a common Extractor interface and are importable directly (from grex import DTreeExtractor, LassoExtractor).
- The three CLI scripts (extract_rules_via_dtree.py, extract_rules_via_lasso.py, check_features.py) still work the same way, just moved under src/grex/cli/. 
- This is just an reorganization. No change to algorithms, output format, or statistics.

## Installation

Clone the project repository
```bash
git clone https://github.com/Autogramm/grex
cd grex
```

Create a virtual environment
```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install python packages
```bash
pip install -e .
```

Or, for the dev version for ruff/pytest

```bash
pip install -e .[dev]
```

Grex use [Grew](https://grew.fr/), a rewriting graph library, which allows writing very flexible queries over graphs. Grew is implemented in Ocaml.

- Install grewpy: https://grew.fr/usage/python/


## Usage

### 1. As a library

Grex can be used as a Python library instead of through the CLI:

```python
from grex.data.data import ExtractionConfig, CorpusData
from grex import LassoExtractor

config = ExtractionConfig.from_yaml("patterns.yaml")
corpus = CorpusData.from_corpus(["corpus.conllu"], config, grew_config="sud")
matrix = corpus.to_matrix(max_degree=2, min_feature_occurence=5)

extractor = LassoExtractor(config.scope, config.conclusion, alpha_num=100)
extractor.fit(matrix.X, matrix.binarize(config.positive), matrix.feature_names)

for rule in extractor.rules():
    print(rule)
```

### 2. CLI usage

Grex has also three CLI entry points under `src/grex/cli/`. Each takes one or more `.conllu` treebank files and a pattern config YAML (see `examples/` for sample configs).

Check which features are available for a pattern config before running a full extraction:

```bash
python3 src/grex/cli/check_features.py <corpus.conllu> \
    --patterns <patterns.yaml> \
    --config ud|sud
```

Extract rules using sparse logistic regression (lasso):

```bash
python3 src/grex/cli/extract_rules_via_lasso.py <corpus.conllu> \
    --patterns <patterns.yaml> \
    --output <results.json> \
    --config sud \
    --max-degree 2 \
    --alpha-num 100
```

Extract rules using a decision tree:

```bash
python3 src/grex/cli/extract_rules_via_dtree.py <corpus.conllu> \
    --patterns <patterns.yaml> \
    --output <results.json> \
    --config sud \
    [--only-leaves] [--grew]
```

Run the examples:

```bash
cd examples && sh run.sh
```

### 3. Lib layout

- `src/grex/data/` — reading and annotating treebanks, building feature matrices.
- `src/grex/extract/` — rule/pattern extraction methods (`dtree`, `lasso`, ...), all implementing the shared `Extractor` interface in `extract/base.py`. Adding a new method means adding a new file here.
- `src/grex/viz/` — visualization functions.
- `src/grex/utils/` — shared utilities.
- `src/grex/cli/` — thin command-line wrappers around the library.

## References

It was conceived by Santiago Herrera, Caio Corro, Bruno Guillaume and Sylvain Kahane. A full description of the method can be found in the paper (the code is deprecated): https://arxiv.org/abs/2403.17534.

If you use this software, please cite the following work:

```
@inproceedings{herrera-etal-2024-sparse,
    title = "Sparse Logistic Regression with High-order Features for Automatic Grammar Rule Extraction from Treebanks",
    author = "Herrera, Santiago  and Corro, Caio  and Kahane, Sylvain",
    editor = "Calzolari, Nicoletta  and Kan, Min-Yen  and Hoste, Veronique  and Lenci, Alessandro  and Sakti, Sakriani  andXue, Nianwen",
    booktitle = "Proceedings of the 2024 Joint International Conference on Computational Linguistics, Language Resources and Evaluation (LREC-COLING 2024)",
    month = may,
    year = "2024",
    address = "Torino, Italia",
    publisher = "ELRA and ICCL",
    url = "https://aclanthology.org/2024.lrec-main.1314/",
    pages = "15114--15125",
}
```

If you use Grex for contrastive studies, please cite the most pertinent of the following papers:

```
@inproceedings{herrera_et_al2024contrastive,
  author = {Santiago Herrera and Ioana-Madalina Silai and Caio Corro and Bruno Guillaume and Sylvain Kahane},
  title = {Building quantitative contrastive grammars from syntactic treebanks},
  booktitle = {Langues \& Langages à la croisée des Disciplines (LLcD)},
  year = {2024},
  address = {Paris}
}
```

```
@inproceedings{herrera-etal-2025-extraction,
    title = "Extraction of Contrastive Rules from Syntactic Treebanks: A Case Study in {R}omance Languages",
    author = "Herrera, Santiago  and Silai, Ioana-Madalina and Corro, Caio and Guillaume, Bruno  and Kahane, Sylvain",
    editor = "Chen, Xinying  and Wang, Yaqin",
    booktitle = "Proceedings of the Third Workshop on Quantitative Syntax (QUASY, SyntaxFest 2025)",
    month = aug,
    year = "2025",
    address = "Ljubljana, Slovenia",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2025.quasy-1.5/",
    pages = "26--38",
    ISBN = "979-8-89176-293-0"
    }
```
## Documentation

The old documentation is available [here](https://grex.grew.fr).

Alternatively, it can be run locally by following the instructions provided in this [repository](https://github.com/Autogramm/grex-doc).

### Troubleshooting

If you get an error caused by Cython, try installing the python3-dev package.

```bash
apt install python3-dev
```

`distutils` missing?
```bash
pip install setuptools
```

A missing dependency during Grew installation? Install it manually:

```bash
opam install dependency_missing
```

`grewpy_backend` is well installed but not in PATH? Try:

```bash
echo ‘eval $(opam env)’ >> ~/.bashrc
```

