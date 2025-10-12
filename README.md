# **Grex**: Automatic **Gr**ammar **Ex**traction

This is the maintained repository of Grex, a tool for the automatic extraction of grammatical rules from treebanks using machine learning.

It was conceived by Santiago Herrera, Caio Corro, Bruno Guillaume and Sylvain Kahane. A full description of the method can be found in the paper (the code is deprecated): https://arxiv.org/abs/2403.17534

Currently, the repository contains the code to use Grex as CLI tool. 

If you use this software, please cite the following work:

```
@inproceedings{herrera_et_al2024grex,
    title = "Sparse Logistic Regression with High-order Features for Automatic Grammar Rule Extraction from Treebanks",
    author = "Herrera, Santiago and Corro, Caio and Kahane, Sylvain",
    booktitle = "Proceedings of the 2024 Joint International Conference on Computational Linguistics, Language Resources and Evaluation",
    month = may,
    year = "2024",
    address = "Torino, Italia",
    url = "https://arxiv.org/abs/2403.17534",
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

The documentation is available [here](https://grex.grew.fr).
Alternatively, it can be run locally by following the instructions provided in this [repository](https://github.com/Autogramm/grex-doc).

## Installation

Clone the project repository
```bash
git clone https://github.com/Autogramm/grex
cd grex
```

Create a virtual environment
```bash
python3 -m venv .venv
source .venv/bin/active
```

Install python packages
```pip
pip install -r requirements.txt
```

Grex use [Grew](https://grew.fr/), a rewriting graph library, which allows writing very flexible queries over graphs. Grew is implemented in Ocaml.

- Install grewpy: https://grew.fr/usage/python/

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

