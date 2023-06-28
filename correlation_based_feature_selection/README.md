# RQ3: "Data-Driven Empirical Analysis of Correlation-Based Feature Selection Techniques"

### Introduction
* Our Research paper focuses on analysis four correlation-based feature selection methods:
- Pearson
- Spearman
- Cramer's V
- Symmetric Uncertainty

### Installation
Prior to installing the project, ensure that you have
[Python 3.10](https://www.python.org/downloads/release/python-3100/) installed on your machine.

To install the project, follow these steps:
1. For **manual** installation, visit the GitHub repository and click on the **<> Code ▼** button,
   then select _Download ZIP_.

    For **Git installation**, use one of the following commands:
    - **HTTPS**:
        > git clone `https://github.com/delftdata/bsc_research_project_q4_2023.git`
    - **SSH**:
        > git clone `git@github.com:delftdata/- bsc_research_project_q4_2023.git`

    For **GitHub CLI installation**, use the command:
    > gh repo clone `delftdata/bsc_research_project_q4_2023`

2. Once the project is installed, navigate to the source folder:
    > cd .\correlation_based_feature_selection\

3. Install the project requirements by running:
    > pip install -r `requirements.txt`

### Run

To perform the experiments, choose the desired method in `src/run.py` and use the command from the `correlation_based_feature_selection` folder:
   > python -m src.run