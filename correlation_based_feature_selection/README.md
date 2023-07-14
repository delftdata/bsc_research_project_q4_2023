# Codebase for "Data-Driven Empirical Analysis of Correlation-Based Feature Selection Techniques"
# Codebase for "AutoFeat: Transitive Feature Discovery over Join Paths"

### Introduction
* The Research paper focuses on the analysis of four correlation-based feature selection techniques:
- Pearson
- Spearman
- Cramér's V
- Symmetric Uncertainty

* Additionally, the AutoFeat paper introduces the analysis of other two correlation techniques:
- Information Gain
- Relief

### Installation
Prior to installing the project, ensure that you have
[Python 3.10](https://www.python.org/downloads/release/python-3100/) installed on your device.

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

To perform the experiments for the Research paper, choose the desired method in `src/run.py` and use the command from the `correlation_based_feature_selection` folder:
   > python -m src.run

To perform the experiments for the AutoFeat paper, choose the desired method in `src/run_autofeat.py` and use the command from the `correlation_based_feature_selection` folder:
   > python -m src.run_autofeat
