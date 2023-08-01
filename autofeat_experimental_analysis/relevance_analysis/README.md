# Relevance analysis for paper "AutoFeat: Transitive Feature Discovery over Join Paths"

# Introduction

* The AutoFeat paper focuses on the analysis of five relevance metrics:
   * Information Gain
   * Symmetrical Uncertainty
   * Pearson
   * Spearman
   * Relief

# Installation

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
    > cd .\autofeat_experimental_analysis\relevance_analysis\

3. Install the project requirements by running:
    > pip install -r `requirements.txt`

# Experiments

To perform the experiments, use the following command from the `relevance_analysis` folder:
   > python -m src.run_autofeat

## Datasets

| Task                  | Size            | Name                    | Source                                                                                                         | #Instances | #Features (excl. target) | #Features to consider for FS                         |
|-----------------------|-----------------|-------------------------|----------------------------------------------------------------------------------------------------------------|------------|--------------------------|------------------------------------------------------|
| Binary classification | Small (< 100)   | Breast Cancer           | [Link](https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data)                                     | 569        | 31                       | 5, 10, 20, 30                                        |
| Binary classification | Small (< 100)   | SPAM E-mail             | [Link](https://www.openml.org/search?type=data&status=active&id=44)                                            | 4601       | 57                       | 5, 10, 20, 30, 40, 50                                |
| Binary classification | Medium (< 1000) | Musk                    | [Link](https://www.openml.org/search?type=data&status=active&id=1116)                                          | 6598       | 169                      | 5, 10, 25, 50, 100, 150                              |
| Binary classification | Medium (< 1000) | Arrhythmia              | [Link](https://www.openml.org/search?type=data&status=active&id=1017)                                          | 452        | 279                      | 5, 10, 25, 50, 100, 150, 200, 250                    |
| Binary classification | Large (< 10k)   | Internet Advertisements | [Link](https://archive.ics.uci.edu/ml/datasets/Internet+Advertisements)                                        | 3279       | 1558                     | 5, 10, 25, 50, 100, 250, 500, 1000, 1500             | 
| Binary classification | Large (< 10k)   | Gisette                 | [Link](https://archive.ics.uci.edu/ml/datasets/Gisette)                                                        | 6000       | 5000                     | 5, 10, 25, 50, 100, 250, 500, 1000, 2000, 3000, 4000 |
