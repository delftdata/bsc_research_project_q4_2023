# Relevance analysis for paper "AutoFeat: Transitive Feature Discovery over Join Paths"

# Introduction

* AutoFeat considers the following correlation metrics in order to decide which one best assesses relevance:
   * Information Gain [1]
   * Symmetrical Uncertainty [2]
   * Pearson [3]
   * Spearman [4]
   * Relief [5]

* We implement the **Select κ best** heuristic approach for feature selection [3]. **Select κ best** sorts the features 
based on their correlation score with the target feature, then selects the top-κ performers. In order to calculate each 
of the five correlation metrics, different sources have been used:
 
| Method                  | Implementation source                                                                     |
|-------------------------|-------------------------------------------------------------------------------------------|
| Information Gain        | [scikit-feature](https://github.com/jundongl/scikit-feature)                              | 
| Symmetrical Uncertainty | [scikit-feature](https://github.com/jundongl/scikit-feature)                              |
| Pearson                 | [SciPy](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.pearsonr.html)   |
| Spearman                | [SciPy](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.spearmanr.html)  |
| Relief                  | [ITMO_FS](https://github.com/ctlab/ITMO_FS/tree/a2e61e2fabb9dfb34d90a1130fc7f5f162a2c921) |

# Installation

Prior to installing the project, ensure that you have
[Python 3.10](https://www.python.org/downloads/release/python-3100/) available on your device.

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

To perform the experiments, use the following command from the `autofeat_experimental_analysis\relevance_analysis` 
folder:
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

# References

[1] J. Li, K. Cheng, S. Wang, F. Morstatter, R. P. Trevino, J. Tang, and H. Liu, “Feature selection: A 
data perspective,” ACM computing surveys (CSUR), vol. 50, no. 6, pp. 1–45, 2017.   
[2] L. Yu and H. Liu, “Feature selection for high-dimensional data: A fast correlation-based filter solution,” in ICML, 
2003, pp. 856–863.   
[3] I. Guyon and A. Elisseeff, “An introduction to variable and feature selection,” Journal of machine learning 
research, vol. 3, no. Mar, pp. 1157–1182, 2003.   
[4]  J. de Winter, S. Gosling, and J. Potter, “Comparing the pearson and spearman correlation coefficients across 
distributions and sample sizes: A tutorial using simulations and empirical data,” Psychological Methods, vol. 21, 
pp. 273–290, 2016.   
[5] R. J. Urbanowicz, M. Meeker, W. La Cava, R. S. Olson, and J. H. Moore, “Relief-based feature selection: 
Introduction and review,” Journal of biomedical informatics, vol. 85, pp. 189–203, 2018.
