# Redundancy analysis for paper "AutoFeat: Transitive Feature Discovery over Join Paths"

## Introduction
Here we evaluate the five information-theoretical-based feature selection methods:

- Mutual Information Feature Selection (MIFS) [1]
- Minimum Redundancy Maximum Relevance (MRMR) [4]
- Conditional Infomax Feature Extraction (CIFE) [3]
- Joint Mutual Information (JMI) [5]
- Conditional Mutual Info Maximisation (CMIM) [2]

The code used to conduct the empirical analysis can be run by using `main.py` file and the associated with it helper files in `skfeature` folder.

The results that came from this evaluation are in `results` folder, where we provide both the logs generated from the evaluation and the plots used in the research paper.

## Datasets

The main sources for finding datasets were [OpenML](https://www.openml.org/), [Kaggle](https://www.kaggle.com/) and
[UC Irvine Machine Learning Repository](https://archive.ics.uci.edu/).

| Task                  | Size            | Dataset name            | Source                                                                                                         | #Instances | #Features (excl. target) | Considered values for κ                              |
|-----------------------|-----------------|-------------------------|----------------------------------------------------------------------------------------------------------------|------------|--------------------------|------------------------------------------------------|
| Binary classification | Small (< 100)   | Breast Cancer           | [Link](https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data)                                     | 569        | 31                       | 5, 10, 20, 30                                        |
| Binary classification | Small (< 100)   | SPAM E-mail             | [Link](https://www.openml.org/search?type=data&status=active&id=44)                                            | 4601       | 57                       | 5, 10, 20, 30, 40, 50                                |
| Binary classification | Medium (< 1000) | Musk                    | [Link](https://www.openml.org/search?type=data&status=active&id=1116)                                          | 6598       | 169                      | 5, 10, 25, 50, 100, 150                              |
| Binary classification | Medium (< 1000) | Arrhythmia              | [Link](https://www.openml.org/search?type=data&status=active&id=1017)                                          | 452        | 279                      | 5, 10, 25, 50, 100, 150, 200, 250                    |
| Binary classification | Large (< 10k)   | Internet Advertisements | [Link](https://archive.ics.uci.edu/ml/datasets/Internet+Advertisements)                                        | 3279       | 1558                     | 5, 10, 25, 50, 100, 250, 500, 1000, 1500             | 
| Binary classification | Large (< 10k)   | Gisette                 | [Link](https://archive.ics.uci.edu/ml/datasets/Gisette)                                                        | 6000       | 5000                     | 5, 10, 25, 50, 100, 250, 500, 1000, 2000, 3000, 4000 |


## How to run the code
 
1. In order to run the code here you need to also make use of the datasets from this repository at its root.
2. Make use of the `docker-compose` file in this folder to start the feature selection. You can start it in a detached mode using the following command:
```
docker-compose up -d --build
```
3. Get into the container and start the needed functions in `main.py`
   1. To get into the docker container, use the following command where you replace `<container-name>` with the name of the container you started in step 2:
   ```
   docker exec -it <container-name> bash
   ```
   2. Navigate to the `scikit-feature` folder and start the main experiment using the command `python main.py`. 

The `main()` function in `scikit-feature/main.py` is split in multiple parts:
1. Initially we select the datasets to use in our feature selection analysis in place them in `datasets` list.
2. Perform feature selection using the five information-theoretical-based methods. The outcome of the analysis is stored in a csv file. In the event that you do not wish to use all entries of this file in your analysis, you can manually filter the relevant rows. For example, we only considered the number of features `k` as described in [Datasets](#datasets) section.
3. Evaluate the resulting subsets of datasets using LightGBM algorithm from AutoGluon.

## Plots

In order to recreate the redundancy plot that was included in the AutoFeat paper, you can take the following steps:

1. Run the experiment as described in [How to run the code](#how-to-run-the-code) section. Alternatively, you can use our results stored in `results/performance.csv`
2. Open `visualisations.ipynb` and run the first and second cells to obtain the redundancy plot.

## References

[1] Roberto Battiti. Using Mutual Information for Selecting Features in Supervised Neural Net Learning. IEEE trans. neural netw. 5:537–550, 07 1994.

[2] Francois Fleuret, “Fast Binary Feature Selection with Conditional Mutual Information François Fleuret,” Journal of Machine Learning Research, vol. 5, 2004.

[3] Dahua Lin and Xiaoou Tang. Conditional infomax learning: An integrated framework for feature extraction and fusion. ECCV, 9:68–82, 01 2006.

[4] Hanchuan Peng, Fuhui Long, and Chris Ding. Feature Selection Based on Mutual Information Criteria of Max-Dependency, Max-relevance, and Min-Redundancy”. IEEE TPAMI, 27:1226–1238, 08 2005.

[5] Howard Yang and John Moody. Data Visualization and Feature Selection: New Algorithms for Nongaussian Data. In: Proceedings of NIPS. Vol. 12., 1999

## Maintainer

This repository is created and maintained by [Kiril Vasilev](https://www.linkedin.com/in/kiril-vasilev/).