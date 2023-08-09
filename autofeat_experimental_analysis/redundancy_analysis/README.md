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

## How to run the code
 
1. In order to run the code here you need to also make use of the datasets from this repository at its root.
2. Make use of the `docker-compose` file in this folder to start the feature selection.
3. Get into the container and start the needed functions in `main.py`

## References

[1] Roberto Battiti. Using Mutual Information for Selecting Features in Supervised Neural Net Learning. IEEE trans. neural netw. 5:537–550, 07 1994.

[2] Francois Fleuret, “Fast Binary Feature Selection with Conditional Mutual Information François Fleuret,” Journal of Machine Learning Research, vol. 5, 2004.

[3] Dahua Lin and Xiaoou Tang. Conditional infomax learning: An integrated framework for feature extraction and fusion. ECCV, 9:68–82, 01 2006.

[4] Hanchuan Peng, Fuhui Long, and Chris Ding. Feature Selection Based on Mutual Information Criteria of Max-Dependency, Max-relevance, and Min-Redundancy”. IEEE TPAMI, 27:1226–1238, 08 2005.

[5] Howard Yang and John Moody. Data Visualization and Feature Selection: New Algorithms for Nongaussian Data. In: Proceedings of NIPS. Vol. 12., 1999

## Maintainer

This repository is created and maintained by [Kiril Vasilev](https://www.linkedin.com/in/kiril-vasilev/).