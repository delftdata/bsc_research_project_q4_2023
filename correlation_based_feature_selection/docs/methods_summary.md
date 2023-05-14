| Name/Types of feature | Discrete | Continuous | Nominal | Ordinal |
|-----------------------|----------|------------|---------|---------|
| Pearson               |          | x          |         |         |
| Spearman              |          | x          |         | x       |
| Cramer's V            |          |            | x       |         |
| Symmetric Uncertainty |          |            | x       | x       |

Approach 1: Work only with continuous (Pearson+Spearman) and nominal features(Cramer's V+SU)
* Discrete -> Continuous: MixMax Scaling
* Continuous -> Nominal: KBinsDiscretizer
* Nominal -> Discrete: OneHotEncoding
* Ordinal -> Discrete: OrdinalEncoding (e.g. encode labels to increasing integers)

* Large # of features -> might be difficult to analyze each feature if necessary?

Approach 2: Ignore the features in the data set that are not the desired type