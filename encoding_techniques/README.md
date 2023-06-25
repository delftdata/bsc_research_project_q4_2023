# Encoding Methods for Categorial Data
The code in this folder was developed during the **CSE3000 Research Project** at **TU Delft**. 
The research project is:

 **Encoding Methods for Categorical Data: A Comparative Analysis for Linear
Models, Decision Trees, and Support Vector Machines**


This readme will provide guidance towards running the code and replicating the results presented in the research paper.

# Requirements
The requirements are provided in the file ***requirements.txt***. Creating a separate environment is recommended. To get started, run 

```
pip install -r "requirements.txt"
```
For MacOS users, follow [this](https://github.com/autogluon/autogluon/issues/1442) guide in case of any errors regarding LightGBM.


# Table Of Contents
-  [Encoding Methods](#encoding-methods)
-  [ML Algorithms](#ml-algorithms)
-  [Datasets](#datasets)
-  [Running the Experiments](#running-the-experiments)
-  [Reading the Results](#reading-the-results)
-  [Plotting the Results](#plotting-the-results)

# Encoding Metohds 
The code for the five encoders used is located in encoders/encoders.py. There are separate classes for each encoder. An instance of an encoder can be created by importing this implementation.


# ML Algorithms
The code for the ML algorithms is located in models/models.py. 

The class AutogluonModel can implement any of the algorithms supported by Autogluon. In the case of this project, the algorithms used were LinearModel, XGBoost, LightGBM and RandomForest.

The class SVMModel implements a Support Vector Machine from SkLearn.


# Datasets
All the datasets used can be found in the root folder of the repository.

# Running the Experiments
The experiments were performed per dataset, meaning that there two files for each dataset:
- **run_dataset-name.py** - this file loads the dataset, performs any preprocessing needed (data imputation, target labels manipulation, grid search), encodes the datasets with all the encoders and runs the ML models.
- **run_dataset-name_combined.py** - this file performs the same experiment, with the only difference being that for each column in the dataset, the best-performing encoder is chosen. For each column, tests are performed with each encoder, while the remaining columns are always one-hot encoded. Finally, the best-performing encoder for each column is saved.

Also, file ***run_datasets_combined.py*** implements the experiments for all datasets with the approach of combining encoders. The heuristic for combining the encoders is defined in ***run_combined_test.py***.

# Reading the Results
The location where the results are saved is defined in models/models.py, in the evaluation method.
The location is the folder ***results***, for results regarding using one encoder for the whole datasets, and ***combined_results*** for the results regarding combining multiple encoders on one dataset. The folders contain subfolders for each dataset.

# Plotting the Results
The code for plotting the results is available in **figures/graph.py** and **figures/graph_combined.py**.



