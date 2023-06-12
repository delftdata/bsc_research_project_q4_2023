# Automatic Feature Discovery: A comparative study between filter and wrapper feature selection techniques

## Project Description

The project was created during the **CSE3000 Research Project** at **TU Delft** to conduct the research project
with the same name, ensuring transparency and a high degree of reproducibility for the experimental results.

## Table of Contents

-   [Installation](#installation)
-   [Usage](#usage)

## Installation

Prior to installing the project, ensure that you have
[Python 3.9](https://www.python.org/downloads/release/python-3913/) installed on your machine.

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

    > cd `filter_vs_wrapper_methods/src`

3. Install the project requirements by running:
    > pip install -r `requirements.txt`

## Usage

Ensure that you are in the source folder of the project - `filter_vs_wrapper_methods/src`.

To perform the experiments, use the command:

> python `main.py`

If you wish to plot the results of the experiments, use:

> python `plot.py`

To customize the specific experiment to run or plot, you can modify the corresponding JSON files:
`arguments_main.json` for running experiments and `arguments_plot.json` for plotting.

For detailed information about the supported arguments, please consult the documentation located in
`filter_vs_wrapper_methods/docs/documentation.md`.
