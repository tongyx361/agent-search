# AgentSearch


<!-- WARNING: THIS FILE WAS AUTOGENERATED! DO NOT EDIT! -->

## Setup

We recommend:

- using [Conda](https://docs.conda.io/projects/miniconda) to set up the
  **basic Python environment** and install some **non-Python packages**
  like `git-lfs` when **without the root permission**;
- using [`pip`](https://pip.pypa.io/en/stable/#) to manage the **Python
  packages** because some packages are **only available via `pip`**.

Run the following commands to setup the **basic environment** and
**install most dependencies**:

``` shell
git clone https://github.com/tongyx361/agent-search.git
cd agent-search
conda env create -f environment.yml
conda activate agent-search
pip install -r requirements.txt
```

For common users/developers, please just run the following command the
install the `dart-math` package:

``` shell
pip install -e "."
```

For intended contributors, we recommend installing the package with the
`dev` extras and setting up the pre-commit hooks by running:

``` shell
pip install -e ".[dev]"
pre-commit install
```

## Contribution Guidelines

### File Structure

    agent-search
    ├── data
    ├── utils # Repository utilities
    ├── agent_search # Package code for common utilities
    ├── nbs # Notebooks and other files to run tests and generate documentation with https://nbdev.fast.ai
    ├── [pipelines] # Reusable (Python / Shell) scripts or notebooks
    └── [scripts] # One-time scripts

### Checklist Before Commit

Run the [`prepare-commit.sh`](utils/prepare-commit.sh) to clean the
notebooks and export scripts for pipeline notebooks, generate
documentation, run tests, render README if needed:

    bash utils/prepare-commit.sh
