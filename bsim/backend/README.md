# bsim backend

## Getting started

Create the conda environment

```bash
conda env create -f environment.yml
conda activate bsim-backend
pip install -r requirements.txt
```

Start the backend:

```bash
conda activate bsim-backend
python main.py
```

Update the conda environemnt ([source](https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#updating-an-environment)):

```bash
conda env update --file environment.yml --prune
```

Write the latest conda config to file:

```bash
conda env export --no-builds --from-history | grep -v "^prefix: " > environment.yml
```

