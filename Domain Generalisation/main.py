# %% [markdown]
# # 📋 Task 2: Domain Generalization via Invariant & Robust Learning
# 
# In this task, we explore Domain Generalization (DG), where a model is trained on multiple source domains and must generalize to a completely unseen target domain. We will implement and compare four methods: ERM, IRM, GroupDRO, and SAM.
# 
# Our setup will use the **PACS dataset**. We will train on the **Art, Cartoon, and Photo** domains, holding out the **Sketch** domain as our unseen test environment, as suggested in the assignment manual.

# %% [markdown]
# ---

# %% [markdown]
# ## **Part 1: Empirical Risk Minimization (ERM) Baseline**

# %% [markdown]
# ### **1.1. Overview**
# 
# We begin by establishing a baseline using standard **Empirical Risk Minimization (ERM)**. This approach involves merging all data from the source domains into a single dataset and training a standard classifier on it. This model's performance on the unseen target domain will serve as the benchmark against which we will compare more advanced DG techniques.

# %% [markdown]
# ### **1.2. Environment Setup**
# 
# First, we need to set up the Python environment to ensure the notebook can find and import the DomainBed library from our `code/` directory.

# %%
import json
import os
import sys

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# The path to the 'domainbed' repository inside the 'code' folder
# This is the parent directory of the actual 'domainbed' package
module_path = os.path.abspath(os.path.join(".", "code", "domainbed"))

if module_path not in sys.path:
    sys.path.append(module_path)
    print(f"✅ Added '{module_path}' to Python path.")

# Import the main training function from DomainBed
try:
    from domainbed.scripts import train

    print("✅ Successfully imported DomainBed.")
except ImportError as e:
    print(
        "❌ Error importing DomainBed. Check that the path is correct and the repository is at './code/domainbed'."
    )
    print(e)

# Set plotting style for later
sns.set_theme(style="whitegrid")

# %% [markdown]
# ### **1.3. Experiment Runner Function**
# 
# To keep our code clean, we'll define a helper function that can launch any DomainBed experiment by taking a dictionary of arguments. This function mimics passing arguments via the command line.

# %%
def run_experiment(args_dict):
    """
    Builds a command-line command from a dictionary of arguments
    and executes the DomainBed training script, setting the PYTHONPATH.
    """
    # Define the path to the directory containing the 'domainbed' package
    module_path = os.path.abspath(os.path.join('.', 'code', 'domainbed'))
    
    # FIX 1: Enclose the module_path in quotes to handle spaces in the directory name.
    command = f'PYTHONPATH="{module_path}" python -m domainbed.scripts.train'
    
    # Append each argument from the dictionary to the command string
    for key, value in args_dict.items():
        if isinstance(value, bool) and value:
            command += f" --{key}"
        elif not (isinstance(value, bool) and not value):
            command += f" --{key} {value}"
            
    print("🚀 Executing Command:")
    print(command)
    
    # Execute the command in the shell
    os.system(command)
    
    # FIX 2: Corrected the typo 'N'/'A' to the string 'N/A'.
    print(f"\n🎉 Training finished for {args_dict.get('algorithm', 'N/A')}.")

# %% [markdown]
# ### **1.4. Run ERM Training**
# 
# Now, we define the specific parameters for our ERM baseline experiment and launch the training.

# %%
# --- ERM Experiment Configuration ---

# 1. Define hyperparameters in their own dictionary.
hparams = {
    'progress_bar': True
    # You can add other hyperparameters like batch_size or lr here too.
    # 'batch_size': 32,
    # 'lr': 5e-5
}

# 2. Configure the main experiment arguments.
erm_args = {
    'data_dir': './data/',
    'dataset': 'PACS',
    'algorithm': 'ERM',
    'test_env': 3,  # The index for the 'Sketch' domain in PACS
    'output_dir': './results/erm',
    'hparams_seed': 0,
    'trial_seed': 0,
    'seed': 0,
    # 3. Add the hparams dictionary as a JSON string.
    # The command line expects a string, so we use json.dumps().
    # We wrap it in single quotes for the shell.
    'hparams': f"'{json.dumps(hparams)}'"
}

# Launch the experiment
run_experiment(erm_args)


