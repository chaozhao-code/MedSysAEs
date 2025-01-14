# AutoEncoders for Medication Recommendation Tasks

[//]: # ([![DOI]&#40;https://zenodo.org/badge/DOI/10.1145/3267471.3267476.svg&#41;]&#40;https://doi.org/&#41;)



## Dependencies

- gensim==4.3.2
- matplotlib==3.8.4
- numpy==2.2.1
- pandas==2.2.3
- pytest==7.4.4
- python_Levenshtein==0.26.1
- python_Levenshtein==0.26.1
- scikit_learn==1.4.2
- scipy==1.15.1
- seaborn==0.13.2
- setuptools==69.5.1
- spacy==3.8.3
- torch==2.6.0.dev20241008
- transformers==4.48.0




## Running

Please download the dataset using the link provided in the email and place it in the data folder.

The [`mimic.py`](eval/mimic.py) file is an executable to run an evaluation of the specified models on the MIMIC-III or MIMIC-IV dataset.

###### Argument supported by mimic.py 
    '-o' ('--outfile') - Log file to store the results. ( default='../../test-run_{}.log'.format(datetime.now().strftime("%Y-%m-%d-%H:%M")))
    '-m' ('--min-count) - Minimum number of occurrences for an item (ICD code) for it to be included in the experiments (default = 50).
    '-dr' ('--drop') - Proportion of items (ICD codes) to be randomly dropped from a user (patients ICU admission) record during model evaluation (default=0.5).
    '-nf' ('--n_folds') - Number of folds used for cross-fold validation (default=5).
    '-mi' ('--model_idx') - Index of model defined in list `MODELS_WITH_HYPERPARAMS` in `mimic.py` to use to run experiments on (-1 runs all models) (default=-1).
    '-fi' ('--fold_index') - Run a specific fold of a cross-fold validation run (-1 runs all folds) (default=-1). If running specific fold, assumes hyperparameter tuning was already performed.
    '-t' ('-tuning') - Perform hyperparameter tuning.
    '-f' ('-four') - Run the experiments on the MIMIC-IV dataset.

### Example run commands
`$> python mimic.py -mi 0`: Runs the count-based model on the MIMIC-III dataset.

`$> python mimic.py -mi 0 -f`: Runs the count-based model on the MIMIC-IV dataset.

the values of `-mi` are as follows:

- 0: Count-based model
- 1: SVD model
- 2: Vanilla Autoencoder model
- 3: Vanilla Autoencoder model with condition data
- 4: Denoising Autoencoder model
- 5: Denoising Autoencoder model with condition data
- 6: Variational Autoencoder model
- 7: Variational Autoencoder model with condition data
- 8: Adversarial Autoencoder model
- 9: Adversarial Autoencoder model with condition data

Condition data refers to additional inputs provided to the model alongside the patient-medication interaction matrix. In this context, the condition data includes vital information, demographic details, and other patient-related data.

## Illustration about the code

`mimic.py` and `helper.py` are the main files of our internship.

In `mimic.py`, we have only one main function, and the code is structured as follows:

- Step 1: Specify the models to be evaluated and their hyperparameters.
- Step 2: Load the dataset.
  -Related parameters: `if_four` (if True, load MIMIC-IV; otherwise, load MIMIC-III).
- Step 3: Analyze the dataset and obtain statistical information about it.
- Step 4: Split the dataset into training, validation, and test sets.
  - Related parameters: `min_count` (if the occurrence of a medication is lower than this number, exclude it), `drop` (the proportion of medications to be dropped for testing), `n_folds` (the number of folds required for cross-validation experiments).
- Step 5: Run cross-validation experiments on the selected models.
  - Related parameters: `tuning` (if True, it first performs hyperparameter tuning, and obtains the best parameters for the experiments, if False, uses the default parameters).

For the implementation of each step, please refer to helper.py.

## Results

If you run `mimic.py` in the terminal, the results will be displayed in the terminal window.

If you run `run.sh`, the results will be saved in the results folder.

`run.sh` includes experiments for both MIMIC-III and MIMIC-IV. You can comment out unnecessary code based on your needs or rewrite it to fit your own requirements.

## References and cite

Please see our papers for additional information on the models implemented and the experiments conducted:

- [Autoencoder-based prediction of ICU clinical codes](in press)
 


If you use our code in your own work please cite one of these papers:

    @inproceedings{Yordanov:2023,
        author    = {Tsvetan R. Yordanov and
                     Ameen Abu-Hanna and
                     Anita CJ Ravelli and
                     Iacopo Vagliano
                     },
        title     = {Autoencoder-based prediction of ICU clinical codes},
        booktitle = {Artificial Intelligence in Medicine},
        year = {in press},
        location = {Portoroz, Slovenia},
        keywords = {prediction, medical codes, recommender systems, autoencoders},
    }
    
  