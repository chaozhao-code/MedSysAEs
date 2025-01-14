"""
Executable to run AAE on the MIMIC Dataset

Run via:

`python3 eval/mimic.py -m <min_count> -o logfile.txt`

"""

import argparse
from irgan.utils import load
import pandas as pd
from helper import *
# from CONSTANTS import *


## Constant Variables

# METRICS = ['mrr', 'mrr@5', 'map', 'map@5', 'maf1', 'maf1@5'] # evaluation metrics





# @param fold_index - run a specific fold of CV (-1 = run all folds)
# see lines at parser.add_argument for param details
def main(min_count=50, drop=0.5, n_folds=5, tuning=False, if_four=False, model_idx=-1, outfile='out.log', fold_index=-1):



    print("Step 1: Selecting Models To Run")
    print("======================================================")
    sets_to_try = MODELS_WITH_HYPERPARAMS if model_idx < 0 else [MODELS_WITH_HYPERPARAMS[model_idx]]
    print("Models For Recommendation: ", sets_to_try)
    print('Hyper-parameter Values: drop = {}; min_count = {}, n_folds = {}, model_idx = {}'.format(drop, min_count, n_folds, model_idx))
    print("====================End of Step 1====================\n\n")


    print("Step 2: Loading Data")
    print("======================================================")
    try:
        if if_four:
            print("Loading data from", "data/patients_full_mimic4.json")
            patients = load("data/patients_full_mimic4.json")
        else:
            print("Loading data from", "data/patients_full.json")
            patients = load("data/patients_full.json")

        print("Data loaded successfully.")
    except Exception as e:
        print(f"Error loading data: {e}")
        return
    
    print("Unpacking MIMIC data...")


    try:
        bags_of_patients, ids, side_info = unpack_patients(patients, if_four)
        ## bags_of_patients is the prescribed medications
        ## ids is the id of the patients
        ## side_info is the other attributes of the patients
        assert len(set(ids)) == len(ids)
        del patients

        bags_of_patients_no_duplicates = []
        for bags_of_patient in bags_of_patients:
            bags_of_patients_no_duplicates.append(list(set(bags_of_patient)))

        bags = Bags(bags_of_patients_no_duplicates, ids, side_info)
        print("MIMIC data unpacked successfully.")
    except Exception as e:
        print(f"Error unpacking patients: {e}")
        return

    print("====================End of Step 2====================\n\n")

    print("Step 3: Statistic Overview of Data")
    print("======================================================")

    log("Whole dataset:", logfile=outfile)
    log(bags, logfile=outfile)
    

    all_codes = [code for c_list in bags_of_patients for code in c_list]


    if not all_codes:
        print("No NDC codes found.")
        return
    
    t_codes = pd.Series(all_codes).value_counts()
    n_codes_uniq = len(t_codes)
    n_codes_all = len(all_codes)
    code_counts = pd.Series(all_codes).value_counts()

    log("Total number of codes in current dataset = {}".format(n_codes_all), logfile=outfile)
    log("Total number of unique codes in current dataset = {}".format(n_codes_uniq), logfile=outfile)



    code_percentages = list(zip(code_counts, code_counts.index))
    code_percentages = [(val / n_codes_all, code) for val, code in code_percentages]
    code_percentages_accum = code_percentages
    for i in range(len(code_percentages)):
        if i > 0:
            code_percentages_accum[i] = (code_percentages_accum[i][0] + code_percentages_accum[i-1][0], code_percentages_accum[i][1])
        else:
            code_percentages_accum[i] = (code_percentages_accum[i][0], code_percentages_accum[i][1])

    for i in range(len(code_percentages_accum)):
        c_code = code_percentages_accum[i][1]
        c_percentage = code_percentages_accum[i][0]

        c_def = None
        log("{}\t#{}\tcode: {}\t( desc: {})".format(c_percentage, i+1, c_code, c_def), logfile=outfile)
        if c_percentage >= 0.5:
            log("first {} codes account for 50% of all code occurrences".format(i), logfile=outfile)
            log("Remaining {} codes account for remaining 50% of all code occurrences".format(n_codes_uniq-i), logfile=outfile)
            log("Last 1000 codes account for only {}% of data".format((1-code_percentages_accum[n_codes_uniq-1000][0])*100), logfile=outfile)
            break

    log("drop = {} min_count = {}".format(drop, min_count), logfile=outfile)
    print("====================End of Step 3====================\n\n")

    print("Step 4: Splitting Dataset")
    print("======================================================")
    if os.path.exists("splitsets100.pkl"):
        pass
    else:
        result = prepare_evaluation_kfold_cv(bags, min_count=min_count, drop=drop, n_folds=n_folds)
        if len(result) == 4:
            train_sets, val_sets, test_sets, y_tests = result
        elif len(result) == 3:
            train_sets, val_sets, test_sets = result
            y_tests = None
        else:
            raise ValueError("Unexpected number of return values from evaluation_kfold_cv")

        save_object((train_sets, val_sets, test_sets, y_tests), "splitsets100.pkl")
    del bags
    print("====================End of Step 4====================\n\n")







    print("Step 5: Running CV Pipeline for Models")
    for model_name, hyperparams_to_try in sets_to_try:
        print(model_name, hyperparams_to_try)
        print(f"Running CV pipeline for model: {model_name}")

        if tuning:
            print("Tuning The Parameters.")
            best_params = get_best_params(model_name, outfile, hyperparams_to_try, if_four, split_sets_filename="splitsets100.pkl", drop=0.5)
        else:
            best_params = BEST_PARAMS[model_name]

        try:
            metrics_df = run_cv_pipeline(drop, n_folds, outfile, model_name, best_params, if_four, split_sets_filename="splitsets100.pkl", fold_index=fold_index)
            metrics_df['metric_val'] = metrics_df['metric_val'].round(4)
            metrics_df['metric_std'] = metrics_df['metric_std'].round(4)
            print(f"Pipeline run completed for model: {model_name}")
            metrics_df.to_csv('./{}_{}_{}.csv'.format(outfile, model_name, fold_index), sep='\t')
            print(f"Metrics saved for model: {model_name}")
        except Exception as e:
            print(f"Error running CV pipeline for model {model_name}: {e}")
    os.remove("splitsets100.pkl")

# Command-Line interface, for running different configurations
if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-o', '--outfile',
                        help="File to store the results.",
                        default='testLog/test-run_{}.log'.format(datetime.now().strftime("%Y-%m-%d-%H:%M")))
    parser.add_argument('-m', '--min-count', type=int,
                        default=50,
                        help="Minimum count of items")
    parser.add_argument('-dr', '--drop', type=float,
                        help='Drop parameter', default=0.5)
    parser.add_argument('-nf', '--n_folds', type=int,
                        help='Number of folds', default=5)
    parser.add_argument('-mi', '--model_idx', type=int, help='Index of model to use',
                        default=-1)
    parser.add_argument('-fi', '--fold_index', type=int, help='cv-fold to run',
                        default=-1)
    parser.add_argument('-t', '--tuning', action='store_true', help='Run tuning to get the best parameters')
    parser.add_argument('-f', '--four', action='store_true', help='Run the models on mimic4 dataset, else mimic3')
    args = parser.parse_args()
    print(args)


    MODELS_WITH_HYPERPARAMS = [
        # *** BASELINES
        ("Count Based Recommender",
         {"order": [1, 2, 3, 4, 5]}),
        ("SVD Recommender",
         {"dims": [50, 100, 200, 500, 1000]}),

        # *** AEs
        ("Vanilla AE",
         {'lr': [0.0001 ,0.001, 0.01],  # [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05],
          'n_code': [100, 200],
          'n_epochs': [20],
          'batch_size': [50, 100],
          'n_hidden': [200, 500],
          'normalize_inputs': [True]}),

        ( "Vanilla AE with Condition",
         {'lr': [0.0001 ,0.001, 0.01],  # [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05],
          'n_code': [100, 200],
          'n_epochs': [20],
          'batch_size': [50, 100],
          'n_hidden': [200, 500],
          'normalize_inputs': [True]}),

        # *** DAEs
        ("Denosing AE",
         {'lr': [0.0001 ,0.001, 0.01],  # [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05],
          'n_code': [100, 200],
          'n_epochs': [20],
          'batch_size': [50, 100],
          'n_hidden': [200, 500],
          'normalize_inputs': [True]}),
        ("Denosing AE with Condition",
         {'lr': [0.0001 ,0.001, 0.01],  # [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05],
          'n_code': [100, 200],
          'n_epochs': [20],
          'batch_size': [50, 100],
          'n_hidden': [200, 500],
          'normalize_inputs': [True]}),

        # *** VAEs
        ("Variational AE",
         {'lr': [0.0001 ,0.001, 0.01],  # [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05],
          'n_code': [100, 200],
          'n_epochs': [20],
          'batch_size': [50, 100],
          'n_hidden': [200, 500],
          'normalize_inputs': [True]
          }),
        ("Variational AE with Condition",
         {'lr': [0.0001 ,0.001, 0.01],  # [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05],
          'n_code': [100, 200],
          'n_epochs': [20],
          'batch_size': [50, 100],
          'n_hidden': [200, 500],
          'normalize_inputs': [True]
          }),

        # *** AAEs
        ("Adverearial AE",
         {'prior': ['gauss'],
          'gen_lr': [0.0001 ,0.001, 0.01],
          'reg_lr': [0.0001 ,0.001, 0.01],
          'n_code': [100, 200],
          'n_epochs': [20],
          'batch_size': [50, 100],
          'n_hidden': [200, 500],
          'normalize_inputs': [True]},),
        ("Adverearial AE with Condition",
        {'prior': ['gauss'],
         'gen_lr': [0.0001 ,0.001, 0.01],
         'reg_lr': [0.0001 ,0.001, 0.01],
         'n_code': [100, 200],
         'n_epochs': [20],
         'batch_size': [50, 100],
         'n_hidden': [200, 500],
         'normalize_inputs': [True]},),
    ]

    main(outfile=args.outfile, min_count=args.min_count, drop=args.drop, n_folds=args.n_folds, tuning=args.tuning, if_four=args.four, model_idx=args.model_idx, fold_index=args.fold_index)
