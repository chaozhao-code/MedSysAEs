import re
import pickle
import os.path
import numpy as np
import scipy.sparse as sp
from sklearn import preprocessing
from aaerec.transforms import lists2sparse
from aaerec.datasets import Bags, corrupt_sets
from aaerec.evaluation import remove_non_missing, evaluate
from datetime import datetime
from aaerec.datasets import Bags, corrupt_sets
from aaerec.condition import ConditionList, PretrainedWordEmbeddingCondition, CategoricalCondition, Condition, ContinuousCondition
from aaerec.baselines import Countbased
from aaerec.svd import SVDRecommender
from aaerec.aae import AAERecommender
from aaerec.vae import VAERecommender
from aaerec.dae import DAERecommender
from matplotlib import pyplot as plt
import itertools as it
import copy
import gc
import sys
from datetime import datetime
import pandas as pd

METRICS = ['mrr', 'mrr@50', 'map', 'map@50', 'maf1', 'maf1@50', 'recall@100', 'p@50', 'p@100']
NUMERIC_FEATURES = ['age', 'drg_severity', 'drg_mortality']

# best hyperparameters for each model, as determined by grid search on the mimic-iii dataset

cb_params = {
    'order': 1.0,
}

svd_params = {
    'dims': 50.0,
}

ae_params = {
    'lr': 0.001,
    'n_code': 200,
    'n_epochs': 20,
    'batch_size': 50,
    'n_hidden': 500,
    'normalize_inputs': True,
}

vae_params = {
    'lr': 0.001,
    'n_code': 200,
    'n_epochs': 20,
    'batch_size': 50,
    'n_hidden': 500,
    'normalize_inputs': True,
}

dae_params = {
    'lr': 0.001,
    'n_code': 100,
    'n_epochs': 20,
    'batch_size': 50,
    'n_hidden': 200,
    'normalize_inputs': True,
}

aae_params = {
    'prior': 'gauss',
    'gen_lr': 0.001,
    'reg_lr': 0.01,
    'n_code': 200,
    'n_epochs': 20,
    'batch_size': 50,
    'n_hidden': 500,
    'normalize_inputs': True,
}

BEST_PARAMS = {
    "Count Based Recommender": cb_params,
    "SVD Recommender": svd_params,
    "Vanilla AE": ae_params,
    "Vanilla AE with Condition": ae_params,
    "Denosing AE": dae_params,
    "Denosing AE with Condition": dae_params,
    "Variational AE": vae_params,
    "Variational AE with Condition": vae_params,
    "Adverearial AE": aae_params,
    "Adverearial AE with Condition": aae_params,
}


# Metadata to use
# optional conditions (ICD9 codes not optional)
# commented out conditions can be used if compute resources permitting
CONDITIONS_MIMIC4 = ConditionList([
# ('length_of_stay_icu', ContinuousCondition(sparse=True)),
('d_icd9', CategoricalCondition(embedding_dim=100, sparse=True, vocab_size=7588, embedding_on_gpu=True, reduce='mean')),
('d_icd10', CategoricalCondition(embedding_dim=100, sparse=True, vocab_size=13691, embedding_on_gpu=True, reduce='mean')),
('p_icd9', CategoricalCondition(embedding_dim=100, sparse=True, vocab_size=2784, embedding_on_gpu=True, reduce='mean')),
('p_icd10', CategoricalCondition(embedding_dim=100, sparse=True, vocab_size=8290, embedding_on_gpu=True, reduce='mean')),
# ('gender', CategoricalCondition(embedding_dim=3, sparse=True, embedding_on_gpu=True)),
# ('age', ContinuousCondition(sparse=True)),
# ('admission_type', CategoricalCondition(embedding_dim=5, sparse=True, embedding_on_gpu=True)),
# ('race', CategoricalCondition(embedding_dim=7, sparse=True, embedding_on_gpu=True)),
# ('drg_type', CategoricalCondition(embedding_dim=3, sparse=True, embedding_on_gpu=True)),
# ('drg_code', CategoricalCondition(embedding_dim=100, sparse=True, embedding_on_gpu=True)),
# ('drg_severity', ContinuousCondition(sparse=True)),
# ('drg_mortality', ContinuousCondition(sparse=True)),
# ('HR', ContinuousCondition(sparse=True, reduce='mean')),
# ('Admission Weight (Kg)', ContinuousCondition(sparse=True, reduce='mean')),
# ('RR', ContinuousCondition(sparse=True, reduce='mean')),
# ('SpO2', ContinuousCondition(sparse=True, reduce='mean')),
# ('Eye Opening', ContinuousCondition(sparse=True, reduce='mean')),
# ('Verbal Response', ContinuousCondition(sparse=True, reduce='mean')),
# ('Motor Response', ContinuousCondition(sparse=True, reduce='mean')),
# ('Alarms On', ContinuousCondition(sparse=True, reduce='mean')),
# ('HR Alarm - Low', ContinuousCondition(sparse=True, reduce='mean')),
# ('HR Alarm - High', ContinuousCondition(sparse=True, reduce='mean')),
# ('HOB', CategoricalCondition(embedding_dim=2, sparse=True, embedding_on_gpu=True, reduce='mean')),
# ('SpO2 Alarm - Low', ContinuousCondition(sparse=True, reduce='mean')),
# ('SpO2 Alarm - High', ContinuousCondition(sparse=True, reduce='mean')),
# ('RUL Lung Sounds', CategoricalCondition(embedding_dim=2, sparse=True, embedding_on_gpu=True, reduce='mean')),
# ('Turn', CategoricalCondition(embedding_dim=2, sparse=True, embedding_on_gpu=True, reduce='mean')),
# ('LUL Lung Sounds', CategoricalCondition(embedding_dim=2, sparse=True, embedding_on_gpu=True, reduce='mean')),
# ('Resp Alarm - High', ContinuousCondition(sparse=True, reduce='mean')),
# ('Resp Alarm - Low', ContinuousCondition(sparse=True, reduce='mean')),
# ('Heart Rhythm', CategoricalCondition(embedding_dim=10, sparse=True, embedding_on_gpu=True, reduce='mean')),
# ('RLL Lung Sounds', CategoricalCondition(embedding_dim=2, sparse=True, embedding_on_gpu=True, reduce='mean')),
# ('LLL Lung Sounds', CategoricalCondition(embedding_dim=2, sparse=True, embedding_on_gpu=True, reduce='mean')),
# ('Abdominal Assessment', CategoricalCondition(embedding_dim=2, sparse=True, embedding_on_gpu=True, reduce='mean')),
# ('Temp Site', CategoricalCondition(embedding_dim=2, sparse=True, embedding_on_gpu=True, reduce='mean')),
# ('Parameters Checked', ContinuousCondition(sparse=True, reduce='mean')),
# ('Skin Integrity', CategoricalCondition(embedding_dim=2, sparse=True, embedding_on_gpu=True, reduce='mean')),
# ('Skin Temp', CategoricalCondition(embedding_dim=2, sparse=True, embedding_on_gpu=True, reduce='mean')),
# ('Skin Condition', CategoricalCondition(embedding_dim=2, sparse=True, embedding_on_gpu=True, reduce='mean')),
# ('Oral Cavity', CategoricalCondition(embedding_dim=2, sparse=True, embedding_on_gpu=True, reduce='mean')),
# ('Skin Color', CategoricalCondition(embedding_dim=2, sparse=True, embedding_on_gpu=True, reduce='mean')),
# ('Braden Mobility', ContinuousCondition(sparse=True, reduce='mean')),
# ('Braden Sensory Perception', ContinuousCondition(sparse=True, reduce='mean')),
# ('Braden Moisture', ContinuousCondition(sparse=True, reduce='mean')),
# ('Braden Activity', ContinuousCondition(sparse=True, reduce='mean')),
# ('Braden Nutrition', ContinuousCondition(sparse=True, reduce='mean')),
# ('Bowel Sounds', CategoricalCondition(embedding_dim=2, sparse=True, embedding_on_gpu=True, reduce='mean')),
# ('Activity Tolerance', CategoricalCondition(embedding_dim=2, sparse=True, embedding_on_gpu=True, reduce='mean')),
# ('Braden Friction/Shear', ContinuousCondition(sparse=True, reduce='mean')),
# ('SpO2 Desat Limit', ContinuousCondition(sparse=True, reduce='mean')),
# ('Urine Source', CategoricalCondition(embedding_dim=2, sparse=True, embedding_on_gpu=True, reduce='mean')),
# ('NBPs', ContinuousCondition(sparse=True, reduce='mean')),
# ('NBPd', ContinuousCondition(sparse=True, reduce='mean')),
# ('NBPm', ContinuousCondition(sparse=True, reduce='mean')),
# ('Diet Type', CategoricalCondition(embedding_dim=10, sparse=True, embedding_on_gpu=True, reduce='mean')),
# ('O2 Delivery Device(s)', CategoricalCondition(embedding_dim=2, sparse=True, embedding_on_gpu=True, reduce='mean')),
# ('Edema Location', CategoricalCondition(embedding_dim=2, sparse=True, embedding_on_gpu=True, reduce='mean')),
# ('Temperature F', ContinuousCondition(sparse=True, reduce='mean')),
# ('Dorsal PedPulse R', CategoricalCondition(embedding_dim=2, sparse=True, embedding_on_gpu=True, reduce='mean')),
# ('Dorsal PedPulse L', CategoricalCondition(embedding_dim=2, sparse=True, embedding_on_gpu=True, reduce='mean')),
# ('Position', CategoricalCondition(embedding_dim=2, sparse=True, embedding_on_gpu=True, reduce='mean')),
# ('Pain Present', CategoricalCondition(embedding_dim=2, sparse=True, embedding_on_gpu=True, reduce='mean')),
# ('Gait/Transferring', ContinuousCondition(sparse=True, reduce='mean')),
# ('IV/Saline lock', ContinuousCondition(sparse=True, reduce='mean')),
# ('Ambulatory aid', ContinuousCondition(sparse=True, reduce='mean')),
# ('Secondary diagnosis', ContinuousCondition(sparse=True, reduce='mean')),
# ('Mental status', ContinuousCondition(sparse=True, reduce='mean')),
# ('History of falling (within 3 mnths)*', ContinuousCondition(sparse=True, reduce='mean')),
# ('Ectopy Type 1', CategoricalCondition(embedding_dim=2, sparse=True, embedding_on_gpu=True, reduce='mean')),
# ('Sodium (serum)', ContinuousCondition(sparse=True, reduce='mean')),
# ('Potassium (serum)', ContinuousCondition(sparse=True, reduce='mean')),
# ('Chloride (serum)', ContinuousCondition(sparse=True, reduce='mean')),
# ('Creatinine (serum)', ContinuousCondition(sparse=True, reduce='mean')),
# ('BUN', ContinuousCondition(sparse=True, reduce='mean')),
# ('HCO3 (serum)', ContinuousCondition(sparse=True, reduce='mean')),
# ('Anion gap', ContinuousCondition(sparse=True, reduce='mean')),
# ('Glucose (serum)', ContinuousCondition(sparse=True, reduce='mean')),
# ('Hematocrit (serum)', ContinuousCondition(sparse=True, reduce='mean')),
# ('Hemoglobin', ContinuousCondition(sparse=True, reduce='mean')),
# ('Platelet Count', ContinuousCondition(sparse=True, reduce='mean')),
# ('WBC', ContinuousCondition(sparse=True, reduce='mean')),
# ('Cough Effort', CategoricalCondition(embedding_dim=2, sparse=True, embedding_on_gpu=True, reduce='mean')),
# ('Magnesium', ContinuousCondition(sparse=True, reduce='mean')),
# ('NBP Alarm - Low', ContinuousCondition(sparse=True, reduce='mean')),
# ('NBP Alarm - High', ContinuousCondition(sparse=True, reduce='mean')),
# ('NBP Alarm Source', CategoricalCondition(embedding_dim=2, sparse=True, embedding_on_gpu=True, reduce='mean')),
# ('Urine Color', CategoricalCondition(embedding_dim=2, sparse=True, embedding_on_gpu=True, reduce='mean')),
# ('Side Rails', CategoricalCondition(embedding_dim=2, sparse=True, embedding_on_gpu=True, reduce='mean')),
# ('Safety Measures', CategoricalCondition(embedding_dim=2, sparse=True, embedding_on_gpu=True, reduce='mean')),
# ('Pain Location', CategoricalCondition(embedding_dim=10, sparse=True, embedding_on_gpu=True, reduce='mean')),
# ('Education Learner', CategoricalCondition(embedding_dim=2, sparse=True, embedding_on_gpu=True, reduce='mean')),
# ('Education Topic', CategoricalCondition(embedding_dim=10, sparse=True, embedding_on_gpu=True, reduce='mean')),
# ('Education Method', CategoricalCondition(embedding_dim=2, sparse=True, embedding_on_gpu=True, reduce='mean')),
# ('Education Barrier', CategoricalCondition(embedding_dim=2, sparse=True, embedding_on_gpu=True, reduce='mean')),
# ('Phosphorous', ContinuousCondition(sparse=True, reduce='mean')),
# ('Education Response', CategoricalCondition(embedding_dim=2, sparse=True, embedding_on_gpu=True, reduce='mean')),
# ('Calcium non-ionized', ContinuousCondition(sparse=True, reduce='mean')),
# ('Urine Appearance', CategoricalCondition(embedding_dim=2, sparse=True, embedding_on_gpu=True, reduce='mean')),
# ('PostTib Pulses R', CategoricalCondition(embedding_dim=2, sparse=True, embedding_on_gpu=True, reduce='mean')),
# ('PostTib Pulses L', CategoricalCondition(embedding_dim=2, sparse=True, embedding_on_gpu=True, reduce='mean')),
# ('Problem List', CategoricalCondition(embedding_dim=10, sparse=True, embedding_on_gpu=True, reduce='mean')),
# ('Pupil Size Right', CategoricalCondition(embedding_dim=2, sparse=True, embedding_on_gpu=True, reduce='mean')),
# ('Pupil Size Left', CategoricalCondition(embedding_dim=2, sparse=True, embedding_on_gpu=True, reduce='mean')),
# ('Pupil Response R', CategoricalCondition(embedding_dim=2, sparse=True, embedding_on_gpu=True, reduce='mean')),
# ('Pupil Response L', CategoricalCondition(embedding_dim=2, sparse=True, embedding_on_gpu=True, reduce='mean')),
# ('RLE Temp', CategoricalCondition(embedding_dim=2, sparse=True, embedding_on_gpu=True, reduce='mean')),
# ('LLE Temp', CategoricalCondition(embedding_dim=2, sparse=True, embedding_on_gpu=True, reduce='mean')),
# ('RLE Color', CategoricalCondition(embedding_dim=2, sparse=True, embedding_on_gpu=True, reduce='mean')),
# ('RUE Temp', CategoricalCondition(embedding_dim=2, sparse=True, embedding_on_gpu=True, reduce='mean')),
# ('LLE Color', CategoricalCondition(embedding_dim=2, sparse=True, embedding_on_gpu=True, reduce='mean')),
# ('LUE Temp', CategoricalCondition(embedding_dim=2, sparse=True, embedding_on_gpu=True, reduce='mean')),
# ('RUE Color', CategoricalCondition(embedding_dim=2, sparse=True, embedding_on_gpu=True, reduce='mean')),
# ('LUE Color', CategoricalCondition(embedding_dim=2, sparse=True, embedding_on_gpu=True, reduce='mean')),
# ('Support Systems', CategoricalCondition(embedding_dim=2, sparse=True, embedding_on_gpu=True, reduce='mean')),
# ('Pain Level', CategoricalCondition(embedding_dim=2, sparse=True, embedding_on_gpu=True, reduce='mean')),
# ('Pain Level Acceptable', CategoricalCondition(embedding_dim=2, sparse=True, embedding_on_gpu=True, reduce='mean')),
# ('Richmond-RAS Scale', ContinuousCondition(sparse=True, reduce='mean')),
# ('Anti Embolic Device', CategoricalCondition(embedding_dim=2, sparse=True, embedding_on_gpu=True, reduce='mean')),
# ('Nares R', CategoricalCondition(embedding_dim=2, sparse=True, embedding_on_gpu=True, reduce='mean')),
# ('Pain Management', CategoricalCondition(embedding_dim=2, sparse=True, embedding_on_gpu=True, reduce='mean')),
# ('Nares L', CategoricalCondition(embedding_dim=2, sparse=True, embedding_on_gpu=True, reduce='mean')),
# ('Capillary Refill R', CategoricalCondition(embedding_dim=2, sparse=True, embedding_on_gpu=True, reduce='mean')),

# ('HR', ContinuousCondition(sparse=True, reduce='mean')),
# ('Admission Weight (Kg)', ContinuousCondition(sparse=True, reduce='mean')),
# ('RR', ContinuousCondition(sparse=True, reduce='mean')),
# ('SpO2', ContinuousCondition(sparse=True, reduce='mean')),
# ('Eye Opening', ContinuousCondition(sparse=True, reduce='mean')),
# ('Verbal Response', ContinuousCondition(sparse=True, reduce='mean')),
# ('Motor Response', ContinuousCondition(sparse=True, reduce='mean')),
# ('Alarms On', ContinuousCondition(sparse=True, reduce='mean')),
# ('HR Alarm - Low', ContinuousCondition(sparse=True, reduce='mean')),
# ('HR Alarm - High', ContinuousCondition(sparse=True, reduce='mean')),
# ('HOB', CategoricalCondition(embedding_dim=2, sparse=True, embedding_on_gpu=True, reduce='mean')),
# ('SpO2 Alarm - Low', ContinuousCondition(sparse=True, reduce='mean')),
# ('SpO2 Alarm - High', ContinuousCondition(sparse=True, reduce='mean')),
# ('RUL Lung Sounds', CategoricalCondition(embedding_dim=2, sparse=True, embedding_on_gpu=True, reduce='mean')),
# ('Turn', CategoricalCondition(embedding_dim=2, sparse=True, embedding_on_gpu=True, reduce='mean')),
# ('LUL Lung Sounds', CategoricalCondition(embedding_dim=2, sparse=True, embedding_on_gpu=True, reduce='mean')),
# ('Resp Alarm - High', ContinuousCondition(sparse=True, reduce='mean')),
# ('Resp Alarm - Low', ContinuousCondition(sparse=True, reduce='mean')),
# ('Heart Rhythm', CategoricalCondition(embedding_dim=2, sparse=True, embedding_on_gpu=True, reduce='mean')),
# ('RLL Lung Sounds', CategoricalCondition(embedding_dim=2, sparse=True, embedding_on_gpu=True, reduce='mean')),
# ('LLL Lung Sounds', CategoricalCondition(embedding_dim=2, sparse=True, embedding_on_gpu=True, reduce='mean')),
#
#
# ('Glucose (serum)', ContinuousCondition(sparse=True, reduce='mean')),
# ('Pain Level', CategoricalCondition(embedding_dim=2, sparse=True, embedding_on_gpu=True, reduce='mean')),
# ('NBPs', ContinuousCondition(sparse=True, reduce='mean')),
# ('NBPd', ContinuousCondition(sparse=True, reduce='mean')),
# ('NBPm', ContinuousCondition(sparse=True, reduce='mean')),
])



CONDITIONS_MIMIC3 = ConditionList([
('gender', CategoricalCondition(embedding_dim=3, sparse=True, embedding_on_gpu=True)),
('ethnicity_grouped', CategoricalCondition(embedding_dim=7, sparse=True, embedding_on_gpu=True)),
('admission_type', CategoricalCondition(embedding_dim=5, sparse=True, embedding_on_gpu=True)),
('icd9_code_p_lst', CategoricalCondition(embedding_dim=100, sparse=True, vocab_size=2000, embedding_on_gpu=True, reduce='mean')),
('icd9_code_d_lst', CategoricalCondition(embedding_dim=100, sparse=True, vocab_size=6545, embedding_on_gpu=True, reduce='mean')),
('los_hospital', ContinuousCondition(sparse=True)),
('age', ContinuousCondition(sparse=True)),
('heartrate_min_lst_slope', ContinuousCondition(sparse=True)),
('heartrate_max_lst_slope', ContinuousCondition(sparse=True)),
('heartrate_mean_lst_slope', ContinuousCondition(sparse=True)),
('sysbp_min_lst_slope', ContinuousCondition(sparse=True)),
('sysbp_max_lst_slope', ContinuousCondition(sparse=True)),
('sysbp_mean_lst_slope', ContinuousCondition(sparse=True)),
('diasbp_min_lst_slope', ContinuousCondition(sparse=True)),
('diasbp_max_lst_slope', ContinuousCondition(sparse=True)),
('diasbp_mean_lst_slope', ContinuousCondition(sparse=True)),
('meanbp_min_lst_slope', ContinuousCondition(sparse=True)),
('meanbp_max_lst_slope', ContinuousCondition(sparse=True)),
('meanbp_mean_lst_slope', ContinuousCondition(sparse=True)),
('resprate_min_lst_slope', ContinuousCondition(sparse=True)),
('resprate_max_lst_slope', ContinuousCondition(sparse=True)),
('resprate_mean_lst_slope', ContinuousCondition(sparse=True)),
('tempc_min_lst_slope', ContinuousCondition(sparse=True)),
('tempc_max_lst_slope', ContinuousCondition(sparse=True)),
('tempc_mean_lst_slope', ContinuousCondition(sparse=True)),
('spo2_min_lst_slope', ContinuousCondition(sparse=True)),
('spo2_max_lst_slope', ContinuousCondition(sparse=True)),
('spo2_mean_lst_slope', ContinuousCondition(sparse=True)),
('glucose_min_lst_slope', ContinuousCondition(sparse=True)),
('glucose_max_lst_slope', ContinuousCondition(sparse=True)),
('glucose_mean_lst_slope', ContinuousCondition(sparse=True)),
('heartrate_min_lst_mean', ContinuousCondition(sparse=True)),
('heartrate_max_lst_mean', ContinuousCondition(sparse=True)),
('heartrate_mean_lst_mean', ContinuousCondition(sparse=True)),
('sysbp_min_lst_mean', ContinuousCondition(sparse=True)),
('sysbp_max_lst_mean', ContinuousCondition(sparse=True)),
('sysbp_mean_lst_mean', ContinuousCondition(sparse=True)),
('diasbp_min_lst_mean', ContinuousCondition(sparse=True)),
('diasbp_max_lst_mean', ContinuousCondition(sparse=True)),
('diasbp_mean_lst_mean', ContinuousCondition(sparse=True)),
('meanbp_min_lst_mean', ContinuousCondition(sparse=True)),
('meanbp_max_lst_mean', ContinuousCondition(sparse=True)),
('meanbp_mean_lst_mean', ContinuousCondition(sparse=True)),
('resprate_min_lst_mean', ContinuousCondition(sparse=True)),
('resprate_max_lst_mean', ContinuousCondition(sparse=True)),
('resprate_mean_lst_mean', ContinuousCondition(sparse=True)),
('heartrate_min_lst_sd', ContinuousCondition(sparse=True)),
('heartrate_max_lst_sd', ContinuousCondition(sparse=True)),
('heartrate_mean_lst_sd', ContinuousCondition(sparse=True)),
('sysbp_min_lst_sd', ContinuousCondition(sparse=True)),
('sysbp_max_lst_sd', ContinuousCondition(sparse=True)),
('sysbp_mean_lst_sd', ContinuousCondition(sparse=True)),
('diasbp_min_lst_sd', ContinuousCondition(sparse=True)),
('diasbp_max_lst_sd', ContinuousCondition(sparse=True)),
('diasbp_mean_lst_sd', ContinuousCondition(sparse=True)),
('meanbp_min_lst_sd', ContinuousCondition(sparse=True)),
('meanbp_max_lst_sd', ContinuousCondition(sparse=True)),
('meanbp_mean_lst_sd', ContinuousCondition(sparse=True)),
('resprate_min_lst_sd', ContinuousCondition(sparse=True)),
('resprate_max_lst_sd', ContinuousCondition(sparse=True)),
('resprate_mean_lst_sd', ContinuousCondition(sparse=True)),
('tempc_min_lst_sd', ContinuousCondition(sparse=True)),
('tempc_max_lst_sd', ContinuousCondition(sparse=True)),
('tempc_mean_lst_sd', ContinuousCondition(sparse=True)),
('spo2_min_lst_sd', ContinuousCondition(sparse=True)),
('spo2_max_lst_sd', ContinuousCondition(sparse=True)),
('spo2_mean_lst_sd', ContinuousCondition(sparse=True)),
('glucose_min_lst_sd', ContinuousCondition(sparse=True)),
('glucose_max_lst_sd', ContinuousCondition(sparse=True)),
('glucose_mean_lst_sd', ContinuousCondition(sparse=True)),
('heartrate_min_lst_delta', ContinuousCondition(sparse=True)),
('heartrate_max_lst_delta', ContinuousCondition(sparse=True)),
('heartrate_mean_lst_delta', ContinuousCondition(sparse=True)),
('sysbp_min_lst_delta', ContinuousCondition(sparse=True)),
('sysbp_max_lst_delta', ContinuousCondition(sparse=True)),
('sysbp_mean_lst_delta', ContinuousCondition(sparse=True)),
('diasbp_min_lst_delta', ContinuousCondition(sparse=True)),
('diasbp_max_lst_delta', ContinuousCondition(sparse=True)),
('diasbp_mean_lst_delta', ContinuousCondition(sparse=True)),
('meanbp_min_lst_delta', ContinuousCondition(sparse=True)),
('meanbp_max_lst_delta', ContinuousCondition(sparse=True)),
('meanbp_mean_lst_delta', ContinuousCondition(sparse=True)),
('resprate_min_lst_delta', ContinuousCondition(sparse=True)),
('resprate_max_lst_delta', ContinuousCondition(sparse=True)),
('resprate_mean_lst_delta', ContinuousCondition(sparse=True)),
('tempc_min_lst_delta', ContinuousCondition(sparse=True)),
('tempc_max_lst_delta', ContinuousCondition(sparse=True)),
('tempc_mean_lst_delta', ContinuousCondition(sparse=True)),
('spo2_min_lst_delta', ContinuousCondition(sparse=True)),
('spo2_max_lst_delta', ContinuousCondition(sparse=True)),
('spo2_mean_lst_delta', ContinuousCondition(sparse=True)),
('glucose_min_lst_delta', ContinuousCondition(sparse=True)),
('glucose_max_lst_delta', ContinuousCondition(sparse=True)),
('glucose_mean_lst_delta', ContinuousCondition(sparse=True)),
('heartrate_min_lst_min', ContinuousCondition(sparse=True)),
('heartrate_max_lst_min', ContinuousCondition(sparse=True)),
('heartrate_mean_lst_min', ContinuousCondition(sparse=True)),
('sysbp_min_lst_min', ContinuousCondition(sparse=True)),
('sysbp_max_lst_min', ContinuousCondition(sparse=True)),
('sysbp_mean_lst_min', ContinuousCondition(sparse=True)),
('diasbp_min_lst_min', ContinuousCondition(sparse=True)),
('diasbp_max_lst_min', ContinuousCondition(sparse=True)),
('diasbp_mean_lst_min', ContinuousCondition(sparse=True)),
('meanbp_min_lst_min', ContinuousCondition(sparse=True)),
('meanbp_max_lst_min', ContinuousCondition(sparse=True)),
('meanbp_mean_lst_min', ContinuousCondition(sparse=True)),
('resprate_min_lst_min', ContinuousCondition(sparse=True)),
('resprate_max_lst_min', ContinuousCondition(sparse=True)),
('resprate_mean_lst_min', ContinuousCondition(sparse=True)),
('tempc_min_lst_min', ContinuousCondition(sparse=True)),
('tempc_max_lst_min', ContinuousCondition(sparse=True)),
('tempc_mean_lst_min', ContinuousCondition(sparse=True)),
('spo2_min_lst_min', ContinuousCondition(sparse=True)),
('spo2_max_lst_min', ContinuousCondition(sparse=True)),
('spo2_mean_lst_min', ContinuousCondition(sparse=True)),
('glucose_min_lst_min', ContinuousCondition(sparse=True)),
('glucose_max_lst_min', ContinuousCondition(sparse=True)),
('glucose_mean_lst_min', ContinuousCondition(sparse=True)),
('heartrate_min_lst_max', ContinuousCondition(sparse=True)),
('heartrate_max_lst_max', ContinuousCondition(sparse=True)),
('heartrate_mean_lst_max', ContinuousCondition(sparse=True)),
('sysbp_min_lst_max', ContinuousCondition(sparse=True)),
('sysbp_max_lst_max', ContinuousCondition(sparse=True)),
('sysbp_mean_lst_max', ContinuousCondition(sparse=True)),
('diasbp_min_lst_max', ContinuousCondition(sparse=True)),
('diasbp_max_lst_max', ContinuousCondition(sparse=True)),
('diasbp_mean_lst_max', ContinuousCondition(sparse=True)),
('meanbp_min_lst_max', ContinuousCondition(sparse=True)),
('meanbp_max_lst_max', ContinuousCondition(sparse=True)),
('meanbp_mean_lst_max', ContinuousCondition(sparse=True)),
('resprate_min_lst_max', ContinuousCondition(sparse=True)),
('resprate_max_lst_max', ContinuousCondition(sparse=True)),
('resprate_mean_lst_max', ContinuousCondition(sparse=True)),
('tempc_min_lst_max', ContinuousCondition(sparse=True)),
('tempc_max_lst_max', ContinuousCondition(sparse=True)),
('tempc_mean_lst_max', ContinuousCondition(sparse=True)),
('spo2_min_lst_max', ContinuousCondition(sparse=True)),
('spo2_max_lst_max', ContinuousCondition(sparse=True)),
('spo2_mean_lst_max', ContinuousCondition(sparse=True)),
('glucose_min_lst_max', ContinuousCondition(sparse=True)),
('glucose_max_lst_max', ContinuousCondition(sparse=True)),
('glucose_mean_lst_max', ContinuousCondition(sparse=True)),
('heartrate_min_lst_mm', ContinuousCondition(sparse=True)),
('heartrate_max_lst_mm', ContinuousCondition(sparse=True)),
('heartrate_mean_lst_mm', ContinuousCondition(sparse=True)),
('sysbp_min_lst_mm', ContinuousCondition(sparse=True)),
('sysbp_max_lst_mm', ContinuousCondition(sparse=True)),
('sysbp_mean_lst_mm', ContinuousCondition(sparse=True)),
('diasbp_min_lst_mm', ContinuousCondition(sparse=True)),
('diasbp_max_lst_mm', ContinuousCondition(sparse=True)),
('diasbp_mean_lst_mm', ContinuousCondition(sparse=True)),
('meanbp_min_lst_mm', ContinuousCondition(sparse=True)),
('meanbp_max_lst_mm', ContinuousCondition(sparse=True)),
('meanbp_mean_lst_mm', ContinuousCondition(sparse=True)),
('resprate_min_lst_mm', ContinuousCondition(sparse=True)),
('resprate_max_lst_mm', ContinuousCondition(sparse=True)),
('resprate_mean_lst_mm', ContinuousCondition(sparse=True)),
('tempc_min_lst_mm', ContinuousCondition(sparse=True)),
('tempc_max_lst_mm', ContinuousCondition(sparse=True)),
('tempc_mean_lst_mm', ContinuousCondition(sparse=True)),
('spo2_min_lst_mm', ContinuousCondition(sparse=True)),
('spo2_max_lst_mm', ContinuousCondition(sparse=True)),
('spo2_mean_lst_mm', ContinuousCondition(sparse=True)),
('glucose_min_lst_mm', ContinuousCondition(sparse=True)),
('glucose_max_lst_mm', ContinuousCondition(sparse=True)),
('glucose_mean_lst_mm', ContinuousCondition(sparse=True)),
])





### A: Dataset Loader Helper Functions


def normalize_conditional_data_bags(bags):
    '''
    Normalize all conditional data (for example: nan to number) in the bags object.
    So this function normalize all the attributes in the bags object (Not the IDs or Medications).
    For example, before normalization:
    [8.584, 15.7438, 42.9292, 5.3021, 9.0806, 17.325,
    1.5993, 20.8229, 4.9979, 6.0521, 5.8542, 8.6868, 1.9694, 10.025, 6.3396]
    After normalization:
    [0.148815021568589, 0.27293964778326557, 0.7442346020412711, 0.09191893358094312,
    0.157424241013016, 0.30035184630426426, 0.027725986019879353, 0.3609925806873919,
    0.08664522324063967, 0.10492117800969915, 0.10149031911309805, 0.15059719587162382,
    0.03414216023732283, 0.1737966671977056, 0.10990537170738895]
    '''
    for k in list(bags.owner_attributes.keys()):
        if k in ['ICD9_defs_txt', 'gender', 'ethnicity_grouped', 'admission_type', 'icd9_code_d_lst', 'icd9_code_p_lst',
                 'ndc_list']:
            continue

        if k not in NUMERIC_FEATURES:
            continue

        c_vals = list(bags.owner_attributes[k].values())

        # Handle empty strings and non-numeric values
        c_vals = [0 if v == '' else v for v in c_vals]
        try:
            c_vals = np.nan_to_num(np.array(c_vals, dtype=np.float64))  # Ensure conversion to float
        except ValueError as e:
            print(f"Error converting attribute '{k}' values to float: {e}")
            print(f"Offending values: {c_vals}")
            continue  # Skip this attribute if conversion fails

        c_vals = preprocessing.normalize([c_vals])[0].tolist()
        c_keys = list(bags.owner_attributes[k].keys())
        bags.owner_attributes[k] = {c_keys[i]: c_vals[i] for i in range(len(c_keys))}
    return bags


def prepare_evaluation_kfold_cv(bags, n_folds=5, n_items=None, min_count=None, drop=1):
    ### This function still need to be re-written, Chao
    """
    Split data into train and dev set.
    Build vocab on train set and applies it to both train and test set.
    """

    split_results = bags.create_kfold_train_validate_test(n_folds=n_folds)

    if len(split_results) == 4:
        train_sets, val_sets, test_sets, y_tests = split_results
    elif len(split_results) == 3:
        train_sets, val_sets, test_sets = split_results
        y_tests = None
    else:
        raise ValueError("Unexpected number of return values from create_kfold_train_validate_test")
    # Split 10% validation data.

    for i in range(n_folds):
        train_sets[i] = normalize_conditional_data_bags(train_sets[i])
        test_sets[i] = normalize_conditional_data_bags(test_sets[i])
        val_sets[i] = normalize_conditional_data_bags(val_sets[i])

    missings = []
    y_tests = []

    # Builds vocabulary only on training set (I don't understand why we only build vocab on training set)
    for i in range(n_folds):
        train_set = train_sets[i]
        test_set = test_sets[i]
        val_set = val_sets[i]

        vocab, __counts = train_set.build_vocab(max_features=n_items, min_count=min_count, apply=False)

        # the vocab is a mapping from the original token (medications code) to the index
        # __counts is the frequency of each token in the training set

        # Apply vocab (turn track ids into indices)
        train_set = train_set.apply_vocab(vocab)
        # Discard unknown tokens in the test set
        test_set = test_set.apply_vocab(vocab)
        val_set = val_set.apply_vocab(vocab)

        # Corrupt sets (currently set to remove 50% of item list items)
        print("Drop parameter:", drop)
        noisy, missing = corrupt_sets(test_set.data, drop=drop)
        ## noisy here is what we want to used for the input of the AutoEncoder
        ## missing is the ground truth
        ## however I will change the missing to be the full text_set data

        # because corrupt_sets used set operations, so the noisy and missing do not contain duplicate
        # some entries might have too few items to drop, resulting in empty missing and a full noisy
        # remove those from the sets (should be just a few)
        entries_to_keep = np.where([len(missing[i]) != 0 for i in range(len(missing))])[0]
        missing = [missing[i] for i in entries_to_keep]
        noisy = [noisy[i] for i in entries_to_keep]
        test_set.data = [test_set.data[i] for i in entries_to_keep]
        test_set.bag_owners = [test_set.bag_owners[i] for i in entries_to_keep]
        y_tests.append(test_set.data) # y_tests is the full data
        assert len(noisy) == len(missing) == len(test_set)
        # Replace test data with corrupted data
        test_set.data = noisy
        train_sets[i] = train_set

        test_sets[i] = test_set
        val_sets[i] = val_set

        missings.append(missing)

    return train_sets, val_sets, test_sets, y_tests


def unpack_patients(patients, if_four):
    """
    Unpacks list of patients in a way that is compatible with our Bags dataset
    format. It is not mandatory that patients are sorted.
    """
    bags_of_codes, ids = [], []
    if if_four:
        other_attributes = {  # 'ICD9_defs_txt': {},
            'length_of_stay_icu': {}, 'd_icd9': {}, 'd_icd10': {}, 'p_icd9': {}, 'p_icd10': {}, 'gender': {},
            'age': {}, 'admission_type': {}, 'race': {},
            'drg_type': {}, 'drg_code': {}, 'drg_severity': {}, 'drg_mortality': {},
            'HR': {},
            'Admission Weight (Kg)': {},
            'RR': {},
            'SpO2': {},
            'Eye Opening': {},
            'Verbal Response': {},
            'Motor Response': {},
            'Alarms On': {},
            'HR Alarm - Low': {},
            'HR Alarm - High': {},
            'HOB': {},
            'SpO2 Alarm - Low': {},
            'SpO2 Alarm - High': {},
            'RUL Lung Sounds': {},
            'Turn': {},
            'LUL Lung Sounds': {},
            'Resp Alarm - High': {},
            'Resp Alarm - Low': {},
            'Heart Rhythm': {},
            'RLL Lung Sounds': {},
            'LLL Lung Sounds': {},
            'Abdominal Assessment': {},
            'Temp Site': {},
            'Parameters Checked': {},
            'Skin Integrity': {},
            'Skin Temp': {},
            'Skin Condition': {},
            'Oral Cavity': {},
            'Skin Color': {},
            'Braden Mobility': {},
            'Braden Sensory Perception': {},
            'Braden Moisture': {},
            'Braden Activity': {},
            'Braden Nutrition': {},
            'Bowel Sounds': {},
            'Activity Tolerance': {},
            'Braden Friction/Shear': {},
            'SpO2 Desat Limit': {},
            'Urine Source': {},
            'NBPs': {},
            'NBPd': {},
            'NBPm': {},
            'Diet Type': {},
            'O2 Delivery Device(s)': {},
            'Edema Location': {},
            'Temperature F': {},
            'Dorsal PedPulse R': {},
            'Dorsal PedPulse L': {},
            'Position': {},
            'Pain Present': {},
            'Gait/Transferring': {},
            'IV/Saline lock': {},
            'Ambulatory aid': {},
            'Secondary diagnosis': {},
            'Mental status': {},
            'History of falling (within 3 mnths)*': {},
            'Ectopy Type 1': {},
            'Sodium (serum)': {},
            'Potassium (serum)': {},
            'Chloride (serum)': {},
            'Creatinine (serum)': {},
            'BUN': {},
            'HCO3 (serum)': {},
            'Anion gap': {},
            'Glucose (serum)': {},
            'Hematocrit (serum)': {},
            'Hemoglobin': {},
            'Platelet Count': {},
            'WBC': {},
            'Cough Effort': {},
            'Magnesium': {},
            'NBP Alarm - Low': {},
            'NBP Alarm - High': {},
            'NBP Alarm Source': {},
            'Urine Color': {},
            'Side Rails': {},
            'Safety Measures': {},
            'Pain Location': {},
            'Education Learner': {},
            'Education Topic': {},
            'Education Method': {},
            'Education Barrier': {},
            'Phosphorous': {},
            'Education Response': {},
            'Calcium non-ionized': {},
            'Urine Appearance': {},
            'PostTib Pulses R': {},
            'PostTib Pulses L': {},
            'Problem List': {},
            'Pupil Size Right': {},
            'Pupil Size Left': {},
            'Pupil Response R': {},
            'Pupil Response L': {},
            'RLE Temp': {},
            'LLE Temp': {},
            'RLE Color': {},
            'RUE Temp': {},
            'LLE Color': {},
            'LUE Temp': {},
            'RUE Color': {},
            'LUE Color': {},
            'Support Systems': {},
            'Pain Level': {},
            'Pain Level Acceptable': {},
            'Richmond-RAS Scale': {},
            'Anti Embolic Device': {},
            'Nares R': {},
            'Pain Management': {},
            'Nares L': {},
            'Capillary Refill R': {},
        }
    else:
        other_attributes = {  # 'ICD9_defs_txt': {},
            'gender': {},
            'los_hospital': {},
            'age': {},
            'ethnicity_grouped': {},
            'admission_type': {},
            'seq_num_len': {},
            'icd9_code_d_lst': {},
            'icd9_code_p_lst': {},
            'los_icu_lst': {},'heartrate_min_lst': {},'heartrate_max_lst': {},'heartrate_mean_lst': {},'sysbp_min_lst': {},'sysbp_max_lst': {},'sysbp_mean_lst': {},'diasbp_min_lst': {},'diasbp_max_lst': {},'diasbp_mean_lst': {},'meanbp_min_lst': {},'meanbp_max_lst': {},'meanbp_mean_lst': {},'resprate_min_lst': {},'resprate_max_lst': {},'resprate_mean_lst': {},'tempc_min_lst': {},'tempc_max_lst': {},'tempc_mean_lst': {},'spo2_min_lst': {},'spo2_max_lst': {},'spo2_mean_lst': {},'glucose_min_lst': {},'glucose_max_lst': {},'glucose_mean_lst': {},
            'los_icu_lst_slope': {}, 'heartrate_min_lst_slope': {}, 'heartrate_max_lst_slope': {}, 'heartrate_mean_lst_slope': {}, 'sysbp_min_lst_slope': {}, 'sysbp_max_lst_slope': {}, 'sysbp_mean_lst_slope': {}, 'diasbp_min_lst_slope': {}, 'diasbp_max_lst_slope': {}, 'diasbp_mean_lst_slope': {}, 'meanbp_min_lst_slope': {}, 'meanbp_max_lst_slope': {}, 'meanbp_mean_lst_slope': {}, 'resprate_min_lst_slope': {}, 'resprate_max_lst_slope': {}, 'resprate_mean_lst_slope': {}, 'tempc_min_lst_slope': {}, 'tempc_max_lst_slope': {}, 'tempc_mean_lst_slope': {}, 'spo2_min_lst_slope': {}, 'spo2_max_lst_slope': {}, 'spo2_mean_lst_slope': {}, 'glucose_min_lst_slope': {}, 'glucose_max_lst_slope': {}, 'glucose_mean_lst_slope': {},
            'los_icu_lst_mean': {}, 'heartrate_min_lst_mean': {}, 'heartrate_max_lst_mean': {}, 'heartrate_mean_lst_mean': {}, 'sysbp_min_lst_mean': {}, 'sysbp_max_lst_mean': {}, 'sysbp_mean_lst_mean': {}, 'diasbp_min_lst_mean': {}, 'diasbp_max_lst_mean': {}, 'diasbp_mean_lst_mean': {}, 'meanbp_min_lst_mean': {}, 'meanbp_max_lst_mean': {}, 'meanbp_mean_lst_mean': {}, 'resprate_min_lst_mean': {}, 'resprate_max_lst_mean': {}, 'resprate_mean_lst_mean': {},
            'los_icu_lst_sd': {}, 'heartrate_min_lst_sd': {}, 'heartrate_max_lst_sd': {}, 'heartrate_mean_lst_sd': {}, 'sysbp_min_lst_sd': {}, 'sysbp_max_lst_sd': {}, 'sysbp_mean_lst_sd': {}, 'diasbp_min_lst_sd': {}, 'diasbp_max_lst_sd': {}, 'diasbp_mean_lst_sd': {}, 'meanbp_min_lst_sd': {}, 'meanbp_max_lst_sd': {}, 'meanbp_mean_lst_sd': {}, 'resprate_min_lst_sd': {}, 'resprate_max_lst_sd': {}, 'resprate_mean_lst_sd': {}, 'tempc_min_lst_sd': {}, 'tempc_max_lst_sd': {}, 'tempc_mean_lst_sd': {}, 'spo2_min_lst_sd': {}, 'spo2_max_lst_sd': {}, 'spo2_mean_lst_sd': {}, 'glucose_min_lst_sd': {}, 'glucose_max_lst_sd': {}, 'glucose_mean_lst_sd': {},
            'los_icu_lst_delta': {}, 'heartrate_min_lst_delta': {}, 'heartrate_max_lst_delta': {}, 'heartrate_mean_lst_delta': {}, 'sysbp_min_lst_delta': {}, 'sysbp_max_lst_delta': {}, 'sysbp_mean_lst_delta': {}, 'diasbp_min_lst_delta': {}, 'diasbp_max_lst_delta': {}, 'diasbp_mean_lst_delta': {}, 'meanbp_min_lst_delta': {}, 'meanbp_max_lst_delta': {}, 'meanbp_mean_lst_delta': {}, 'resprate_min_lst_delta': {}, 'resprate_max_lst_delta': {}, 'resprate_mean_lst_delta': {}, 'tempc_min_lst_delta': {}, 'tempc_max_lst_delta': {}, 'tempc_mean_lst_delta': {}, 'spo2_min_lst_delta': {}, 'spo2_max_lst_delta': {}, 'spo2_mean_lst_delta': {}, 'glucose_min_lst_delta': {}, 'glucose_max_lst_delta': {}, 'glucose_mean_lst_delta': {},
            'los_icu_lst_min': {}, 'heartrate_min_lst_min': {}, 'heartrate_max_lst_min': {}, 'heartrate_mean_lst_min': {}, 'sysbp_min_lst_min': {}, 'sysbp_max_lst_min': {}, 'sysbp_mean_lst_min': {}, 'diasbp_min_lst_min': {}, 'diasbp_max_lst_min': {}, 'diasbp_mean_lst_min': {}, 'meanbp_min_lst_min': {}, 'meanbp_max_lst_min': {}, 'meanbp_mean_lst_min': {}, 'resprate_min_lst_min': {}, 'resprate_max_lst_min': {}, 'resprate_mean_lst_min': {}, 'tempc_min_lst_min': {}, 'tempc_max_lst_min': {}, 'tempc_mean_lst_min': {}, 'spo2_min_lst_min': {}, 'spo2_max_lst_min': {}, 'spo2_mean_lst_min': {}, 'glucose_min_lst_min': {}, 'glucose_max_lst_min': {}, 'glucose_mean_lst_min': {},
            'los_icu_lst_max': {}, 'heartrate_min_lst_max': {}, 'heartrate_max_lst_max': {}, 'heartrate_mean_lst_max': {}, 'sysbp_min_lst_max': {}, 'sysbp_max_lst_max': {}, 'sysbp_mean_lst_max': {}, 'diasbp_min_lst_max': {}, 'diasbp_max_lst_max': {}, 'diasbp_mean_lst_max': {}, 'meanbp_min_lst_max': {}, 'meanbp_max_lst_max': {}, 'meanbp_mean_lst_max': {}, 'resprate_min_lst_max': {}, 'resprate_max_lst_max': {}, 'resprate_mean_lst_max': {}, 'tempc_min_lst_max': {}, 'tempc_max_lst_max': {}, 'tempc_mean_lst_max': {}, 'spo2_min_lst_max': {}, 'spo2_max_lst_max': {}, 'spo2_mean_lst_max': {}, 'glucose_min_lst_max': {}, 'glucose_max_lst_max': {}, 'glucose_mean_lst_max': {},
            'heartrate_min_lst_mm': {}, 'heartrate_max_lst_mm': {}, 'heartrate_mean_lst_mm': {}, 'sysbp_min_lst_mm': {}, 'sysbp_max_lst_mm': {}, 'sysbp_mean_lst_mm': {}, 'diasbp_min_lst_mm': {}, 'diasbp_max_lst_mm': {}, 'diasbp_mean_lst_mm': {}, 'meanbp_min_lst_mm': {}, 'meanbp_max_lst_mm': {}, 'meanbp_mean_lst_mm': {}, 'resprate_min_lst_mm': {}, 'resprate_max_lst_mm': {}, 'resprate_mean_lst_mm': {}, 'tempc_min_lst_mm': {}, 'tempc_max_lst_mm': {}, 'tempc_mean_lst_mm': {}, 'spo2_min_lst_mm': {}, 'spo2_max_lst_mm': {}, 'spo2_mean_lst_mm': {}, 'glucose_min_lst_mm': {}, 'glucose_max_lst_mm': {}, 'glucose_mean_lst_mm': {}

        }

    for patient in patients:
        # Extract 'ids'
        ids.append(patient["hadm_id"])  # id of the patients
        try:
            # Subject may be missing
            # bags_of_codes.append(patient["ndc_list"])
            bags_of_codes.append(patient.get("ndc_list", []))  # this is the codes of the medication
        except KeyError:
            bags_of_codes.append([])

        # Features that can be easily used: age, gender, ethnicity, adm_type, icu_stay_seq, hosp_stay_seq
        # Use dict here such that we can also deal with unsorted ids
        c_hadm_id = patient["hadm_id"]
        for c_var in other_attributes.keys():
            if c_var == "NDC_defs_txt" or c_var not in patient.keys():
                continue
            other_attributes[c_var][c_hadm_id] = patient[c_var]

    '''
    The format of other_attributes is as follows:
    {
        'gender': {131072: 'M', 131073: 'M'}, 
        'los_hospital': {131072: 3.2708, 131073: 8.2917}, 
        'age': {131072: 70.0, 131073: 84.0}, 
        'ethnicity_grouped': {131072: 'white', 131073: 'white'}
        ...
    }
    '''
    return bags_of_codes, ids, other_attributes

### B. Model Initialization

def initialize_models(model_name, model_params, if_four):
    conditions = CONDITIONS_MIMIC4 if if_four else CONDITIONS_MIMIC3
    if model_name == "Count Based Recommender":
        return Countbased(model_params)
    if model_name == "SVD Recommender":
        return SVDRecommender(model_params)
    if model_name == "Vanilla AE":
        model = AAERecommender(adversarial=False, prior='gauss', gen_lr=0.001, reg_lr=0.001, conditions=None, **ae_params)
        model.model_params = model_params
        return model
    if model_name == "Vanilla AE with Condition":
        model = AAERecommender(adversarial=False, prior='gauss', gen_lr=0.001, reg_lr=0.001, conditions=conditions, **ae_params)
        model.model_params = model_params
        return model
    if model_name == "Denosing AE":
        model = DAERecommender(conditions=None, **ae_params)
        model.model_params = model_params
        return model
    if model_name == "Denosing AE with Condition":
        model = DAERecommender(conditions=conditions, **ae_params)
        model.model_params = model_params
        return model
    if model_name == "Variational AE":
        model = VAERecommender(conditions=None, **vae_params)
        model.model_params = model_params
        return model
    if model_name == "Variational AE with Condition":
        model = VAERecommender(conditions=conditions, **vae_params)
        model.model_params = model_params
        return model
    if model_name == "Adverearial AE":
        model = AAERecommender(adversarial=True, prior='gauss', gen_lr=0.001, reg_lr=0.001, conditions=None, **ae_params)
        model.model_params = model_params
        return model
    if model_name == "Adverearial AE with Condition":
        model = AAERecommender(adversarial=True, prior='gauss', gen_lr=0.001, reg_lr=0.001, conditions=conditions, **ae_params)
        model.model_params = model_params
        return model

### C: Cross-Validation Helper Functions


def get_best_params(model_name, outfile, hyperparams_to_try, if_four, split_sets_filename=None, drop=0.5):

    with open(split_sets_filename, "rb") as openfile:
        train_sets, val_sets, test_sets, y_tests = pickle.load(openfile)

    log('Optimizing on following hyper params: ', logfile=outfile)
    log(hyperparams_to_try, logfile=outfile)
    best_params, _, _ = hyperparam_optimize(model_name, train_sets[0].copy(), val_sets[0].copy(), if_four, tunning_params=hyperparams_to_try, drop=drop)
    log('After hyperparam_optimize best params: ', logfile=outfile)
    log(best_params, logfile=outfile)
    return best_params




def run_cv_pipeline(drop, n_folds, outfile, model_name, best_params, if_four, split_sets_filename=None, fold_index=-1):


    metrics_per_drop_per_model = []

    # Load split sets if available
    with open(split_sets_filename, "rb") as openfile:
        train_sets, val_sets, test_sets, y_tests = pickle.load(openfile)


    best_params = best_params

    for c_fold in range(n_folds):
        if fold_index >= 0 and c_fold != fold_index:
            continue

        print(f"Starting fold {c_fold}")
        log("FOLD = {}".format(c_fold), logfile=outfile)
        log("TIME: {}".format(datetime.now().strftime("%Y-%m-%d-%H:%M")), logfile=outfile)
        train_set = train_sets[c_fold]
        val_set = val_sets[c_fold]
        test_set = test_sets[c_fold] # dropped items
        y_test = y_tests[c_fold] if y_tests else None # the full items

        log("Train set:", logfile=outfile)
        log(train_set, logfile=outfile)

        log("Validation set:", logfile=outfile)
        log(val_set, logfile=outfile)

        log("Test set:", logfile=outfile)
        log(test_set, logfile=outfile)


        y_test = lists2sparse(y_test, test_set.size(1)).tocsr(copy=False) if y_test is not None else None
        x_test = lists2sparse(test_set.data, test_set.size(1)).tocsr(copy=False)


        # log('=' * 78, logfile=outfile)
        log(model_name, logfile=outfile)
        log("training model \n TIME: {}  ".format(datetime.now().strftime("%Y-%m-%d-%H:%M")), logfile=outfile)


        model = initialize_models(model_name, best_params, if_four)
        gc.collect()

        # try:
        print(f"Training model for fold {c_fold}")
        model.train(train_set)

        print(f"Model training completed for fold {c_fold}")

        # Prediction
        y_pred = model.predict(test_set)

        print(f"Prediction completed for fold {c_fold}")
        log(" TRAIN AND PREDICT COMPLETE \n TIME: {}".format(datetime.now().strftime("%Y-%m-%d-%H:%M")),
            logfile=outfile)
        # Sanity-fix #1 make sparse stuff dense expect array

        if sp.issparse(y_pred):
            y_pred = y_pred.toarray()
        else:
            y_pred = np.asarray(y_pred)
        # Sanity-fix remove predictions for already present items

        # y_pred = remove_non_missing(y_pred, x_test, copy=False)
        # remove_non_missing will scale the matrix to 0-1

        # reduce memory usage
        del test_set
        del train_set
        del val_set
        del model

        # Evaluate metrics
        if y_test is not None and y_test.size > 0:
            results = evaluate(y_test, y_pred, METRICS)
            # log("-" * 78, logfile=outfile)
            for metric, stats in zip(METRICS, results):
                log("* FOLD#{} {}: {} ({})".format(c_fold, metric, *stats), logfile=outfile)
                metrics_per_drop_per_model.append([c_fold, drop, model_name, metric, stats[0], stats[1]])
            # log('=' * 78, logfile=outfile)

    # Return result metrics
    metrics_df = pd.DataFrame(metrics_per_drop_per_model,
                              columns=['fold', 'drop', 'model', 'metric', 'metric_val', 'metric_std'])
    return metrics_df


### D. Hyperparameter Optimization Helper Functions

def hyperparam_optimize(model_name, train_set, val_set, if_four,
                        tunning_params={'prior': ['gauss'], 'gen_lr': [0.001], 'reg_lr': [0.001],
                                        'n_code': [10, 25, 50], 'n_epochs': [20, 50, 100],
                                        'batch_size': [100], 'n_hidden': [100], 'normalize_inputs': [True]},
                        metric='maf1@10', drop=0.5):
    noisy, y_val = corrupt_sets(val_set.data, drop=drop)
    val_set.data = noisy


    exp_grid_n_combs = [len(x) for x in tunning_params.values()]
    exp_grid_cols = tunning_params.keys()
    l_rows = list(it.product(*tunning_params.values()))
    exp_grid_df = pd.DataFrame(l_rows, columns=exp_grid_cols)
    reses = []
    y_val = lists2sparse(y_val, val_set.size(1)).tocsr(copy=False)
    # the known items in the test set, just to not recompute
    x_val = lists2sparse(val_set.data, val_set.size(1)).tocsr(copy=False)

    # process = psutil.Process(os.getpid())
    # print("MEMORY USAGE: {}".format(process.memory_info().rss))

    for c_idx, c_row in exp_grid_df.iterrows():
        # gc.collect()
        # if hasattr(model, 'reset_parameters'):
        #     model.reset_parameters()
        # else:
        #     model = copy.deepcopy(model_cpy)
        model = initialize_models(model_name, c_row.to_dict(), if_four)
        # model.model_params = c_row.to_dict()
        # THE GOLD (put into sparse matrix)
        model.train(train_set)
        # Prediction
        y_pred = model.predict(val_set)

        del model
        # Sanity-fix #1, make sparse stuff dense, expect array
        if sp.issparse(y_pred):
            y_pred = y_pred.toarray()
        else:
            y_pred = np.asarray(y_pred)
        # Sanity-fix, remove predictions for already present items
        y_pred = remove_non_missing(y_pred, x_val, copy=False)
        # Evaluate metrics
        results = evaluate(y_val, y_pred, [metric])[0][-2]
        print("Model Params: ", c_row.to_dict(), "MF1", results)

        reses.append(results)

    exp_grid_df[metric] = reses
    best_metric_val = np.max(exp_grid_df[metric])
    best_params = exp_grid_df.iloc[np.where(exp_grid_df[metric].values == best_metric_val)[0][0]].to_dict()
    del best_params[metric]
    return best_params, best_metric_val, exp_grid_df


### E. Utils Functions

def log(*print_args, logfile=None):
    """ Maybe logs the output also in the file `outfile` """
    if logfile:
        with open(logfile, 'a') as fhandle:
            print(*print_args, file=fhandle)
    print(*print_args)


def save_object(obj, filename):
    with open(filename, 'wb') as outp:  # Overwrites any existing file.
        pickle.dump(obj, outp, pickle.HIGHEST_PROTOCOL)