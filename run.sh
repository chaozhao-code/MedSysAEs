#!/bin/sh

echo "Running baselines mimic3"
python3 mimic.py -mi 0 > results/log_cb_mimic3.log 2>&1 || true
python3 mimic.py -mi 1 > results/log_svd_mimic3.log 2>&1 || true

echo "Running AEs mimic3"
python3 mimic.py -mi 2 > results/log_ae_mimic3.log 2>&1 || true
python3 mimic.py -mi 3 > results/log_aec_mimic3.log 2>&1 || true

echo "Running DAEs mimic3"
python3 mimic.py -mi 4 > results/log_dae_mimic3.log 2>&1 || true
python3 mimic.py -mi 5 > results/log_daec_mimic3.log 2>&1 || true

echo "Running VAEs mimic3"
python3 mimic.py -mi 6 > results/log_vae_mimic3.log 2>&1 || true
python3 mimic.py -mi 7 > results/log_vaec_mimic3.log 2>&1 || true

echo "Running AAEs mimic3"
python3 mimic.py -mi 8 > results/log_aae_mimic3.log 2>&1 || true
python3 mimic.py -mi 9 > results/log_aaec_mimic3.log 2>&1 || true

echo "Running baselines mimic4"
python3 mimic.py -mi 0 -f > results/log_cb_mimic4.log 2>&1 || true
python3 mimic.py -mi 1 -f > results/log_svd_mimic4.log 2>&1 || true

echo "Running AEs mimic4"
python3 mimic.py -mi 2 -f > results/log_ae_mimic4.log 2>&1 || true
python3 mimic.py -mi 3 -f > results/log_aec_mimic4.log 2>&1 || true

echo "Running DAEs mimic4"
python3 mimic.py -mi 4 -f > results/log_dae_mimic4.log 2>&1 || true
python3 mimic.py -mi 5 -f > results/log_daec_mimic4.log 2>&1 || true

echo "Running VAEs mimic4"
python3 mimic.py -mi 6 -f > results/log_vae_mimic4.log 2>&1 || true
python3 mimic.py -mi 7 -f > results/log_vaec_mimic4.log 2>&1 || true

echo "Running AAEs mimic4"
python3 mimic.py -mi 8 -f > results/log_aae_mimic4.log 2>&1 || true
python3 mimic.py -mi 9 -f > results/log_aaec_mimic4.log 2>&1 || true

echo "DONE"