#!/bin/sh

echo "Running baselines"
python3 mimic.py -mi 0 > log_cb.log 2>&1 || true
python3 mimic.py -mi 1 > log_svd.log 2>&1 || true

echo "Running AEs"
python3 mimic.py -mi 2 > log_ae.log 2>&1 || true
python3 mimic.py -mi 3 > log_aec.log 2>&1 || true

echo "Running DAEs"
python3 mimic.py -mi 4 > log_dae.log 2>&1 || true
python3 mimic.py -mi 5 > log_daec.log 2>&1 || true

echo "Running VAEs"
python3 mimic.py -mi 6 > log_vae.log 2>&1 || true
python3 mimic.py -mi 7 > log_vaec.log 2>&1 || true

echo "Running AAEs"
python3 mimic.py -mi 8 > log_aae.log 2>&1 || true
python3 mimic.py -mi 9 > log_aaec.log 2>&1 || true

#python3 mimic.py -mi 10 -le 1 > log_vaect.log 2>&1 || true
#python3 mimic.py -mi 11 -le 0 > log_aae.log 2>&1 || true
#python3 mimic.py -mi 12 -le 0 > log_aaec.log 2>&1 || true
#python3 mimic.py -mi 13 -le 1 > log_aaect.log 2>&1 || true
echo "DONE"