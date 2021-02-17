#!/usr/bin/env bash

# Symlink to aminer and vectors exist
# ../aminer -> /data22/ivagliano/aminer
# ../vectors/GoogleNews-vectors-negative300.bin.gz -> /data21/lgalke/vectors/GoogleNews-vectors-negative300.bin.gz

# STOP ON ERROR
set -e 

AMINER_PY="eval/aminer.py"

echo "Branch: $(git branch | grep '^*')"

echo "Checking aminer symlink to data:"
ls -l1 . | grep "aminer"

echo "Checking symlink to word vectors:"
ls -l1 ./vectors/ | grep "GoogleNews"


DATASET="dblp"
YEAR=2018 # TODO: Verify that 2018 is correct split year for DBLP
MINCOUNT=55 # TODO ask Iacopo
RESULTS_DIR="results-drop-$DATASET-$YEAR-m$MINCOUNT"

echo "Using dataset $DATASET with split on year $YEAR and min count $MINCOUNT"
echo "Creating dir '$RESULTS_DIR' to store results"


mkdir -p "$RESULTS_DIR"

echo "Starting experiments..."

for DROP in "0.1" "0.2" "0.3" "0.4" "0.5" "0.6" "0.7" "0.8" "0.9"; do
	OUTPUT_FILE="$RESULTS_DIR"/"$DATASET-$YEAR-m$MINCOUNT-drop$DROP.txt"
	echo "python3 $AMINER_PY $YEAR --drop $DROP -d $DATASET -m $MINCOUNT -o $OUTPUT_FILE"
	python3 "$AMINER_PY" "$YEAR" --drop "$DROP" -d "$DATASET" -m "$MINCOUNT" -o "$OUTPUT_FILE"
done

