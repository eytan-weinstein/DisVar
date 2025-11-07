#!/bin/bash
# nohup bash run_batches.sh > nohup.txt &

UNIPROT_FILE="/valr/eytan/projects/DisVar/data/human_UniProt_IDs.tsv"
SAVE_DIR="/neuhaus/eytan/DisVar_protein_data"

# Ensure save directory exists
mkdir -p "$SAVE_DIR"

# Read UniProt IDs and filter out those that already exist
echo "Reading UniProt IDs..."
mapfile -t ALL_IDS < <(cut -f1 "$UNIPROT_FILE")
NONEXISTENT_IDS=()
for ID in "${ALL_IDS[@]}"; do
    if [ ! -f "$SAVE_DIR/$ID.json" ]; then
        NONEXISTENT_IDS+=("$ID")
    fi
done

TOTAL=${#NONEXISTENT_IDS[@]}
echo "Found $TOTAL missing UniProt IDs."

# Determine batch size for 20 batches
BATCHES=20
BATCH_SIZE=$(( (TOTAL + BATCHES - 1) / BATCHES ))

# Launch 20 background processes
for i in $(seq 0 $((BATCHES-1))); do
    START=$(( i * BATCH_SIZE ))
    END=$(( START + BATCH_SIZE - 1 ))
    if [ $END -ge $TOTAL ]; then
        END=$(( TOTAL - 1 ))
    fi

    # Slice the IDs for this batch
    BATCH_IDS=("${NONEXISTENT_IDS[@]:$START:$((END-START+1))}")

    # Launch Python script for this batch in the background
    (
        for ID in "${BATCH_IDS[@]}"; do
            python protein.py --UniProt "$ID" --save_dir "$SAVE_DIR"
        done
    ) &
done

echo "All batches launched in background."