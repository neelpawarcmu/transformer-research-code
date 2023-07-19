#!/bin/bash
END=20
for i in $(seq 1 $END); do
    python3 generate_ger_eng_translations.py --model_epoch $i
done