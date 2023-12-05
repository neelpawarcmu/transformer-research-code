#!/bin/bash
# Run this script from the project root directory
NEPOCHS=20
EPOCHS=$(for x in `seq $NEPOCHS`; do if [ $x -lt 10 ]; then echo 0$x;  else echo $x; fi;  done)
NLAYERS=6

for n in `seq 1 $NLAYERS`; do
    python3 train_model.py --epochs $NEPOCHS --N $n  --batch_size 16
done

for i in $EPOCHS; do
    python3 generate_ger_eng_translations.py --epoch $i --N 1 --num_examples 5 
    python3 generate_ger_eng_translations.py --epoch $i --N 2 --num_examples 5 
    python3 generate_ger_eng_translations.py --epoch $i --N 3 --num_examples 5 
    python3 generate_ger_eng_translations.py --epoch $i --N 4 --num_examples 5 
    python3 generate_ger_eng_translations.py --epoch $i --N 5 --num_examples 5 
    python3 generate_ger_eng_translations.py --epoch $i --N 6 --num_examples 5
done
