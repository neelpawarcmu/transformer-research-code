#!/bin/bash
NEPOCHS=20
EPOCHS=$(for x in `seq $NEPOCHS`; do if [ $x -lt 10 ]; then echo 0$x;  else echo $x; fi;  done)

for i in $EPOCHS; do
    python3 generate_ger_eng_translations.py --model_epoch $i
done
