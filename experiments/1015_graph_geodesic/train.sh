OUTPUT_FOLDER=$2
NETWORK_COUNT=$1

for net_id in $(seq 1 $NETWORK_COUNT);
do
    PYTHONPATH=./:../../tda-nn/ python3 trainer.py \
        --model model \
        --csv-file disk6.csv \
        --training-threshold 0.99 \
        --batch-size 2560 \
        --max-epochs 8000 \
        --learning-rate 0.01 \
        --output-name $OUTPUT_FOLDER/network${net_id} \
        ;
done
