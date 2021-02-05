OUTPUT_FOLDER=$2
NETWORK_COUNT=$1

for net_id in $(seq 1 $NETWORK_COUNT);
do
    PYTHONPATH=./:../../tda-nn/ python3 trainer.py \
        --model models/model \
        --csv-file data/circles_type_8.csv \
        --training-threshold 1.1 \
        --testing-threshold 0.9999 \
        --batch-size 512 \
        --max-epochs 8000 \
        --learning-rate 0.001 \
        --output-name $OUTPUT_FOLDER/network${net_id} \
        ;
done
