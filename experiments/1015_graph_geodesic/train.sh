OUTPUT_FOLDER=$2
NETWORK_COUNT=$1

for net_id in $(seq 1 $NETWORK_COUNT);
do
    PYTHONPATH=./:../../tda-nn/ python3 trainer.py \
        --model models/model \
        --csv-file data/circles_type_8.csv \
        --training-threshold 0.5 0.6 0.7 0.8 0.9 1.0 1.1 \
        --testing-threshold 0.9999 \
        --batch-size 512 \
        --max-epochs 8000 \
        --learning-rate 0.001 \
        --output-name $OUTPUT_FOLDER/network${net_id} \
        --log-name $OUTPUT_FOLDER/log.csv \
        ;
done
