OUTPUT_FOLDER=$1
if [ $# -gt 0 ]; then
    mkdir -p $OUTPUT_FOLDER
    PYTHONPATH=../../tda-nn python3 ./run.py \
        --output_folder $OUTPUT_FOLDER \
        --model model \
        --network-count 30 \
        --training-threshold 0.995 \
        --batch-size 2560 \
        --max-epochs 8000 \
        --learning-rate 0.01 \
        --diagram-metric GG \
        --nn-graph-k 12 \
        --max-diagram-dimension 1 1 1 1 1 1 1 1 1 1 1 1\
        --diagram-threshold 100000 100000 100000 100000 100000 100000 100000 100000 100000 100000 100000 100000\
        --persistence-layers 0 1 2 3 4 5 6 7 8 9 10 11\
        --persistence-data-samples 4000 \
        --landscape-dx 0.001 \
        --landscape-min-x 0 \
        --landscape-max-x 5 \
        --persistence-class 1 \
        --save-landscape \
        --save-diagram \
        --save-activations \
        --save-mean-landscapes \
        --ignore-failed \
        #| tee $OUTPUT_FOLDER/log.txt
else
    echo ERROR: Did not provide output folder argument.
fi
