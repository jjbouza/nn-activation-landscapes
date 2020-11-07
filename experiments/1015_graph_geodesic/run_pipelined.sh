OUTPUT_FOLDER=$1
NETWORK_COUNT=30
if [ $# -gt 0 ]; then
    mkdir -p $OUTPUT_FOLDER
    mkdir -p $OUTPUT_FOLDER/networks/
    mkdir -p $OUTPUT_FOLDER/activations/
    mkdir -p $OUTPUT_FOLDER/diagrams/
    mkdir -p $OUTPUT_FOLDER/landscapes/
    for net_id in {1..30}
    do
        echo Starting network training for network ${net_id}.
        PYTHONPATH=./:../../tda-nn/ python3 trainer.py \
            --model model \
            --training-threshold 0.8 \
            --batch-size 2560 \
            --max-epochs 8000 \
            --learning-rate 0.01 \
            --output-name $OUTPUT_FOLDER/networks/network${net_id}.pt \

        echo Starting activation computations for network ${net_id}.
        PYTHONPATH=./:../../tda-nn/ python3 ../../tda-nn/activations.py \
            --network $OUTPUT_FOLDER/networks/network${net_id}.pt \
            --input_data disk6.csv \
            --persistence-class 0 \
            --sample-count 100 \
            --layers 0 1 2 4 5 6 7 8 \
            --output_dir $OUTPUT_FOLDER/activations/network${net_id} \

        echo Starting diagram computations for network ${net_id}.
        PYTHONPATH=../../tda-nn python3 ../../tda-nn/diagram.py \
            --activations $OUTPUT_FOLDER/activations/network${net_id} \
            --persistence-data-samples 1000 \
            --max-diagram-dimension 1 1 1 1 1 1 1 1 1 1 1 1\
            --diagram-threshold 100000 100000 100000 100000 100000 100000 100000 100000 100000 100000 100000 100000\
            --persistence-layers 0 1 2 3 4 5 6 7 8 9 10 11\
            --diagram-metric GG \
            --nn-graph-k 12 \
            --save-gg-diagram-plots $OUTPUT_FOLDER/diagram_visualizations/network${net_id} \
            --output-dir $OUTPUT_FOLDER/diagrams/network${net_id} \

        echo Starting landscape computation for network ${net_id}.
        PYTHONPATH=../../tda-nn python3 ../../tda-nn/landscape.py \
            --diagrams $OUTPUT_FOLDER/diagrams/network${net_id} \
            --landscape-dx 0.001 \
            --landscape-min-x 0 \
            --landscape-max-x 5 \
            --output-dir $OUTPUT_FOLDER/landscapes/network${net_id} \
            ;
    done
else
    echo ERROR: Did not provide output folder argument.
fi
