PYTHONPATH=../../tda-nn python3 ./run.py \
    --iterations 1 \
    --training_threshold 0.85 \
    --batch-size 64 \
    --epochs 2 \
    --maxdim 1 1 1 1 \
    --threshold 10000 10000 10000 10000 \
    --n 0 1 2 3 \
    --data_samples 1000 \
    --dx 0.01 \
    --min_x 0 \
    --max_x 250 \
    --save landscapes.ph \

