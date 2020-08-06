PYTHONPATH=../../tda-nn python3 ./run.py \
    --iterations 5 \
    --training_threshold 1.0 \
    --batch-size 64 \
    --epochs 2 \
    --maxdim 1 1 1 1 \
    --threshold 100000 100000 100000 100000 \
    --n 0 1 2 3 \
    --data_samples 1000 \
    --dx 0.01 \
    --min_x 0 \
    --max_x 5 \
    --save landscapes.ph \

