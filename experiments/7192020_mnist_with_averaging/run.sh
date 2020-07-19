PYTHONPATH=../../tda-nn python3 ./run.py \
    --iterations 16 \
    --batch-size 64 \
    --epochs 1 \
    --maxdim 1 1 1 1 1\
    --threshold 10000 10000 10000 10000 10000 \
    --n 1 2 3 4 5 \
    --data_samples 1000 \
    --dx 0.1 \
    --min_x 0 \
    --max_x 250 \
    --save landscapes.ph \

