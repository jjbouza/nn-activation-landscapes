PYTHONPATH=../../tda-nn python3 ./compute_landscapes.py \
    ./mnist_cnn.pt \
    --maxdim 2 2 2 2 2\
    --threshold 10000 10000 10000 10000 10000 \
    --n 1 2 3 4 5\
    --dx 0.1 \
    --min_x 0 \
    --max_x 250 \
    --save landscapes.ph \
