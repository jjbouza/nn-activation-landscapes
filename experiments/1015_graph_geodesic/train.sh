PYTHONPATH=./:../../tda-nn/ python3 trainer.py \
    --model model \
    --csv-file disk6.csv \
    --training-threshold 0.8 \
    --batch-size 2560 \
    --max-epochs 8000 \
    --learning-rate 0.01 \
    --output-name output/network
