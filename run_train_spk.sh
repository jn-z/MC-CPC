#!/bin/bash
    python spk_class_train.py \
    --train-raw /H/hjbisai/home/zjn/naval_identification/data_h5/train_zhanshi.h5 \
    --validation-raw /H/hjbisai/home/zjn/naval_identification/data_h5/train_zhanshi.h5 \
    --eval-raw /H/hjbisai/home/zjn/naval_identification/data_h5/train_zhanshi.h5 \
    --train-list ./test/train.txt \
	  --all-sets ./test/train.txt \
    --validation-list ./test/validation.txt \
    --eval-list ./test/validation.txt \
	  --index-file ./test/index_all.list \
    --index-test-file ./test/index_all.list \
    --logging-dir ./snapshot/resnet/ --log-interval 50 \
    --model-path ./snapshot/cpc/cpc-model_best.pth \
    --log-interval 50 --audio-window 2000 --timestep 25 --n-warmup-steps 1000

