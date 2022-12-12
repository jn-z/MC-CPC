
python main.py \
    --train-raw /H/hjbisai/home/zjn/naval_identification/data_h5/train_zhanshi.h5 \
    --validation-raw /H/hjbisai/home/zjn/naval_identification/data_h5/train_zhanshi.h5 \
    --eval-raw /H/hjbisai/home/zjn/naval_identification/data_h5/train_zhanshi.h5 \
    --train-list ./test/train.txt \
	  --all-sets ./test/train.txt \
    --validation-list ./test/validation.txt \
    --eval-list ./test/validation.txt \
	  --index-file ./test/index_all.list \
    --index-test-file ./test/index_all.list \
    --logging-dir snapshot/cpc/ \
    --log-interval 50 --audio-window 2000 --timestep 25 --masked-frames 10 --n-warmup-steps 1000
# fi


