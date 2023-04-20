set -x

./smore/cli/bpr -train ./data/nyc_train.txt -save ./emb/nyc_bpr.txt -dimensions 64 -threads 12 -sample_times 10
python3 predict.py --emb_file ./emb/nyc_bpr.txt --dataset nyc --K 1
python3 predict.py --emb_file ./emb/nyc_bpr.txt --dataset nyc --K 5
python3 predict.py --emb_file ./emb/nyc_bpr.txt --dataset nyc --K 10
python3 predict.py --emb_file ./emb/nyc_bpr.txt --dataset nyc --K 20
python3 predict.py --emb_file ./emb/nyc_bpr.txt --dataset nyc --K 40

