set -x

./smore/cli/bpr -train ./data/tky_train.txt -save ./emb/tky_bpr.txt -dimensions 64 -threads 12 -sample_times 10
python3 predict.py --emb_file ./emb/tky_bpr.txt --dataset tky --K 1
python3 predict.py --emb_file ./emb/tky_bpr.txt --dataset tky --K 5
python3 predict.py --emb_file ./emb/tky_bpr.txt --dataset tky --K 10
python3 predict.py --emb_file ./emb/tky_bpr.txt --dataset tky --K 20
python3 predict.py --emb_file ./emb/tky_bpr.txt --dataset tky --K 40

