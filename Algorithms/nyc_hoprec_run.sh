set -x

./smore/cli/hoprec -train ./data/nyc_train.txt -field ./data/nyc_field.txt -save ./emb/nyc_hoprec.txt -dimensions 64 -threads 12 -sample_times 10
python3 predict.py --emb_file ./emb/nyc_hoprec.txt --dataset nyc --K 1
python3 predict.py --emb_file ./emb/nyc_hoprec.txt --dataset nyc --K 5
python3 predict.py --emb_file ./emb/nyc_hoprec.txt --dataset nyc --K 10
python3 predict.py --emb_file ./emb/nyc_hoprec.txt --dataset nyc --K 20
python3 predict.py --emb_file ./emb/nyc_hoprec.txt --dataset nyc --K 40

