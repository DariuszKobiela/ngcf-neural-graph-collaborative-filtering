set -x

./smore/cli/hoprec -train ./data/gowalla_train.txt -field ./data/gowalla_field.txt -save ./emb/gowalla_hoprec.txt -dimensions 64 -threads 12 -sample_times 10
python3 predict.py --emb_file ./emb/gowalla_hoprec.txt --dataset gowalla --K 1
python3 predict.py --emb_file ./emb/gowalla_hoprec.txt --dataset gowalla --K 5
python3 predict.py --emb_file ./emb/gowalla_hoprec.txt --dataset gowalla --K 10
python3 predict.py --emb_file ./emb/gowalla_hoprec.txt --dataset gowalla --K 20
python3 predict.py --emb_file ./emb/gowalla_hoprec.txt --dataset gowalla --K 40

