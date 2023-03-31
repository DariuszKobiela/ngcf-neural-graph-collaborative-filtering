set -x

./smore/cli/bpr -train ./data/amazon_book_train.txt -save ./emb/amazon_book_bpr.txt -dimensions 64 -threads 12 -sample_times 10
python3 predict.py --emb_file ./emb/amazon_book_bpr.txt --dataset amazon_book --K 1
python3 predict.py --emb_file ./emb/amazon_book_bpr.txt --dataset amazon_book --K 5
python3 predict.py --emb_file ./emb/amazon_book_bpr.txt --dataset amazon_book --K 10
python3 predict.py --emb_file ./emb/amazon_book_bpr.txt --dataset amazon_book --K 20
python3 predict.py --emb_file ./emb/amazon_book_bpr.txt --dataset amazon_book --K 40

