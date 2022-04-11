data=$1
mkdir -p data/porn/k-fold
shuf $data > data/porn/k-fold/data.txt
split -l 94276 data/porn/k-fold/data.txt data/porn/k-fold/

./tools/dist_train.sh configs/k-fold-exp/a.py 2
sh convert.sh a
python eval.py work_dirs/a/model.engine data/porn/k-fold/ab data/porn/k-fold/ab_predict.txt

./tools/dist_train.sh configs/k-fold-exp/b.py 2
sh convert.sh b
python eval.py work_dirs/a/model.engine data/porn/k-fold/aa data/porn/k-fold/aa_predict.txt