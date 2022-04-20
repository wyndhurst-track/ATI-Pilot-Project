python3 train/train_triplet_resnet.py --folds_file="/home/will/work/1-RA/src/Datasets/data/OpenSetCows2019/30-70.pkl"
mv /home/will/work/1-RA/src/Identifier/Supervised-Triplet-Network/results/* /home/will/work/CEiA/results/STL/30-70

python3 train/train_triplet_resnet.py --folds_file="/home/will/work/1-RA/src/Datasets/data/OpenSetCows2019/20-80.pkl"
mv /home/will/work/1-RA/src/Identifier/Supervised-Triplet-Network/results/* /home/will/work/CEiA/results/STL/20-80

python3 train/train_triplet_resnet.py --folds_file="/home/will/work/1-RA/src/Datasets/data/OpenSetCows2019/10-90.pkl"
mv /home/will/work/1-RA/src/Identifier/Supervised-Triplet-Network/results/* /home/will/work/CEiA/results/STL/10-90