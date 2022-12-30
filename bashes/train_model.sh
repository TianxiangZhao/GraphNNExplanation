#python train_GNN.py --dataset=BA_shapes --nlayer=3 --epoch=1010 --save --lr=0.01 > "./logs/BA_shapes_cls"
#python train_GNN.py --dataset=infected --nlayer=3 --epoch=1010 --save --lr=0.01 > "./logs/infected_cls" #
#python train_GNN.py --dataset=Tree_cycle --nlayer=3 --epoch=1010 --save --lr=0.002 > "./logs/TreeCycle_cls" 
#python train_GNN.py --dataset=Tree_grid --nlayer=3 --epoch=5010 --save --lr=0.004 > "./logs/TreeGrid_cls"


#python train_GNN.py --dataset=LoadBA_shapes --nlayer=3 --epoch=5010 --save --lr=0.004  > "./logs/LoadBA_shapes_cls"
#python train_GNN.py --dataset=LoadTree_cycle --nlayer=3 --epoch=1010 --save --lr=0.002 > "./logs/LoadTreeCycle_cls" 
#python train_GNN.py --dataset=LoadTree_grid --nlayer=3 --epoch=5010 --save --lr=0.004 > "./logs/LoadTreeGrid_cls"



#python train_GNN.py --dataset=mutag --nlayer=3 --epoch=1010 --save --batch_size=64 --lr=0.001 --datatype=graph --task=gcls > "./logs/mutag_cls"

#CUDA_VISIBLE_DEVICES=1 python train_GNN.py --dataset=SpuriousMotif_0 --nlayer=3 --epoch=1010 --save --batch_size=640 --lr=0.01 --datatype=graph --dropout=0 --task=gcls > "./logs/SpuriousMotif_0_cls"
#CUDA_VISIBLE_DEVICES=1 python train_GNN.py --dataset=SpuriousMotif_0.3 --nlayer=3 --epoch=1010 --save --batch_size=640 --lr=0.01 --datatype=graph --dropout=0 --task=gcls > "./logs/SpuriousMotif_03_cls"
#CUDA_VISIBLE_DEVICES=1 python train_GNN.py --dataset=SpuriousMotif_0.5 --nlayer=3 --epoch=1010 --save --batch_size=640 --lr=0.01 --datatype=graph --dropout=0 --task=gcls > "./logs/SpuriousMotif_05_cls"
#CUDA_VISIBLE_DEVICES=1 python train_GNN.py --dataset=SpuriousMotif_0.7 --nlayer=3 --epoch=1010 --save --batch_size=640 --lr=0.01 --datatype=graph  --dropout=0 --task=gcls > "./logs/SpuriousMotif_07_cls"
#CUDA_VISIBLE_DEVICES=1 python train_GNN.py --dataset=SpuriousMotif_1 --nlayer=3 --epoch=1010 --save --batch_size=640 --lr=0.01 --datatype=graph --dropout=0 --task=gcls > "./logs/SpuriousMotif_1_cls"

CUDA_VISIBLE_DEVICES=1 python train_GNN.py --dataset=Twitter --nlayer=3 --epoch=1010 --save --batch_size=640 --lr=0.01 --datatype=graph --dropout=0 --task=gcls > "./logs/Twitter"
CUDA_VISIBLE_DEVICES=1 python train_GNN.py --dataset=SST2 --nlayer=3 --epoch=1010 --save --batch_size=640 --lr=0.01 --datatype=graph --dropout=0 --task=gcls > "./logs/SST2"
CUDA_VISIBLE_DEVICES=1 python train_GNN.py --dataset=SST5 --nlayer=3 --epoch=1010 --save --batch_size=640 --lr=0.01 --datatype=graph --dropout=0 --task=gcls > "./logs/SST5"