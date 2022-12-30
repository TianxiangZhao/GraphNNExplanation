#!/bin/bash

#expls=(gnnexplainer pgexplainer pgexplainer2)
expls=(pgexplainer pgexplainer2)

expl_losses=(Dif)
aligners=(anchor emb)
datasets=(SpuriousMotif_0 SpuriousMotif_0.3 SpuriousMotif_0.5 SpuriousMotif_0.7 SpuriousMotif_1)
edge_sizes=(0.005)
edge_ent=1
lr=0.005
epochss=(10 30 50 100)
align_weights=(0 0.1 1 10)

for expl in ${expls[*]}
  do
    for epochs in ${epochss[*]}
      do
        for edge_size in ${edge_sizes[*]}
        do
            for expl_loss in ${expl_losses[*]}
            do
            for dataset in ${datasets[*]}
                do
                CUDA_VISIBLE_DEVICES=2 python explain_GNN.py --log --nlayer=3 --dataset=$dataset --edge_size=$edge_size --explainer=$expl --expl_loss=$expl_loss --directional --lr=$lr --dropout=0 --epochs=$epochs --edge_ent=$edge_ent --datatype=graph --batch_size=32
                CUDA_VISIBLE_DEVICES=2 python explain_consistency.py --nlayer=3 --dataset=$dataset --log --edge_size=$edge_size --explainer=$expl --expl_loss=$expl_loss --directional --dropout=0 --lr=$lr --epochs=$epochs --edge_ent=$edge_ent --datatype=graph --batch_size=32

                for aligner in ${aligners[*]}
                    do
                    for align_weight in ${align_weights[*]}
                        do
                        CUDA_VISIBLE_DEVICES=2 python explain_consistency.py --nlayer=3 --dataset=$dataset --log --edge_size=$edge_size --explainer=$expl --expl_loss=$expl_loss --dropout=0 --align_emb --align_with_grad --aligner=$aligner --align_weight=$align_weight --lr=$lr --epochs=$epochs --edge_ent=$edge_ent --directional --datatype=graph --batch_size=32
                        CUDA_VISIBLE_DEVICES=2 python explain_consistency.py --nlayer=3 --dataset=$dataset --log --edge_size=$edge_size --explainer=$expl --expl_loss=$expl_loss --dropout=0 --align_emb --aligner=$aligner --align_weight=$align_weight --lr=$lr --epochs=$epochs --edge_ent=$edge_ent --directional --datatype=graph --batch_size=32
                        CUDA_VISIBLE_DEVICES=2 python explain_GNN.py --log --nlayer=3 --dataset=$dataset --edge_size=$edge_size --explainer=$expl --expl_loss=$expl_loss --directional --dropout=0 --aligner=$aligner --align_weight=$align_weight --align_emb --lr=$lr --epochs=$epochs --edge_ent=$edge_ent --datatype=graph --batch_size=32
                        CUDA_VISIBLE_DEVICES=2 python explain_GNN.py --log --nlayer=3 --dataset=$dataset --edge_size=$edge_size --explainer=$expl --expl_loss=$expl_loss --directional --dropout=0 --aligner=$aligner --align_weight=$align_weight  --align_with_grad --align_emb --lr=$lr --epochs=$epochs --edge_ent=$edge_ent --datatype=graph --batch_size=32
                        done
                    done
                done
            done
        done
      done
  done