#!/bin/bash

expl=pgexplainer
aligners=(anchor emb)
expl_losses=(Entropy)

datasets=(Tree_grid)
edge_sizes=(0.1 1 0.05)
edge_ent=1
lr=0.003
epochs=30
align_weights=(0 0.00001 0.0001 0.001 0.01 0.1 1 10 100 1000 10000)
for edge_size in ${edge_sizes[*]}
  do
    for expl_loss in ${expl_losses[*]}
      do
      for dataset in ${datasets[*]}
        do
          #CUDA_VISIBLE_DEVICES=2 python explain_GNN.py --log --nlayer=3 --dataset=$dataset --edge_size=$edge_size --explainer=$expl --expl_loss=$expl_loss --lr=$lr --epochs=$epochs --edge_ent=$edge_ent --directional
          #CUDA_VISIBLE_DEVICES=2 python explain_consistency.py --nlayer=3 --dataset=$dataset --log --edge_size=$edge_size --explainer=$expl --expl_loss=$expl_loss --lr=$lr --epochs=$epochs --edge_ent=$edge_ent --directional

          for aligner in ${aligners[*]}
            do
              for align_weight in ${align_weights[*]}
                do
                  #CUDA_VISIBLE_DEVICES=2 python explain_consistency.py --nlayer=3 --dataset=$dataset --log --edge_size=$edge_size --explainer=$expl --expl_loss=$expl_loss --align_emb --align_with_grad --aligner=$aligner --align_weight=$align_weight --lr=$lr --epochs=$epochs --edge_ent=$edge_ent --directional
                  #python explain_consistency.py --nlayer=3 --dataset=$dataset --log --edge_size=$edge_size --explainer=$expl --expl_loss=$expl_loss --align_emb --aligner=$aligner --align_weight=$align_weight --lr=$lr --epochs=$epochs --edge_ent=$edge_ent --directional
                  CUDA_VISIBLE_DEVICES=3 python explain_GNN.py --log --nlayer=3 --dataset=$dataset --edge_size=$edge_size --explainer=$expl --expl_loss=$expl_loss --aligner=$aligner --align_weight=$align_weight --align_emb --lr=$lr --epochs=$epochs --edge_ent=$edge_ent --directional
                  #python explain_GNN.py --log --nlayer=3 --dataset=$dataset --edge_size=$edge_size --explainer=$expl --expl_loss=$expl_loss --aligner=$aligner --align_weight=$align_weight  --align_with_grad --align_emb --lr=$lr --epochs=$epochs --edge_ent=$edge_ent --directional
                done
            done
        done
      done
  done


expl=pgexplainer2
datasets=(mutag)
edge_sizes=(0.05)
edge_ent=1
lr=0.0003
epochs=30
align_weights=(0 0.00001 0.0001 0.001 0.01 0.1 1 10 100 1000 10000)
for edge_size in ${edge_sizes[*]}
  do
    for expl_loss in ${expl_losses[*]}
      do
      for dataset in ${datasets[*]}
        do
          #CUDA_VISIBLE_DEVICES=1 python explain_GNN.py --log --nlayer=3 --dataset=$dataset --edge_size=$edge_size --explainer=$expl --expl_loss=$expl_loss --directional --datatype=graph --batch_size=32 --lr=$lr --epochs=$epochs --edge_ent=$edge_ent
          #CUDA_VISIBLE_DEVICES=1 python explain_consistency.py --log --nlayer=3 --dataset=$dataset --edge_size=$edge_size --explainer=$expl --expl_loss=$expl_loss --directional --datatype=graph --batch_size=32 --lr=$lr --epochs=$epochs --edge_ent=$edge_ent
          
          for aligner in ${aligners[*]}
            do
              for align_weight in ${align_weights[*]}
                do

                  #CUDA_VISIBLE_DEVICES=1 python explain_GNN.py --log --nlayer=3 --dataset=$dataset --edge_size=$edge_size --batch_size=32 --datatype=graph --explainer=$expl  --expl_loss=$expl_loss --directional --align_emb --align_with_grad --aligner=$aligner --align_weight=$align_weight --lr=$lr --epochs=$epochs --edge_ent=$edge_ent
                  CUDA_VISIBLE_DEVICES=1 python explain_GNN.py --log --nlayer=3 --dataset=$dataset --edge_size=$edge_size --batch_size=32 --explainer=$expl --datatype=graph --expl_loss=$expl_loss --directional --align_emb --aligner=$aligner --align_weight=$align_weight --lr=$lr --epochs=$epochs --edge_ent=$edge_ent
                  #CUDA_VISIBLE_DEVICES=1 python explain_consistency.py --log --nlayer=3 --dataset=$dataset --edge_size=$edge_size --explainer=$expl --batch_size=32 --datatype=graph --expl_loss=$expl_loss --directional --align_emb --align_with_grad --aligner=$aligner --align_weight=$align_weight --lr=$lr --epochs=$epochs --edge_ent=$edge_ent
                  #CUDA_VISIBLE_DEVICES=1 python explain_consistency.py --log --nlayer=3 --dataset=$dataset --edge_size=$edge_size --explainer=$expl --batch_size=32 --datatype=graph --expl_loss=$expl_loss --directional --align_emb --aligner=$aligner --align_weight=$align_weight --lr=$lr --epochs=$epochs --edge_ent=$edge_ent
                done
            done
        done
      done
  done