
#!/bin/bash
expl=gnnexplainer
expl_losses=(Dif)
aligners=(anchor emb)


datasets=(SST5)

edge_sizes=(0.3 0.1 1)
edge_ent=1
lr=0.003
epochs=30
align_weights=(0 0.01 1 100 1000)
for edge_size in ${edge_sizes[*]}
  do
    for expl_loss in ${expl_losses[*]}
      do
      for dataset in ${datasets[*]}
        do
          CUDA_VISIBLE_DEVICES=2 python explain_Senti.py --log --nlayer=3 --dataset=$dataset --edge_size=$edge_size --explainer=$expl --expl_loss=$expl_loss  --datatype=graph --batch_size=32 --lr=$lr --epochs=$epochs --edge_ent=$edge_ent --directional

          for aligner in ${aligners[*]}
            do
              for align_weight in ${align_weights[*]}
                do
                  CUDA_VISIBLE_DEVICES=2 python explain_Senti.py --log --nlayer=3 --dataset=$dataset --edge_size=$edge_size --batch_size=32 --explainer=$expl --datatype=graph --expl_loss=$expl_loss  --align_emb --aligner=$aligner --align_weight=$align_weight --lr=$lr --epochs=$epochs --edge_ent=$edge_ent --directional
                done
            done
        done
      done
  done

expl=pgexplainer2
edge_sizes=(0.05 0.1 0.01)
edge_ent=1
lr=0.0003
epochs=30
align_weights=(0 0.01 1 100 1000)
for edge_size in ${edge_sizes[*]}
  do
    for expl_loss in ${expl_losses[*]}
      do
      for dataset in ${datasets[*]}
        do
          CUDA_VISIBLE_DEVICES=2 python explain_Senti.py --log --nlayer=3 --dataset=$dataset --edge_size=$edge_size --explainer=$expl --expl_loss=$expl_loss  --datatype=graph --batch_size=32 --lr=$lr --epochs=$epochs --edge_ent=$edge_ent --directional
          
          for aligner in ${aligners[*]}
            do
              for align_weight in ${align_weights[*]}
                do

                  CUDA_VISIBLE_DEVICES=2 python explain_Senti.py --log --nlayer=3 --dataset=$dataset --edge_size=$edge_size --batch_size=32 --explainer=$expl --datatype=graph --expl_loss=$expl_loss  --align_emb --aligner=$aligner --align_weight=$align_weight --lr=$lr --epochs=$epochs --edge_ent=$edge_ent --directional
                done
            done
        done
      done
  done

