cuda=1

# clip-vit-base-patch32
task_num=8
for run_id in 1 2 3 4 5 6 7 8 9 10; do
  for rank_k in 32; do
    for select_k_factor in 0.75; do
      select_k=$(printf "%.0f" $(echo "$rank_k * $task_num * $select_k_factor" | bc)) || true
      rank_k_name=$(echo $rank_k | tr '.' '_') || true
      select_k_factor_name=$(echo $select_k_factor | tr '.' '_') || true
      CUDA_VISIBLE_DEVICES=$cuda fusion_bench \
          method=rankone_wemoe/rankone_wemoe \
          method.name=rankone_wemoe \
          method.rank_k=$rank_k \
          method.select_k=$select_k \
          fast_dev_run=false \
          modelpool=CLIPVisionModelPool/clip-vit-base-patch32_TA8 \
          taskpool=CLIPVisionModelTaskPool/clip-vit-classification_TA8 \
          report_save_path=outputs/rankone_wemoe/clip-vit-base-patch32/multi_runs/rankone_wemoe_rank_${rank_k_name}_select_factor_${select_k_factor_name}_runid_${run_id}.json || true
    done
  done
done

# clip-vit-base-patch16
task_num=8
for run_id in 1 2 3 4 5 6 7 8 9 10; do
  for rank_k in 32; do
    for select_k_factor in 0.75; do
      select_k=$(printf "%.0f" $(echo "$rank_k * $task_num * $select_k_factor" | bc)) || true
      rank_k_name=$(echo $rank_k | tr '.' '_') || true
      select_k_factor_name=$(echo $select_k_factor | tr '.' '_') || true
      CUDA_VISIBLE_DEVICES=$cuda fusion_bench \
          method=rankone_wemoe/rankone_wemoe \
          method.name=rankone_wemoe \
          method.rank_k=$rank_k \
          method.select_k=$select_k \
          fast_dev_run=false \
          modelpool=CLIPVisionModelPool/clip-vit-base-patch16_TA8 \
          taskpool=CLIPVisionModelTaskPool/clip-vit-classification_TA8_B16 \
          report_save_path=outputs/rankone_wemoe/clip-vit-base-patch16/multi_runs/rankone_wemoe_rank_${rank_k_name}_select_factor_${select_k_factor_name}_runid_${run_id}.json || true
    done
  done
done

# clip-vit-large-patch14
task_num=8
for run_id in 1 2 3 4 5 6 7 8 9 10; do
  for rank_k in 32; do
    for select_k_factor in 0.75; do
      select_k=$(printf "%.0f" $(echo "$rank_k * $task_num * $select_k_factor" | bc)) || true
      rank_k_name=$(echo $rank_k | tr '.' '_') || true
      select_k_factor_name=$(echo $select_k_factor | tr '.' '_') || true
      CUDA_VISIBLE_DEVICES=$cuda fusion_bench \
          method=rankone_wemoe/rankone_wemoe \
          method.name=rankone_wemoe \
          method.rank_k=$rank_k \
          method.select_k=$select_k \
          fast_dev_run=false \
          modelpool=CLIPVisionModelPool/clip-vit-large-patch14_TA8 \
          taskpool=CLIPVisionModelTaskPool/clip-vit-classification_TA8_L14 \
          report_save_path=outputs/rankone_wemoe/clip-vit-large-patch14/multi_runs/rankone_wemoe_rank_${rank_k_name}_select_factor_${select_k_factor_name}_runid_${run_id}.json || true
    done
  done
done