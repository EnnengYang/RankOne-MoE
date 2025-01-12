# clip-vit-base-patch32
task_num=8
for rank_k in 16 32 64 128 256 512; do
  for select_k_factor in 0.25 0.5 0.75 1; do
    select_k=$(printf "%.0f" $(echo "$rank_k * $task_num * $select_k_factor" | bc)) || true
    rank_k_name=$(echo $rank_k | tr '.' '_') || true
    select_k_factor_name=$(echo $select_k_factor | tr '.' '_') || true
    CUDA_VISIBLE_DEVICES=0 fusion_bench \
        method=rankone_wemoe/rankone_wemoe \
        method.name=rankone_wemoe \
        method.rank_k=$rank_k \
        method.select_k=$select_k \
        fast_dev_run=false \
        modelpool=CLIPVisionModelPool/clip-vit-base-patch32_TA8 \
        taskpool=CLIPVisionModelTaskPool/clip-vit-classification_TA8 \
        report_save_path=outputs/rankone_wemoe/clip-vit-base-patch32/rankone_wemoe_rank_${rank_k_name}_select_factor_${select_k_factor_name}.json || true
  done
done

# clip-vit-base-patch16
task_num=8
for rank_k in 16 32 64 128 256 512; do
  for select_k_factor in 0.25 0.5 0.75 1; do
    select_k=$(printf "%.0f" $(echo "$rank_k * $task_num * $select_k_factor" | bc)) || true
    rank_k_name=$(echo $rank_k | tr '.' '_') || true
    select_k_factor_name=$(echo $select_k_factor | tr '.' '_') || true
    CUDA_VISIBLE_DEVICES=0 fusion_bench \
        method=rankone_wemoe/rankone_wemoe \
        method.name=rankone_wemoe \
        method.rank_k=$rank_k \
        method.select_k=$select_k \
        fast_dev_run=false \
        modelpool=CLIPVisionModelPool/clip-vit-base-patch16_TA8 \
        taskpool=CLIPVisionModelTaskPool/clip-vit-classification_TA8_B16 \
        report_save_path=outputs/rankone_wemoe/clip-vit-base-patch16/rankone_wemoe_rank_${rank_k_name}_select_factor_${select_k_factor_name}.json || true
  done
done

# clip-vit-large-patch14
task_num=8
for rank_k in 16 32 64 128 256 512; do
  for select_k_factor in 0.25 0.5 0.75 1; do
    select_k=$(printf "%.0f" $(echo "$rank_k * $task_num * $select_k_factor" | bc)) || true
    rank_k_name=$(echo $rank_k | tr '.' '_') || true
    select_k_factor_name=$(echo $select_k_factor | tr '.' '_') || true
    CUDA_VISIBLE_DEVICES=0 fusion_bench \
        method=rankone_wemoe/rankone_wemoe \
        method.name=rankone_wemoe \
        method.rank_k=$rank_k \
        method.select_k=$select_k \
        fast_dev_run=false \
        modelpool=CLIPVisionModelPool/clip-vit-large-patch14_TA8 \
        taskpool=CLIPVisionModelTaskPool/clip-vit-classification_TA8_L14 \
        report_save_path=outputs/rankone_wemoe/clip-vit-large-patch14/rankone_wemoe_rank_${rank_k_name}_select_factor_${select_k_factor_name}.json || true
  done
done

# clip-vit-base-patch32_TALL20
task_num=20
for rank_k in 32 64; do
  for select_k_factor in 0.75; do
    select_k=$(printf "%.0f" $(echo "$rank_k * $task_num * $select_k_factor" | bc)) || true
    rank_k_name=$(echo $rank_k | tr '.' '_') || true
    select_k_factor_name=$(echo $select_k_factor | tr '.' '_') || true
    CUDA_VISIBLE_DEVICES=$cuda fusion_bench \
        method=rankone_wemoe/rankone_wemoe \
        method.name=rankone_wemoe \
        method.rank_k=$rank_k \
        method.num_workers=8 \
        method.init_lambda=0.1 \
        method.select_k=$select_k \
        fast_dev_run=false \
        modelpool=CLIPVisionModelPool/clip-vit-base-patch32_TALL20 \
        taskpool=CLIPVisionModelTaskPool/clip-vit-classification_TALL20 \
        report_save_path=outputs/rankone_wemoe/clip-vit-base-patch32_TALL20/rankone_wemoe_rank_${rank_k_name}_select_factor_${select_k_factor_name}.json || true
  done
done