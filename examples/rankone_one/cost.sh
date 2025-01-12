cuda=1

# task_arithmetic
CUDA_VISIBLE_DEVICES=$cuda fusion_bench \
    method=task_arithmetic \
    method.scaling_factor=0.3 \
    modelpool=CLIPVisionModelPool/clip-vit-base-patch32_TA8 \
    taskpool=CLIPVisionModelTaskPool/clip-vit-classification_TA8 \
    report_save_path=outputs/rankone_wemoe/rebuttal/clip-vit-base-patch32/task_arithmetic.json || true

CUDA_VISIBLE_DEVICES=$cuda fusion_bench \
    method=task_arithmetic \
    method.scaling_factor=0.3 \
    modelpool=CLIPVisionModelPool/clip-vit-large-patch14_TA8 \
    taskpool=CLIPVisionModelTaskPool/clip-vit-classification_TA8_L14 \
    report_save_path=outputs/rankone_wemoe/rebuttal/clip-vit-large-patch14/task_arithmetic.json || true

CUDA_VISIBLE_DEVICES=$cuda fusion_bench \
    method=task_arithmetic \
    method.scaling_factor=0.3 \
    modelpool=CLIPVisionModelPool/clip-vit-base-patch32_TALL20 \
    taskpool=CLIPVisionModelTaskPool/clip-vit-classification_TALL20 \
    report_save_path=outputs/rankone_wemoe/rebuttal/clip-vit-base-patch32_TALL20/task_arithmetic.json || true

## adamerging
CUDA_VISIBLE_DEVICES=$cuda fusion_bench \
    method=adamerging \
    method.name=clip_layer_wise_adamerging \
    fast_dev_run=false \
    modelpool=CLIPVisionModelPool/clip-vit-base-patch32_TA8 \
    taskpool=CLIPVisionModelTaskPool/clip-vit-classification_TA8 \
    report_save_path=outputs/rankone_wemoe/rebuttal/clip-vit-base-patch32/layer_wise_adamerging.json || true

CUDA_VISIBLE_DEVICES=$cuda fusion_bench \
    method=adamerging \
    method.name=clip_layer_wise_adamerging \
    fast_dev_run=false \
    modelpool=CLIPVisionModelPool/clip-vit-large-patch14_TA8 \
    taskpool=CLIPVisionModelTaskPool/clip-vit-classification_TA8_L14 \
    report_save_path=outputs/rankone_wemoe/rebuttal/clip-vit-large-patch14/layer_wise_adamerging.json || true

CUDA_VISIBLE_DEVICES=$cuda fusion_bench \
    method=adamerging \
    method.name=clip_layer_wise_adamerging \
    fast_dev_run=false \
    modelpool=CLIPVisionModelPool/clip-vit-base-patch32_TALL20 \
    taskpool=CLIPVisionModelTaskPool/clip-vit-classification_TALL20 \
    report_save_path=outputs/rankone_wemoe/rebuttal/clip-vit-base-patch32_TALL20/layer_wise_adamerging.json || true

## weight_ensembling_moe
CUDA_VISIBLE_DEVICES=$cuda fusion_bench \
    method=wemoe/weight_ensembling_moe \
    method.name=clip_weight_ensembling_moe \
    fast_dev_run=false \
    modelpool=CLIPVisionModelPool/clip-vit-base-patch32_TA8 \
    taskpool=CLIPVisionModelTaskPool/clip-vit-classification_TA8 \
    report_save_path=outputs/rankone_wemoe/rebuttal/clip-vit-base-patch32/weight_ensembling_moe.json || true

CUDA_VISIBLE_DEVICES=$cuda fusion_bench \
    method=wemoe/weight_ensembling_moe \
    method.name=clip_weight_ensembling_moe \
    method.use_grad_accumulate=true \
    fast_dev_run=false \
    modelpool=CLIPVisionModelPool/clip-vit-large-patch14_TA8 \
    taskpool=CLIPVisionModelTaskPool/clip-vit-classification_TA8_L14 \
    report_save_path=outputs/rankone_wemoe/rebuttal/clip-vit-large-patch14/weight_ensembling_moe.json || true

CUDA_VISIBLE_DEVICES=$cuda fusion_bench \
    method=wemoe/weight_ensembling_moe \
    method.name=clip_weight_ensembling_moe \
    fast_dev_run=false \
    modelpool=CLIPVisionModelPool/clip-vit-base-patch32_TALL20 \
    taskpool=CLIPVisionModelTaskPool/clip-vit-classification_TALL20 \
    report_save_path=outputs/rankone_wemoe/rebuttal/clip-vit-base-patch32_TALL20/weight_ensembling_moe.json || true

## rankone_wemoe
task_num=8
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
        report_save_path=outputs/rankone_wemoe/rebuttal/clip-vit-base-patch32/rankone_wemoe_rank_${rank_k_name}_select_factor_${select_k_factor_name}.json || true
  done
done

task_num=8
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
        method.use_grad_accumulate=true \
        fast_dev_run=false \
        modelpool=CLIPVisionModelPool/clip-vit-large-patch14_TA8 \
        taskpool=CLIPVisionModelTaskPool/clip-vit-classification_TA8_L14 \
        report_save_path=outputs/rankone_wemoe/rebuttal/clip-vit-large-patch14/rankone_wemoe_rank_${rank_k_name}_select_factor_${select_k_factor_name}.json || true
  done
done

task_num=20
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
        modelpool=CLIPVisionModelPool/clip-vit-base-patch32_TALL20 \
        taskpool=CLIPVisionModelTaskPool/clip-vit-classification_TALL20 \
        report_save_path=outputs/rankone_wemoe/rebuttal/clip-vit-base-patch32_TALL20/rankone_wemoe_rank_${rank_k_name}_select_factor_${select_k_factor_name}.json || true
  done
done
