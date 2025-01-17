# clip-vit-large-patch14

CUDA_VISIBLE_DEVICES=1 fusion_bench \
    fast_dev_run=false \
    method=wemoe/weight_ensembling_moe \
    method.name=clip_weight_ensembling_moe \
    method.use_grad_accumulate=true \
    modelpool=CLIPVisionModelPool/clip-vit-large-patch14_TA8 \
    taskpool=CLIPVisionModelTaskPool/clip-vit-classification_TA8_L14 \
    report_save_path=outputs/rankone_wemoe/clip-vit-large-patch14/weight_ensembling_moe.json  || true

CUDA_VISIBLE_DEVICES=1 fusion_bench \
    method=adamerging \
    method.name=clip_layer_wise_adamerging \
    fast_dev_run=false \
    modelpool=CLIPVisionModelPool/clip-vit-large-patch14_TA8 \
    taskpool=CLIPVisionModelTaskPool/clip-vit-classification_TA8_L14 \
    report_save_path=outputs/rankone_wemoe/clip-vit-large-patch14/layer_wise_adamerging.json || true

CUDA_VISIBLE_DEVICES=1 fusion_bench \
    method=ties_merging \
    method.scaling_factor=0.3 method.threshold=20 \
    modelpool=CLIPVisionModelPool/clip-vit-large-patch14_TA8 \
    taskpool=CLIPVisionModelTaskPool/clip-vit-classification_TA8_L14 \
    report_save_path=outputs/rankone_wemoe/clip-vit-large-patch14/ties_merging.json || true

CUDA_VISIBLE_DEVICES=1 fusion_bench \
    method=task_arithmetic \
    method.scaling_factor=0.3 modelpool=CLIPVisionModelPool/clip-vit-large-patch14_TA8 \
    taskpool=CLIPVisionModelTaskPool/clip-vit-classification_TA8_L14 \
    report_save_path=outputs/rankone_wemoe/clip-vit-large-patch14/task_arithmetic.json || true

CUDA_VISIBLE_DEVICES=1 fusion_bench \
    method=simple_average \
    modelpool=CLIPVisionModelPool/clip-vit-large-patch14_TA8 \
    taskpool=CLIPVisionModelTaskPool/clip-vit-classification_TA8_L14 \
    report_save_path=outputs/rankone_wemoe/clip-vit-large-patch14/simple_average.json || true

CUDA_VISIBLE_DEVICES=1 fusion_bench \
    method=regmean/clip_regmean \
    modelpool=CLIPVisionModelPool/clip-vit-large-patch14_TA8 \
    taskpool=CLIPVisionModelTaskPool/clip-vit-classification_TA8_L14 \
    report_save_path=outputs/rankone_wemoe/clip-vit-large-patch14/regmean.json || true

CUDA_VISIBLE_DEVICES=1 fusion_bench \
    method=fisher_merging/clip_fisher_merging \
    modelpool=CLIPVisionModelPool/clip-vit-large-patch14_TA8 \
    taskpool=CLIPVisionModelTaskPool/clip-vit-classification_TA8_L14 \
    report_save_path=outputs/rankone_wemoe/clip-vit-large-patch14/fisher_merging.json || true

CUDA_VISIBLE_DEVICES=1 fusion_bench \
    method=dare/task_arithmetic \
    modelpool=CLIPVisionModelPool/clip-vit-large-patch14_TA8 \
    taskpool=CLIPVisionModelTaskPool/clip-vit-classification_TA8_L14 \
    report_save_path=outputs/rankone_wemoe/clip-vit-large-patch14/dare.json || true

