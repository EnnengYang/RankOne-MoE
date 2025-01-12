# clip-vit-base-patch32
cuda=1

CUDA_VISIBLE_DEVICES=$cuda fusion_bench \
    method=wemoe/weight_ensembling_moe \
    method.name=clip_weight_ensembling_moe \
    method.num_workers=8 \
    method.init_lambda=0.0125 \
    fast_dev_run=false \
    modelpool=CLIPVisionModelPool/clip-vit-base-patch32_TALL20 \
    taskpool=CLIPVisionModelTaskPool/clip-vit-classification_TALL20 \
    report_save_path=outputs/rankone_wemoe/clip-vit-base-patch32_TALL20/weight_ensembling_moe.json  || true

CUDA_VISIBLE_DEVICES=$cuda fusion_bench \
    method=adamerging \
    method.name=clip_layer_wise_adamerging \
    method.init_values=0.2 \
    fast_dev_run=false \
    modelpool=CLIPVisionModelPool/clip-vit-base-patch32_TALL20 \
    taskpool=CLIPVisionModelTaskPool/clip-vit-classification_TALL20 \
    report_save_path=outputs/rankone_wemoe/clip-vit-base-patch32_TALL20/layer_wise_adamerging.json || true

CUDA_VISIBLE_DEVICES=$cuda fusion_bench \
    method=task_arithmetic \
    method.scaling_factor=0.075 \
    modelpool=CLIPVisionModelPool/clip-vit-base-patch32_TALL20 \
    taskpool=CLIPVisionModelTaskPool/clip-vit-classification_TALL20 \
    report_save_path=outputs/rankone_wemoe/clip-vit-base-patch32_TALL20/task_arithmetic.json || true

CUDA_VISIBLE_DEVICES=$cuda fusion_bench \
    method=ties_merging \
    method.scaling_factor=0.05 \
    method.threshold=20 \
    modelpool=CLIPVisionModelPool/clip-vit-base-patch32_TALL20 \
    taskpool=CLIPVisionModelTaskPool/clip-vit-classification_TALL20 \
    report_save_path=outputs/rankone_wemoe/clip-vit-base-patch32_TALL20/ties_merging.json || true

CUDA_VISIBLE_DEVICES=$cuda fusion_bench \
    method=simple_average \
    modelpool=CLIPVisionModelPool/clip-vit-base-patch32_TALL20 \
    taskpool=CLIPVisionModelTaskPool/clip-vit-classification_TALL20 \
    report_save_path=outputs/rankone_wemoe/clip-vit-base-patch32_TALL20/simple_average.json || true

CUDA_VISIBLE_DEVICES=$cuda fusion_bench \
    method=regmean/clip_regmean \
    modelpool=CLIPVisionModelPool/clip-vit-base-patch32_TALL20 \
    taskpool=CLIPVisionModelTaskPool/clip-vit-classification_TALL20 \
    report_save_path=outputs/rankone_wemoe/clip-vit-base-patch32_TALL20/regmean.json || true

CUDA_VISIBLE_DEVICES=$cuda fusion_bench \
    method=fisher_merging/clip_fisher_merging \
    modelpool=CLIPVisionModelPool/clip-vit-base-patch32_TALL20 \
    taskpool=CLIPVisionModelTaskPool/clip-vit-classification_TALL20 \
    report_save_path=outputs/rankone_wemoe/clip-vit-base-patch32_TALL20/fisher_merging.json || true
