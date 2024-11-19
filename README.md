# "秩一专家混合用于多任务学习"的代码

## 开发环境配置
本项目依赖于[FusionBench-v0.2.4](https://github.com/tanganke/fusion_bench)，可参考该库来配置基本环境，主要步骤如下：

第一步：创建一个Conda环境
```bash
conda create --name fusionbench python=3.12.4
```

第二步：激活Conda环境
```bash
conda activate fusionbench
```

第三步：安装本项目的开发环境
```bash
git clone https://github.com/EnnengYang/RankOne-WEMoE
cd RankOne-WEMoE

pip install -e . # install the package in editable mode
```


## 运行

> 注意：本项目涉及到的所有数据集和模型权重均可在代码允许时自动下载，请确保您的网络能访问[huggingface](https://huggingface.co/)网站，或者可以考虑手动下载[相关资源](https://huhuggingface.co/tanganke).


- 实验：我们的RankOne-MoE方法在CLIP-ViT-B/32, CLIP-ViT-B/16, CLIP-ViT-L/14模型下的合并性能
```bash
bash examples/rankone_weone/rankone_wemoe_gridsearch.sh
```
> ViT-B/32的运行结果在[results/clip-vit-base-patch32](https://github.com/EnnengYang/RankOne-MoE/tree/main/examples/results/clip-vit-base-patch32)

> ViT-B/16的运行结果在[results/clip-vit-base-patch16](https://github.com/EnnengYang/RankOne-MoE/tree/main/examples/results/clip-vit-base-patch16)

> ViT-L/14的运行结果在[results/clip-vit-large-patch14](https://github.com/EnnengYang/RankOne-MoE/tree/main/examples/results/clip-vit-large-patch14)

- 实验：基线方法们在CLIP-ViT-B/32模型下的合并性能
```bash
bash examples/rankone_weone/baseline_b32.sh
```
> 运行结果在[results/clip-vit-base-patch32](https://github.com/EnnengYang/RankOne-MoE/tree/main/examples/results/clip-vit-base-patch32)

- 实验：基线方法们在CLIP-ViT-B/16模型下的合并性能
```bash
bash examples/rankone_weone/baseline_b16.sh
```
>  ViT-B/16的运行结果在[results/clip-vit-base-patch16](https://github.com/EnnengYang/RankOne-MoE/tree/main/examples/results/clip-vit-base-patch16)

- 实验：基线方法们在CLIP-ViT-L/14模型下的合并性能
```bash
bash examples/rankone_weone/baseline_l14.sh
```
> ViT-L/14的运行结果在[results/clip-vit-large-patch14](https://github.com/EnnengYang/RankOne-MoE/tree/main/examples/results/clip-vit-large-patch14)
