# "秩一专家混合用于多任务学习"

## 开发环境配置
本项目依赖于[FusionBench-v0.2.4](https://github.com/tanganke/fusion_bench)，可参考该库来配置基本环境，或者按照如下步骤创建开发环境：

第一步：创建一个Conda环境
```bash
conda create --name rankone-moe python=3.12.4
```

第二步：激活Conda环境
```bash
conda activate rankone-moe
```

第三步：安装本项目的依赖的环境
```bash
git clone https://github.com/EnnengYang/RankOne-MoE
cd RankOne-MoE

pip install -e . # install the package in editable mode
```


## 运行实验

> 注意：本项目涉及到的所有数据集和模型权重均可在代码运行时自动下载，请确保您的网络能访问[huggingface](https://huggingface.co/)网站，您也可以考虑手动下载[相关资源](https://huggingface.co/tanganke).

<br>

- 实验：我们的RankOne-MoE方法在CLIP-ViT-B/32, CLIP-ViT-B/16, CLIP-ViT-L/14模型下的合并性能(8个任务)
```bash
bash examples/rankone_one/rankone_moe_gridsearch.sh
```
> ViT-B/32(8个任务)的运行结果在[examples/results/clip-vit-base-patch32](https://github.com/EnnengYang/RankOne-MoE/tree/main/examples/results/clip-vit-base-patch32)

> ViT-B/16(8个任务)的运行结果在[examples/results/clip-vit-base-patch16](https://github.com/EnnengYang/RankOne-MoE/tree/main/examples/results/clip-vit-base-patch16)

> ViT-L/14(8个任务)的运行结果在[examples/results/clip-vit-large-patch14](https://github.com/EnnengYang/RankOne-MoE/tree/main/examples/results/clip-vit-large-patch14)

<br>


- 实验：基线方法们在CLIP-ViT-B/32模型下的合并性能(8个任务)
```bash
bash examples/rankone_one/baseline_b32.sh
```
> ViT-B/32(8个任务)的运行结果在[examples/results/clip-vit-base-patch32](https://github.com/EnnengYang/RankOne-MoE/tree/main/examples/results/clip-vit-base-patch32)

<br>

- 实验：基线方法们在CLIP-ViT-B/16模型下的合并性能(8个任务)
```bash
bash examples/rankone_one/baseline_b16.sh
```
>  ViT-B/16(8个任务)的运行结果在[examples/results/clip-vit-base-patch16](https://github.com/EnnengYang/RankOne-MoE/tree/main/examples/results/clip-vit-base-patch16)

<br>

- 实验：基线方法们在CLIP-ViT-L/14模型下的合并性能(8个任务)
```bash
bash examples/rankone_one/baseline_l14.sh
```
> ViT-L/14(8个任务)的运行结果在[examples/results/clip-vit-large-patch14](https://github.com/EnnengYang/RankOne-MoE/tree/main/examples/results/clip-vit-large-patch14)

<br>


- 实验：所有对比方法们在CLIP-ViT-B/32模型下的合并性能(20个任务)
```bash
bash examples/rankone_one/baseline_b32_20tasks.sh
```
> ViT-B/32(20个任务)的运行结果在[examples/results/clip-vit-base-patch32_20Tasks](https://github.com/EnnengYang/RankOne-MoE/tree/main/examples/results/clip-vit-base-patch32_20Tasks)

<br>
