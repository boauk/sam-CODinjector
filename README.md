# Vision-camo-SAM

面向伪装目标检测与分割（Camouflaged Object Detection, COD）的实验代码。在 Grounded-SAM 基础流程上进行任务适配，研究语义与空间先验、编码器先验注入及分组 Adapter，并提供相应的消融实验与可视化脚本。

## 实验评测

**本项目实验报告中的最终指标统一使用 [PySODEvalToolkit](https://github.com/lartpang/PySODEvalToolkit) 进行评测。**



### 百度网盘资源

实验图像及相关资源下载入口。

| 资源 | 百度网盘链接 | 提取码 |
| --- | --- | --- |
| 实验图像及相关资源 | 待补充 | 待补充 |

## 本地图像与结果路径

可按以下结构整理数据；`<dataset>` 分别替换为 `CHAMELEON`、`CAMO`、`COD10K` 或 `NC4K`。

```text
data/
├── CAMO_COD10K/
│   ├── Train/
│   │   ├── Image/          # 合并后的 CAMO 与 COD10K 训练图像
│   │   └── GT/             # 训练标注
│   └── Test/
│       ├── Image/          # 训练期间用于验证的图像
│       └── GT/             # 对应标注
└── TestDataset/
    └── <dataset>/
        ├── Image/          # 测试原图
        └── GT/             # 测试标注
```

- 训练路径通过 `--tr_npy_path`、`--train_img_dir` 和 `--train_gt_dir` 设置；验证路径通过 `--test_img_dir` 和 `--test_gt_dir` 设置。
- 测试原图和 GT 路径分别通过 `--image_dir` 和 `--gt_dir` 设置，预测图保存到 `--result_dir` 下的 `pred_masks/`。
- 例如，CAMO 的预测图可保存为 `work_dir/eval_results/CAMO/pred_masks/<image_name>.png`，与 `data/TestDataset/CAMO/GT/<image_name>.png` 对应。
- 可视化脚本通过 `--image` 指定原图，通过 `--output` 指定输出位置；支持 GT 的脚本还可通过 `--gt` 指定标注。

在 PySODEvalToolkit 中，将数据集配置的 `mask.path` 指向对应的 `GT/`，将方法配置的 `path` 指向对应的 `pred_masks/`。

## 代码入口

| 文件或目录 | 用途 |
| --- | --- |
| `train.py` | 模型训练 |
| `eval_vlsam_dino.py` | 批量推理、预测图导出及即时指标检查 |
| `vlsam_dino_model.py` | 模型与先验分支 |
| `segment_anything/` | SAM 相关组件 |
| [消融/](消融/README.md) | 各消融版本及替换说明 |
| [纯 SAM 基线](消融/纯sam/README.md) | 无外部提示的 SAM 微调基线及运行命令 |
| `visualize_m3_*.py` | 先验、特征与交叉注意力可视化 |

感谢公开数据集及 PySODEvalToolkit 的作者提供实验资源与评测工具。
