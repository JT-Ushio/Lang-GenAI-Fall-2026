# Multi30k 四语言 CLIP 实验

服务器工作目录：`/home/taoji/data/clip-multi30k`，通过 `ssh 75` 访问。

此目录的脚本副本位于服务器 `scripts/`。数据、权重、环境和训练产物仅保存在服务器，不提交 Git。

## 环境

`env` 是单独的 Python 虚拟环境，以现有 `babylm26` 的 Python 3.13 为基础，使用 `--system-site-packages` 只读复用其 PyTorch CUDA 等依赖，没有修改原环境。因而它依赖基础环境继续存在；实际版本保存在服务器 `runs/environment.txt`。进入服务器后：

```bash
cd /home/taoji/data/clip-multi30k
source env.sh
```

## 数据与权重

- 官方文本版本：`a3d2e0d26b56f3846f66a952536ffed4e401d05a`。
- 官方文本与图片 ID：`https://github.com/multi30k/dataset` 的 `data/task1`，保存在 `data/official-task1`。
- Flickr30k 原图公开镜像：`nlphuji/flickr30k` 的 `flickr30k-images.zip`。配对以官方 Multi30k ID 为准，不采用镜像的文本或划分。
- 2017 Flickr 图片：官方仓库链接的 ERDA `multi30k_test2017_task1.zip`。
- 2017 MSCOCO 图片：根据官方列表，仅从 `images.cocodataset.org` 下载对应的 461 张原图。
- 图片许可仍由原始数据说明约束，使用镜像不改变许可。
- 图像编码器：`openai/clip-vit-base-patch32`。
- 文本编码器：`sentence-transformers/clip-ViT-B-32-multilingual-v1`，多语言 DistilBERT，按 attention mask 均值池化，再使用其已训练的 768→512 投影。与 CLIP ViT-B/32 图像空间匹配。
- 下载使用 `https://hf-mirror.com`，模型缓存保存在项目内；运行训练时仅使用本地文件。

```bash
python scripts/download_models.py --root models
python scripts/download_coco.py --root data
python scripts/prepare_data.py --root data
```

数据准备检查 ZIP CRC、图片可读性、各语言行数、重复 ID 与跨划分重叠，生成 `data/manifests/*.jsonl` 和 `data/data_report.json`。

## 256 图片演示

先用 `nvidia-smi` 确认空闲卡；搭建时 GPU 1 空闲。下面的输出目录必须不存在，避免覆盖已有实验。

```bash
CUDA_VISIBLE_DEVICES=1 python -u scripts/train_demo.py \
  --root /home/taoji/data/clip-multi30k \
  --train-limit 256 --eval-limit 256 \
  --epochs 4 --batch-size 32 --lr 1e-5 \
  --output runs/demo256_new
```

- 固定种子抽取 256 张训练图片，以及各 256 张验证/测试图片；图片 ID 保存在 `subset_ids.json`。
- 真正微调两个编码器和投影层，BF16 混合精度；不是缓存特征后仅训练线性层。
- 每批图片不重复，每图每轮选一个语言；四轮覆盖每图全部四种语言。
- 在每个子集的完整候选库上评测，不是 batch 内评测。
- 训练前记录 `baseline.json`，每轮记录 `history.json`，仅依验证集双向 R@1 四语宏平均选择 `best.pt`，最后输出 `results.json`。
- 若预训练基线优于所有微调轮次，`best_epoch=0`，如实保留基线；不能把其结果声称为微调提升。
- 双向 Recall@1/5/10 以百分比计；CLIP_Score 为正确图文对 `100 * max(cosine, 0)` 的平均，不乘温度。
- 三个测试子集的分数不是完整作业排行榜分数。需要完整评测时设 `--eval-limit 0`（会加载全部验证与测试图像，需更多主机内存和时间）。
- 这是单 GPU 教学原型；完整 29,000 图片与双 GPU 分布式训练需另外配置按批读取和跨 GPU 特征汇聚，不应直接把演示运行时间外推为完整实验耗时。

## 输出

`config.json`、`subset_ids.json`、`baseline.json`、`history.json`、`best.pt`、`last.pt`（最后一轮训练权重）、`results.json`。结果记录实际耗时与 CUDA 显存峰值，不预先填写成绩。

## 下载中途停顿时

`download_ranges.py` 会按 4 MiB 分块续传，检查服务器返回的范围和长度；若响应中提供 SHA-256，还会核对合并文件的摘要。重跑会跳过已完成块。以图像权重为例：

```bash
python scripts/download_ranges.py \
  'https://hf-mirror.com/openai/clip-vit-base-patch32/resolve/main/pytorch_model.bin?download=true' \
  models/clip-vit-base-patch32/pytorch_model.bin --workers 8
```

不要同时启动两个程序写同一目标文件。数据原始图片与权重只保存在服务器。

## 已完成的实测

见 [RESULTS.md](RESULTS.md)。完整数据已准备，256 图片演示实际运行完成；服务器产物位于 `runs/demo256/`。
