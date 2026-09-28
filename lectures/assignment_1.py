"""作业一：Multi30k 多语言图文检索。编译不会下载数据或启动训练。"""
import numpy as np
from execute_util import link, text


# 句子数、词数、平均词数/句；沿用官方仓库统计。
STATISTICS = {
    "train": {"en": (29000, 377534, 13.0), "de": (29000, 360706, 12.4), "fr": (29000, 409845, 14.1), "cs": (29000, 297212, 10.2)},
    "val": {"en": (1014, 13308, 13.1), "de": (1014, 12828, 12.7), "fr": (1014, 14381, 14.2), "cs": (1014, 10342, 10.2)},
    "test_2016_flickr": {"en": (1000, 12968, 13.0), "de": (1000, 12103, 12.1), "fr": (1000, 13988, 14.0), "cs": (1000, 10497, 10.5)},
    "test_2017_flickr": {"en": (1000, 11376, 11.4), "de": (1000, 10758, 10.8), "fr": (1000, 12596, 12.6)},
    "test_2017_mscoco": {"en": (461, 5239, 11.4), "de": (461, 5158, 11.2), "fr": (461, 5710, 12.4)},
}
LANGUAGES = {"en": "英语", "de": "德语", "fr": "法语", "cs": "捷克语"}


def main():
    text("# Assignment 1 · Multi30k 多语言图文检索")
    text("## 使用四语言图文数据训练与评测 CLIP")
    text("本作业使用 Multi30k 四语言图文配对数据训练 CLIP，在三个测试集上评测双向检索性能与 CLIP_Score，并据此排名。")
    introduction()
    statistics()
    preparation()
    training()
    evaluation()
    score_and_ranking()
    retrieval_demo()
    deliverables()
    references()


def introduction():
    text("## 01 · 数据集介绍")
    text("Multi30k 为同一张图片提供多语言文本。本作业使用官方 task1 目录中的英、德、法、捷四语言数据；task1 仅表示数据来源。")
    text("**en：英语**。\n\n**de：德语**。\n\n**fr：法语**。\n\n**cs：捷克语**。")
    text("训练集有 29,000 张不同图片，每张对应四种语言各一条描述，可组织成 116,000 个图文对；不是 116,000 张不同图片。")
    text("### 同一视觉内容，不同语言表达")
    text("以下为教学构造的例子，不是数据集原文：")
    text("| 图片内容 | 语言 | 描述 |\n|---|---|---|\n| 猫正在吃鱼 | en | A cat is eating a fish. |\n| 同一图片 | de | Eine Katze frisst einen Fisch. |\n| 同一图片 | fr | Un chat mange un poisson. |\n| 同一图片 | cs | Kočka jí rybu. |")
    text("词序、词形和句子长度可以不同，四条描述仍然指向同一画面。模型需要学习这种跨语言、跨模态的对应关系。")


def statistics():
    text("## 02 · 数据规模")
    text("下表保留官方仓库的句子数、词数和平均句长。word（词）的统计不等于模型分词器产生的 token（词元）数。")
    for split, languages in STATISTICS.items():
        text("### " + split)
        rows = ["| 语言 | 句子数 | 词数 | 平均词数／句 |", "|---|---:|---:|---:|"]
        for language, (sentences, words, average) in languages.items():
            rows.append(f"| {LANGUAGES[language]}（{language}） | {sentences:,} | {words:,} | {average:.1f} |")
        text("\n".join(rows))
    text("**train**：29,000 张图片，四语；用于更新参数。")
    text("**val**：1,014 张图片，四语；用于调参和选择模型。")
    text("**test_2016_flickr**：1,000 张图片，四语；作为主要测试集。")
    text("**test_2017_flickr**：1,000 张图片，此处列出的版本为英德法三语；补充检验另一批 Flickr 数据。")
    text("**test_2017_mscoco**：461 张图片，英德法三语；用于补充检验不同数据来源上的图文匹配表现。")


def preparation():
    text("## 03 · 数据准备")
    text("获取官方四语言原始文本、划分对应表和对应图片。文本仓库与图片获取入口分开，只有文本不能完成图文训练。建议读取 raw 文本，再使用模型自己的分词器。")
    text("每条记录保留 `split`（划分）、`image_id`（图片标识）、`image_path`（路径）、`language`（语言）、`text`（描述）。")
    text("**配对检查**：按官方图片列表与文本行序建立对应关系，不要分别排序图片与句子；抽查图片及四语描述。")
    text("**完整性检查**：核对每种语言的样本数、图片可读性、空文本、重复 ID 和跨划分重叠。缺失或异常需记录，不能静默删减测试集。")
    text("**划分检查**：同一图片的所有语言描述留在同一划分。不能把它的英语描述用于训练、德语描述用于测试，声称是在新图片上泛化。")
    text("**许可**：遵守官方非商业研究与教学使用要求，留意图片自身权利；提交代码时不要重新打包发布整套图片。")


def training():
    text("## 04 · 使用 train 训练，使用 val 调参")
    text("**图像编码器**：例如 ViT（视觉 Transformer），把图片转换为向量。")
    text("**文本编码器**：把描述转换为向量；说明其英、德、法、捷语言支持。能输入字符不等于已经理解该语言。")
    text("**投影与归一化**：两分支输出映射到相同维度，再做 L2 归一化，使用余弦相似度比较。")
    text("**对比学习目标**：计算批次内图文相似度矩阵，最小化图→文与文→图交叉熵的平均值。")
    text("### 每张图片可以有多条正确描述")
    text("建议每批抽取 B 张不同图片，每图随机选择一种语言，在训练中尽量均衡覆盖四语。按配对顺序排列时，对角线对应正样本。")
    text("若同一批同时加入同图的多语描述，需要多正样本目标或相应掩码，避免把其他正确描述当成负样本。不同图片也可能有语义相近的描述，应在误差分析中讨论。")
    text("### 验证与选模")
    text("只使用 val 比较学习率、批次大小、冻结或微调编码器以及训练轮数。记录随机种子、模型和预训练权重、图片预处理、分词器与语言采样方式。")
    text("建议固定选模指标：完整 val 候选库上，四种语言、两个方向共 8 个 Recall@1 的算术平均值。保存该指标最好的 checkpoint（模型检查点）。")
    text("模型与评测配置确定后，再统一评测三个测试集。测试集不参与参数更新、早停或超参数选择。")
    text("### 计算预算")
    text("建议从预训练编码器微调开始，并记录训练前基线。两张 3090、12 小时以内作为预算目标，先测实际每步耗时再确定训练轮数，不能预先保证耗时。")
    text("课堂可用 256～512 张训练图片，提前缓存冻结编码器特征，只训练小型投影层；与端到端训练区分，训练子集匹配率不能代替正式测试成绩。")
    text("说明预训练权重来源；若无法确认其预训练数据是否包含测试图片，应注明这一限制。")


def evaluation():
    text("## 05 · 固定模型，评测三个测试集")
    text("每个测试集、每种语言独立建立完整候选库。不要把三个测试集合并，也不要只在一个小 batch 内检索。")
    text("**文本 → 图片**：每条描述作为查询，在该测试集全部图片中排序。")
    text("**图片 → 文本**：每张图片作为查询，在该测试集当前语言全部描述中排序。")
    text("**Recall@1**：正确答案排第一的查询比例。")
    text("**Recall@5**：正确答案进入前五的查询比例。")
    text("**Recall@10**：正确答案进入前十的查询比例。")
    text("使用 image_id 确定配对；候选顺序变化时标签也要更新，对角线不是天然的正确答案。")
    text("两个 Flickr 测试集各有 1,000 个候选，MSCOCO 有 461 个；不能把分数差异全部归因于跨域泛化能力。")
    text("评测时使用 evaluation 模式、关闭梯度和随机训练增强，保留不足一个 batch 的末尾样本。")
    text("逐语言报告结果，可附宏平均。2016 的四语平均和 2017 的三语平均语言集合不同；跨集比较应另外列出共同的英德法三语平均。")


def score_and_ranking():
    text("## 06 · CLIP_Score 与排名")
    text("**CLIP_Score**：衡量测试集中正确配对的图片与文本在所提交模型表示空间中的相似程度；本作业用训练后的 CLIP 编码器计算，不另用固定外部模型给同一批数据打分。")
    text("本作业采用 0～100 的尺度：先将图像、文本向量分别做 L2 归一化，再计算每个正确图文对的 `100 × max(余弦相似度, 0)`，最后在当前测试集、当前语言内取平均。")
    text("不要乘训练时的温度倒数或 logit_scale，不要使用 softmax 概率，也不要把所有候选的分数平均。这里计算的是按 image_id 找到的正确图文对。")
    text("原始 CLIPScore 论文使用 2.5 倍截断余弦；本作业明确使用 100 倍尺度。由于使用学生提交的多语言编码器，分数不直接与原论文结果比较。")
    text("### 建议排名规则")
    text("**第一排序指标：检索总分**。 每个测试集先平均各语言、双向的 R@1、R@5、R@10，再对三个测试集等权平均，范围为 0～100。先逐集平均，避免语言数和样本数不同造成隐含权重。")
    text("**第二排序指标：CLIP_Score 总分**。 每个测试集先平均各语言的 CLIP_Score，再对三个测试集等权平均。检索总分相同时，按 CLIP_Score 从高到低排名；两项都相同则并列。按未四舍五入的数值排序。")
    text("排名规则需在正式提交前固定；成绩表同时展示两项总分和全部逐集、逐语言结果。主办方用提交的模型检查点和统一评测脚本复算。")
    text("CLIP_Score 高不一定意味着能区分正确与错误配对：若所有输入都被映射到同一个单位向量，它也能达到 100。因此以检索为第一排序指标，CLIP_Score 为辅助指标，不将两者随意相加。")


def retrieval_demo():
    text("## 07 · 可执行演示：计算检索指标")
    text("下面是人工构造的相似度矩阵，不是 Multi30k 实测结果。行是图片，列是文本；本例正确配对恰好位于对角线。")
    scores = np.array([[0.9, 0.8, 0.1], [0.7, 0.6, 0.2], [0.1, 0.2, 0.95]])  # @inspect scores
    targets = np.arange(3)  # @inspect targets
    image_to_text = np.argsort(-scores, axis=1, kind="stable")  # @inspect image_to_text
    text_to_image = np.argsort(-scores.T, axis=1, kind="stable")  # @inspect text_to_image
    print("图→文排序（编号从 0 开始）：\n", image_to_text)
    print("文→图排序（编号从 0 开始）：\n", text_to_image)
    for k in (1, 2):
        i2t = (image_to_text[:, :k] == targets[:, None]).any(axis=1).mean()  # @inspect i2t
        t2i = (text_to_image[:, :k] == targets[:, None]).any(axis=1).mean()  # @inspect t2i
        print(f"R@{k}：图→文 {i2t:.1%}，文→图 {t2i:.1%}")
    paired_cosines = np.diag(scores)  # @inspect paired_cosines
    clip_score = 100 * np.maximum(paired_cosines, 0).mean()  # @inspect clip_score
    print(f"正确配对的 CLIP_Score：{clip_score:.2f}")
    text("此处把人工矩阵视为余弦分数示例，取正确配对的 0.9、0.6、0.95，得到 CLIP_Score＝81.67；正式评测需从归一化后的模型向量计算。")
    text("本例两个方向 R@1 均为 66.7%，R@2 均为 100%。实际作业在完整候选库上计算 R@1、R@5、R@10。")


def deliverables():
    text("## 08 · 提交要求")
    text("**可复现代码**：数据准备、训练、验证选模、最终评测入口，以及依赖版本和运行命令。")
    text("**配置与记录**：模型来源、训练参数、随机种子、硬件、耗时、验证曲线和检查点选择依据。")
    text("**结果表**：共 10 行，每行包含双向检索的 6 个指标和 1 个 CLIP_Score，共 70 个指标。检索指标用百分比，CLIP_Score 用 0～100 分；只填写实际运行的结果。")
    rows = ["| 测试集 | 语言 | 候选数 | 图→文 R@1/5/10 | 文→图 R@1/5/10 | CLIP_Score |", "|---|---|---:|---|---|---|"]
    for split, languages in STATISTICS.items():
        if split.startswith("test_"):
            for language, (count, _, _) in languages.items():
                rows.append(f"| {split} | {language} | {count} | 待评测 | 待评测 | 待评测 |")
    text("\n".join(rows))
    text("**误差分析**：展示成功和失败的检索，比较不同语言。关注动作、施事与受事、数量、颜色和空间关系，区分模型错误与多个合理候选造成的评价局限。")
    text("**建议对照**：在相同候选库与处理流程下比较微调前后成绩；若更换编码器或随机初始化投影层，说明基线的具体含义。")


def references():
    text("## 参考资料")
    link(title="Multi30k 官方数据、统计与图片入口", url="https://github.com/multi30k/dataset")
    link(title="Multi30k 原始论文（2016）", url="https://aclanthology.org/W16-3210/")
    link(title="WMT17：Flickr 与 MSCOCO 测试集", url="https://www.statmt.org/wmt17/multimodal-task.html")
    link(title="WMT18：四语言数据说明", url="https://www.statmt.org/wmt18/multimodal-task.html")
    link(title="CLIPScore 原论文：注意原论文与本作业的评分尺度区别", url="https://aclanthology.org/2021.emnlp-main.595/")


if __name__ == "__main__":
    main()
