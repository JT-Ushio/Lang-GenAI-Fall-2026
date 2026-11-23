# 第八讲备课说明：音乐生成

主线为节拍与节奏、旋律与音程、和声、音色、曲式、音乐表示，以及歌曲生成模型。网页正文、代码与试听位于 lecture_08.py；配套演示源 content.json 共 44 页。语音规范化、声调、TTS、声码器与通用波形重建已移出本讲主线。

## 45 分钟 × 2 课时

第一课时 45 分钟：产业与任务 5′，拍与节奏 7′，音高与移调 6′，和声 5′，音色与曲式 5′，音乐表示和分层条件 5′，模型路线与时间线 4′，MusicGen 架构与编码 8′。约对应演示源第 1—20 页。

第二课时 45 分钟：微型旋律模型训练与试听 7′，ACE-Step 规划和流匹配 7′，歌词与 YuE2 5′，Muse 数据与分段训练 8′，指标与 Suno 证据 4′，多音轨与混音实验 8′，评价、授权与回顾 6′。约对应第 21—44 页。

完整小模型训练可在备课时编译，课堂查看真实损失和生成轨迹。时间不足时，将延迟排列、连续潜变量 shape 和标签右移作为可展开代码，优先保留移调、和弦、节奏及分轨试听。

## 原创音乐案例

- 基础旋律为 MIDI [60,62,64,67,69,67,64,60]，对应 C、D、E、G、A 五声音阶。所有音乐事件和解析合成音均为本课程序构造。
- 拍号比较：120 BPM，4/4 与 3/4 各两小节。仅用高低点击示意小节首拍的重音，不宣称这种重音是所有音乐的唯一模式。
- 节奏比较：等时值八音符与 [.5,.5,1,2,.5,.5,1,2] 拍版本，总长均为八拍、四秒。代码打印起音位置与持续时间。
- 移调：所有音高加两个半音，验证相邻音程不变、频率比恒为 2^(2/12)。这里讨论十二平均律，不把该体系当作所有音乐的唯一标准。
- 和弦：C–E–G 与 C–E♭–G，验证相对半音 [0,4,7] 和 [0,3,7]；试听只改三度音，不把大调小调固定解释为快乐悲伤。
- 音色：保持音高、节奏与包络，只改变谐波配比；是解析合成音，不是精确的钢琴或古筝物理建模。
- 符号事件为 [8,3]，三列是音高、起音拍数与持续拍数；解析渲染得到 [64000] 的 16 kHz 波形。该数组是简化事件表，不宣称写出了标准 MIDI 文件。
- 三轨低音、鼓点、旋律各四秒，单声道。混音加权求和，静音鼓点不修改其他音轨；没有运行大型分轨生成或音源分离。
- 小型旋律模型为两层因果 Transformer、17,927 参数；十条训练乐句、两个条件、七个 token，240 次更新。音符生成真实执行，再由解析合成器渲染；是拟合演示，不是大模型音乐质量评测。
- MusicGen 官方试听来自作者页面，非本机推理。其余 MusicGen、ACE、Muse 等 shape 和标签演示不加载预训练权重。流匹配示例用已知端点演示目标速度，不训练速度网络。

## 本地编译与试听

从 lectures 执行 python3 execute.py -m lecture_08，再从 trace-viewer 执行 npm run build 和 npm run preview -- --host 127.0.0.1 --port 4173。

入口：http://127.0.0.1:4173/?trace=var%2Ftraces%2Flecture_08.json&step=1&animate=1 。

重建新增音乐素材：从仓库根目录执行 python3 lectures/scripts/build_lecture_08_music.py；分轨素材执行 python3 lectures/scripts/build_lecture_08_multitrack.py。需要 matplotlib；常规编译仅用现有 torch/numpy 与已保存素材。

Moshi 保留为独立延伸文件 lecture_08_moshi_appendix.py，可单独编译 python3 execute.py -m lecture_08_moshi_appendix，不进入音乐主线。原演示文字备份为 presentations/lecture_08/moshi_appendix.json。

## 商业数据与时间线

2025 年全球录制音乐收入 317 亿美元、付费订阅账户用户 8.37 亿，来源 IFPI 2026 报告。中国 5190.35 亿元为更广的产业范围，不能直接比较或计算全球占比。

Suno 2026-06-03 官方公告 D 轮融资超过 4 亿美元、投后估值 54 亿美元；累计超过 1 亿创作者、2026 年初超过 200 万付费订阅者、3 亿美元年化运行率来源官方 Careers 页面。累计用户不是 MAU，年化运行率不是全年审计收入；公司估值不是模型估值。核对日 2026-11-16。

时间线除 Suno 用产品公告外，采用论文首次提交日；团队名只列核心机构。来源随对应页面保存在 content.json 和网页链接中。

## 模型版本与参考资料

- MusicGen 报告 https://arxiv.org/abs/2306.05284 ，四码本、50 Hz、2048 码本容量与 small 结构来自 AudioCraft 公开配置。示意图由 AI 辅助生成，采用中文条件描述。官方样例实际提示为英文，不能据中文讲解图推断中文提示效果。
- MusicGen 官方样例 https://ai.honu.io/papers/musicgen/ ，输入为 Pop dance track with catchy melodies, tropical percussion, and upbeat rhythms, perfect for the beach。取 samples/musicgen/000_sample.mp3 前八秒，转 32 kHz 单声道。页面归于 MusicGen 3.3B 样例。版权与来源归作者和权利人。
- ACE-Step 1.5 报告 https://arxiv.org/abs/2602.00744 ，结构图采用报告 Figure 2。按该报告的约 2B DiT、48 kHz 立体声、25 Hz × 64 维 VAE 和 5 Hz 规划码讲解。项目后续 XL 版本另有结构配置。
- YuE2 报告 https://arxiv.org/abs/2609.33757 ，结构图采用 yue2_overview.png。其符号、语义、声学联合路线对应 YuE2，区别于 2025 年原 YuE。
- EnCodec https://arxiv.org/abs/2210.13438 ，Flow Matching https://arxiv.org/abs/2210.02747 。
- Muse https://arxiv.org/abs/2601.03973 ，仓库 https://github.com/yuhui1038/Muse 。采用 v3 的 Figure 1 与数据流程 Figure 2。主表模型为 Muse-0.6B；116,489 首合成中英歌曲，训练用 Qwen3 和 MuCodec。数据流程正文写 GPT-5 mini，图内标 GPT-5.1，本课只称“文字模型”，避免把该细节当一致配置。作者声明许可发布，使用应核对数据与权重各自条款。
- 指标比较使用 Muse Table 1 同一评测：早期 ACE-Step 3.5B 的 PER/MuLan-T/PQ 为 0.39/0.28/7.31，Muse-0.6B 为 0.16/0.33/8.14。Muse 默认额外接受分段条件，基线仅全局条件，不能视为严格等输入实验。前者引用 https://arxiv.org/abs/2506.00045 ，不是 1.5。ACE-Step 1.5 的 Table 1 没有 Muse，测试集合与设置不同，不拼接表格排名。
- Suno v6 官方发布说明 https://suno.com/release-notes/introducing-v6 ，日期 2026-09-09。检索截至 2026-11-16，未找到可核验的 v5/v6 完整公开架构与训练报告；这不构成“绝不存在”的证明。Bark https://github.com/suno-ai/bark 是早期公开实现，不能推断为 Suno 歌曲产品的架构。
- SongGen https://arxiv.org/abs/2502.13128 ，仓库 https://github.com/LiuZH-19/SongGen 。Figure 2 架构，公开 Interleaving A–V 权重生成独立人声与伴奏，两轨在推理代码分别返回，可加权混音。
- MusicGen-Stem https://arxiv.org/abs/2501.01757 ，作者试听 https://simonrouard.github.io/musicgenstem/ 。bass/drums/other 三轨及条件补轨，当前课件核验了报告与试听，不把论文中将发布代码权重的承诺等同于已验证下载。

公开代码和公开权重分别检查许可。MusicGen 代码 MIT、权重 CC BY-NC 4.0。ACE-Step 1.5 项目为 MIT。YuE2 代码 Apache-2.0，模型权重 CC BY-NC 4.0 并有附加个人创作与学术使用条款，具体见其 MODEL_LICENSE。课程选材依据公开报告和实现，未复现这些模型的训练规模或论文分数。
