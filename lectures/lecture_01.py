"""第一讲：课程导论。两课时各 45 分钟；讲授提示见 notes/lecture_01.md。"""

from execute_util import image, link, text, video


def main():
    text("# 以语言为中心的生成式AI")
    text("## Language-Centric Generative AI")
    text("### 第一讲 · 语言中心生成式AI导论")
    text("复旦大学 · 外国语言文学学院｜FORE30067.01｜2026—2027 学年第一学期")

    course_opening()
    three_videos()
    multimodal_release_news()
    multilingual_song_demo()
    student_ai_survey()
    artificial_analysis()
    what_is_generative_ai()
    history_and_milestones()
    future_directions()
    limitations_and_risks()
    course_information()


def course_opening():
    text("### 教学团队")
    text("**纪焘｜授课教师** · 外国语言文学学院助理教授")
    text("联系邮箱：taoji@fudan.edu.cn"), text("    "), link(title="教师主页", url="https://www.taoji.me/")
    image("images/lecture_01/Instuctor_and_TA.png", width=600)
    text("答疑时间、教室、课程群：TA联系大家创建。")

    text("## 01 · 为什么要开这门课？")
    text("### 从春季课程出发")
    text("《自然语言处理与语言习得》：理解语言如何被表示、学习与处理。")
    text("这门课继续追问：语言如何连接图像、声音、视频，并指挥模型与工具完成任务？")
    image("images/lecture_01/language-interface.svg", width=600, style={"maxWidth": "calc(100vw - 120px)"})
    text("### 为什么强调『以语言为中心』？")
    text("语言是**任务接口**：用自然语言描述目标、对象、风格、受众和约束。")
    text("语言是**语义桥梁**：把画面、声音和动作关联到共同的概念。")
    text("语言是**控制接口**：提出计划、调用工具、解释反馈并修正结果。")
    text("### 学完这门课，你应当能做到")
    text("1. 解释一个多模态系统如何完成表示、对齐、生成与交互。")
    text("2. 用语义、语用和结构约束分析模型为什么成功、为什么出错。")
    text("3. 动手训练一个视觉语言模型。")


def three_videos():
    text("## 02 · 先看三个视频：生成式AI正在被如何呈现？")

    text("### 视频一：GPT-6 Astra 宣传片｜约 2′43″")
    video("https://www.bilibili.com/video/BV1CJbc6JExi/", width=900, style={"maxWidth": "calc(100vw - 120px)"})

    text("### 视频二：Kimi K3｜约 57″")
    video("https://www.bilibili.com/video/BV156KL6XESt/", width=900, style={"maxWidth": "calc(100vw - 120px)"})

    text("### 视频三：Qwen3.8｜约 44″")
    video("https://www.bilibili.com/video/BV1xF3f6qEJ9/", width=900, style={"maxWidth": "calc(100vw - 120px)"})

    text("🤔 你最喜欢哪个宣传片？")
    text("🤔 有哪些展示的功能超出了你的想象？")


def multimodal_release_news():
    text('## 新闻速览 · 最新模型正在把多模态带向哪里？')
    text('**近期官方发布与更新｜截至 2026-09-07｜按日期从新到旧**')

    text('### 2026-09-03 ｜ OpenAI · GPT-6 Astra')
    text('**新闻：发布新一代旗舰模型，重点面向复杂推理、编程、计算机操作与专业工作。**')
    text('**输入 → 输出：文本、图像 → 文本／代码**')
    text('看懂界面与图表，再通过工具完成任务。GPT-6 Astra API本身不直接接收音频／视频，也不原生输出图像；产品中的媒体工具需要单独区分。')
    link(title="官方发布／更新", url='https://openai.com/index/gpt-6-astra/')
    link(title="官方说明：能力与使用方式", url='https://developers.openai.com/api/docs/models/gpt-6-astra')

    text('### 2026-09-02 ｜ Qwen · Qwen3.8-Max-0902')
    text('**新闻：更新Qwen3.8-Max快照，增强图表推理、文档解析、视觉感知和多工具协作。**')
    text('**输入 → 输出：文本、图像 → 文本／代码（本条重点：视觉理解）**')
    text('从看懂材料，到完成复杂工作：让模型分析双语图表、解释信息，再组织成面向不同受众的内容。日期是快照更新日。')
    link(title="官方发布／更新", url='https://docs.qwencloud.com/changelog/models')

    text('### 2026-09-01 ｜ Anthropic · Claude Fable 5.1')
    text('**新闻：发布Claude Fable 5.1，增强长程编程、多步骤研究以及文档、表格和幻灯片工作。**')
    text('**输入 → 输出：文本、图像 → 文本／代码**')
    text('视觉材料进入持续工作的智能体流程；文档与幻灯片等交付物由模型结合工具完成。')
    link(title="官方发布／更新", url='https://www.anthropic.com/claude-fable-and-mythos-5-1')
    link(title="官方说明：能力与使用方式", url='https://platform.claude.com/docs/en/models/fable-5-1/overview')

    text('### 2026-09-01 ｜ Google · Gemini 3.7 Flash 的 Agentic Video')
    text('**新闻：推出主动视频理解功能：模型按任务搜索、扫描并回看相关片段，结合画面、音频与转写进行分析。**')
    text('**输入 → 输出：视频（画面＋音频）、文本问题 → 文本分析**')
    text('例如：从一段外语讲座中定位论点出现的时刻，并结合讲者语音和投影片解释。日期是功能发布日；同时支持3.6 Flash与3.5 Flash-Lite。')
    link(title="官方发布／更新", url='https://blog.google/innovation-and-ai/models-and-research/gemini-models/introducing-agentic-video-in-gemini/')

    text('### 2026-08-26 ｜ 智谱 · GLM-5.3-Flash')
    text('**新闻：发布GLM-5系列首个原生多模态模型，让视觉理解进入编程、浏览器操作与办公任务。**')
    text('**输入 → 输出：文本、图像、视频 → 文本／代码**')
    text('读懂网页、视频或文档页面，调用工具产出结果，再通过视觉反馈检查。视频成品属于智能体结合剪辑工具的交付。')
    link(title="官方发布／更新", url='https://github.com/zai-org/GLM-V')
    link(title="官方说明：能力与使用方式", url='https://docs.z.ai/guides/vlm/glm-5.3-flash')

    text('### 2026-08-21 ｜ DeepSeek · V4-Flash-Vision-Exp（实验版）')
    text('**新闻：在API平台上线V4 Flash视觉理解实验版，扩展看图推理与视觉智能体能力。**')
    text('**输入 → 输出：文本、图像 → 文本／代码**')
    text('从纯文本任务走向截图、图表和视觉信息分析。正式模型名含Vision-Exp，与普通V4-Flash区分。')
    link(title="官方发布／更新", url='https://api-docs.deepseek.com/updates/')
    link(title="官方说明：能力与使用方式", url='https://api-docs.deepseek.com/guides/vision/')

    text('### 2026-08-13 ｜ Suno · Studio 2.0；音乐模型 v5.5')
    text('**新闻：Suno发布Studio 2.0，加入MIDI编辑、对话式创作和更精细的分轨处理；v5.5模型于2026-03-26发布。**')
    text('**输入 → 输出：文字描述／歌词、参考音频 → 音乐音频（人声与伴奏）**')
    text('v5.5加入个人声音与风格定制；Studio 2.0还可用MIDI片段提示生成音频。MIDI是音符与演奏信息，不是录制的声音。')
    link(title="官方模型发布：Suno v5.5（2026-03-26）", url='https://about.suno.com/blog/v5-5')
    link(title="官方工具更新：Studio 2.0（2026-08-13）", url='https://about.suno.com/blog/studio-2')

    text('### 2026-07-31 ｜ 字节跳动 · Seedance 2.5')
    text('**新闻：发布Seedance 2.5，增强30秒叙事、多模态参考与音视频编辑能力。**')
    text('**输入 → 输出：文本、图像、音频、视频参考 → 视频＋音频**')
    text('支持单次最长30秒的音视频联合生成、多轮延长，以及按时间点修改内容；语言可以指定人物、动作、镜头和叙事顺序。')
    link(title="官方发布：Seedance 2.5（2026-07-31）", url='https://seed.bytedance.com/en/blog/one-take-creation-flexible-referencing-introducing-seedance-2-5')
    link(title="官方模型页面与作品示例", url='https://seed.bytedance.com/en/seedance2_5')

    text('### 2026-07-17 ｜ Kimi · K3')
    text('**新闻：发布原生支持视觉理解的Kimi K3，面向长程编程、知识工作和推理。**')
    text('**输入 → 输出：文本、图像、视频 → 文本／代码**')
    text('结合截图、视频和视觉反馈完成任务，与前面的K3宣传片对应。这里记录模型发布日，权重开放是后续事件。')
    link(title="官方发布／更新", url='https://www.kimi.com/news/kimi-k3')
    link(title="官方说明：能力与使用方式", url='https://platform.kimi.com/docs/guide/kimi-k3-quickstart')

    text('### 从近期新闻读出什么？')
    text('- 多模态能力正在进入完整任务流程。')
    text('- 语言是提出目标、解释内容和协调工具的接口。')


def multilingual_song_demo():
    text("## 演示 · B站多语言《朋友的酒》")
    text("**多语言改编：语言内容 × 演唱声音 × 音乐风格**")
    text("### 英语版 · 《朋友的酒》")
    video("https://www.bilibili.com/video/BV1Este6wExx/", width=900, style={"maxWidth": "calc(100vw - 120px)"})
    text("### 韩语版 · 《朋友的酒》")
    video("https://www.bilibili.com/video/BV1pqtG6CE59/", width=900, style={"maxWidth": "calc(100vw - 120px)"})
    text("### 日语版 · 《朋友的酒》")
    video("https://www.bilibili.com/video/BV1Nibn63EPm/", width=900, style={"maxWidth": "calc(100vw - 120px)"})
    text("### 印地语版 · 《朋友的酒》")
    video("https://www.bilibili.com/video/BV1VVtZ6oE2p/", width=900, style={"maxWidth": "calc(100vw - 120px)"})
    text("### 蒙古语版 · 《朋友的酒》")
    video("https://www.bilibili.com/video/BV1Vfbn6sE7e/", width=900, style={"maxWidth": "calc(100vw - 120px)"})
    text("### 汉越音版 · 《朋友的酒》")
    video("https://www.bilibili.com/video/BV1uBbL6eE3s/", width=900, style={"maxWidth": "calc(100vw - 120px)"})
    text("### 葡语版 · 《朋友的酒》")
    video("https://www.bilibili.com/video/BV1cCtZ6TE5a/", width=900, style={"maxWidth": "calc(100vw - 120px)"})
    text("### 法语版 · 《朋友的酒》")
    video("https://www.bilibili.com/video/BV1z5tX6DEsS/", width=900, style={"maxWidth": "calc(100vw - 120px)"})
    text("### 听完之后 · ")
    text("| 观察维度 |  |\n|---|---|\n| 语义与文化 | 相同意思、改写之处、需要核实的表达 |\n| 发音与韵律 | 清晰度、重音、停顿、歌词与旋律的配合 |\n| 声音与音乐 | 音色、唱法、配器、速度与情绪 |\n| 多模态一致性 | 字幕、演唱内容与画面是否对应 |")

    text("### 『外语＋计算机』出路+1（误）")


def student_ai_survey():
    # 建议8分钟；延伸讨论可留作课后。数据是学生自报，不能解释为因果效应。
    text("## 学生怎样使用AI？｜HEPI / Kortext 2026调查")
    text("从模型能做什么，回到我们自己：**怎样使用AI，才是在学习？**")
    text("**样本：1,054名英国全日制本科生｜调查：2025年12月｜报告：2026年3月**")
    text("Savanta开展调查，按人口特征加权；报告给出的总体误差约±3个百分点。以下是英国学生的自报结果，不代表中国或本班学生。")
    text("来源：Rose Stephenson、Charlotte Armstrong，Student Generative AI Survey 2026，HEPI Report 199，方法见第9页。")

    text("### 1 · 几乎人人在用，但『用AI』不等于『让AI代写』")
    text("**95%**报告至少一种AI用途；**94%**曾用生成式AI辅助准备考核任务。两者是不同问题的口径。")
    image("images/lecture_01/hepi/figure04-assessment-uses.png", width=850, style={"maxWidth": "calc(100vw - 120px)"})
    text("原图：HEPI / Kortext 2026，Figure 4，第13页。多选题；星号为本年新增选项；排除不足1%的『不知道』回答。")
    text("**61%解释概念，49%总结文章，40%建议研究思路，39%梳理想法；12%直接把AI生成文本纳入考核作品。**")
    text("12%在2024／2025年分别为3%／8%。是否允许取决于具体考核规则，不能把这些比例直接叫作『作弊率』。")
    text("🤔 同样是用AI总结论文：怎样判断自己理解得更深了，而不是只是少读了？")

    text("### 2 · AI正在成为阅读入口，但没有统一的使用方式")
    image("images/lecture_01/hepi/figure10-source-balance.png", width=850, style={"maxWidth": "calc(100vw - 120px)"})
    text("原图：HEPI / Kortext 2026，Figure 10，第19页。传统资料包括非AI网页、教材、讲义与课程材料。")
    text("**34%偏向传统资料，29%两者均衡，37%偏向AI；其中8%主要使用AI、很少查传统资料。**")
    text("外语阅读可以试试：先自己概括一段 → 请AI解释难点 → 回原文核对 → 合上材料独立复述。")

    text("### 3 · 更方便，不一定意味着每个人都学得更好")
    image("images/lecture_01/hepi/figure12-student-experience.png", width=850, style={"maxWidth": "calc(100vw - 120px)"})
    text("原图：HEPI / Kortext 2026，Figure 12，第20页。这里测的是主观学生体验，不是考试成绩。")
    text("**49%认为体验变好，35%认为没有显著影响，16%认为变差。**")
    text("开放回答呈现两种路径：有人借助AI理解难点，把时间用于深入分析；也有人觉得自己减少了独立思考与查证。来源：第21—24、38页，概述而非逐字引语。")

    text("### 4 · 『AI技能很重要』与『有人教我』之间有落差")
    image("images/lecture_01/hepi/figure22-subject-support.png", width=850, style={"maxWidth": "calc(100vw - 120px)"})
    text("原图：HEPI / Kortext 2026，Figure 22，第32页。绿色为同意，蓝色为中立／不知道，黄色为不同意。")
    text("感到获得职业AI技能支持的比例：**STEM 53%，社会科学50%，健康学科44%，艺术与人文学科26%**。")

    text("### 延伸讨论 · AI也在改变人与人的互动")
    text("**21%**自报AI使自己更少孤独，**20%**自报更孤独，**59%**认为没有影响。（Figure 13，第25页）")


def artificial_analysis():
    text("## 实时观察站 · Artificial Analysis")
    text("看完演示，我们换一个问题：**为具体任务选模型，要比较哪些维度？**")
    link(title="打开：模型比较——智能、性能与价格分析", url="https://artificialanalysis.ai/zh/models")
    text("### 不只看一个总分")
    text("### 对外语任务，怎样把指标变成选择？")
    text("**口语练习助手**：关注响应等待、语音交互和反馈质量。")
    text("**长篇双语资料整理**：关注上下文利用、事实与术语一致性。")
    text("**批量字幕初译**：关注批量成本、失败重试和人工修订量。")
    text("### 性能与价格：读懂测量口径")
    text("首个token、首个答案token与完整回答的等待时间不同；输出快，不代表很快开始回答。")
    text("输入、输出和缓存的费率可能不同；单位token便宜，也不保证整个任务便宜。网站的每任务成本有自己的评测工作负载。")
    text("同一内容在不同语言和分词器下的token量可能不同。做双语项目时，同时记录实际用量与交付质量。")
    link(title="核对口径：Artificial Analysis 方法论", url="https://artificialanalysis.ai/zh/methodology")
    text("### 从语言到图像：Image Arena")
    link(title="打开：图像竞技场（文生图／图像编辑）", url="https://artificialanalysis.ai/zh/image/arena")
    text("看一组成对结果：先写出你偏好的理由，再讨论是否满足提示词要求。课堂口头选择即可。")
    text("评价清单：对象、数量、空间关系、画面文字、风格。**画面好看与指令遵循分开评价。**")
    text("延伸实验：把同一含义的中英文提示交给相同模型，检查构图和约束是否一致。")
    text("示例提示：『红色方块在蓝色圆左边，画面中恰好两个物体。』先检查关系与数量，再评价美感。")
    text("### 从图像到时间：Video Arena")
    link(title="打开：视频竞技场", url="https://artificialanalysis.ai/zh/video/arena")
    text("视频需要额外观察：人物与物体能否跨帧保持、动作顺序是否成立、声音与画面是否协调。")
    text("网站还提供语音与音乐等模态的方法论入口，可在对应周次继续使用。")
    text("### 一分钟记录：留下可复核的选择理由")
    text("记录：**查看日期｜任务与语言｜模型版本／设置｜能力依据｜时间与成本｜下一步自己的小测试**。")
    text("排行榜会更新；Arena偏好、综合基准和本课程任务是不同证据，应分别解释。")


def what_is_generative_ai():
    text("## 04 · 什么是生成式AI？")
    text("### 工作定义")
    text("生成式AI是一类从数据中学习规律，并据此产生文本、图像、声音、视频等内容的人工智能方法与系统。")
    text("『生成』关注输出内容如何被构造；它并不自动意味着内容原创、事实正确或理解了世界。")
    text("### 判别、检索与生成")
    text("| 任务 | 输入 → 输出 | 例子 |\n|---|---|---|\n| 判别 | 对象 → 标签或分数 | 判断评论是正面还是负面 |\n| 检索 | 查询 → 已有材料 | 找到讨论同一主题的文章 |\n| 生成 | 条件 → 构造的内容 | 根据文章生成面向中学生的摘要 |")
    text("实际系统可以组合三者：先检索证据，再生成回答，最后检查是否被证据支持。")
    link(title="RAG：Retrieval-Augmented Generation（2020）", url="https://arxiv.org/abs/2005.11401")
    text("### 一个统一的视角：条件生成")
    text(r"我们希望生成输出 $y$，同时满足条件 $c$：$y\sim p_\theta(y\mid c)$。")
    text("$c$ 可以包含文字指令、参考图像、语音、历史对话和工具返回的结果。")
    text("$y$ 可以是文字、图像、声音、视频，也可以是供工具执行的结构化指令。")
    text("**条件决定要做什么，模型根据学到的规律构造候选结果。**")
    text("### 文本为什么能一个 token 一个 token 地生成？")
    text(r"自回归语言模型：$p_\theta(x_{1:T}\mid c)=\prod_{t=1}^{T}p_\theta(x_t\mid x_{<t},c)$。")
    text("token 是模型处理文本的单位，可能是一个字、一个词的一部分或其他片段，并不固定等于一个词。")
    text("例：『为了让游客理解这件展品，请用……』，不同续写会给后面的内容带来不同约束。")
    text("下一步的高概率选择不保证整段事实正确；这条公式也不是所有图像、音频生成方法的共同算法。")
    text("### 模型怎样学会？与我们怎样使用？")
    image("images/lecture_01/training-inference.svg", width=600, style={"maxWidth": "calc(100vw - 120px)"})
    text("**预训练**：从大量数据中学习表示与生成规律。")
    text("**后训练**：通过示范、偏好或反馈，让行为更符合任务与交互要求。")
    text("**推理／生成**：给定当前输入，用已经学到的参数产生结果。")
    text("聊天窗口只是系统入口；检索、工具、权限和人工审核也会影响最终效果。")
    text("### 一个教学视角：图像是理解多模态的关键入口")
    text("在本课程的技术主线中，我们把图像作为最重要的切入模态：先理解二维视觉表示，再联系语音和视频的表示与生成。")
    image("images/lecture_01/image-as-multimodal-bridge.png", width=600, style={"maxWidth": "calc(100vw - 120px)"})
    text("**图像：二维空间信号。** 像素沿宽度与高度排列，模型需要理解对象、位置与空间关系。")
    text("**语音：沿时间展开的一维信号。** 单声道采样序列可以类比为『单行像素』，但每个值表示振幅；转成声谱图后，横轴是时间、纵轴是频率。")
    text("**视频：按时间排列、相互关联的图像帧。** 除了每一帧的内容，还要理解运动、事件顺序和跨帧一致性；有声视频还包含同步音轨。")
    text("这些联系让图像成为有用的学习桥梁；时间、频率与空间有不同含义，不能直接把所有模态当成普通图片。")
    text("图为AI生成的概念示意；波形与声谱图用于解释表示方式，并非同一段实测音频。")

    text("## 生成式AI的核心技术")
    image("images/lecture_01/genai_tech.png", width=600, style={"maxWidth": "calc(100vw - 120px)"})




def history_and_milestones():
    text("## 05 · 生成式AI是怎样走到今天的？")
    text("概率与规则 → 神经生成模型 → 可迁移的预训练 → 多模态对齐与生成 → 交互与工具系统。")

    text("### 阶段1：从手写规则到学习数据分布")
    text("**2013／2014｜VAE**：学习潜在变量与数据之间的关系，在连续潜在空间中进行生成。")
    link(title="Kingma & Welling：Auto-Encoding Variational Bayes", url="https://arxiv.org/abs/1312.6114")
    text("**2014｜GAN**：生成器与判别器通过对抗训练，使生成样本更接近数据分布。")
    link(title="Goodfellow 等：Generative Adversarial Nets", url="https://arxiv.org/abs/1406.2661")
    text("意义：生成不再只是把模板拼在一起，而可以从数据中学出复杂内容的结构。")

    text("### 阶段2：Transformer 与大规模预训练")
    text("**2017｜Transformer**：以注意力机制为核心建模序列，成为后来许多语言与多模态模型的重要基础。")
    link(title="Vaswani 等：Attention Is All You Need", url="https://arxiv.org/abs/1706.03762")
    text("**2020｜GPT-3**：展示大规模语言模型利用上下文中的少量示例完成多种任务的能力。")
    link(title="Brown 等：Language Models are Few-Shot Learners", url="https://arxiv.org/abs/2005.14165")
    text("意义：一个模型开始服务多种任务；用户可以通过语言提供任务说明与示例。")

    text("### 阶段3：语言与视觉接上了")
    text("**2020｜DDPM**：学习逐步去噪，推动扩散模型成为高质量图像生成的重要路线。")
    link(title="Ho 等：Denoising Diffusion Probabilistic Models", url="https://arxiv.org/abs/2006.11239")
    text("**2021｜CLIP**：用图文配对数据学习可比较的表示，让自然语言参与视觉任务。")
    link(title="Radford 等：Learning Transferable Visual Models", url="https://arxiv.org/abs/2103.00020")
    text("**对齐与生成是不同问题**：CLIP 式模型学习『哪张图对应哪段文字』，本身不是图像生成器。")
    text("后续系统可以把语言条件与生成过程结合起来；会对齐不自动保证空间关系、计数与细节生成正确。")

    text("### 阶段4：从模型能力到可交互的系统")
    text("**2022｜ChatGPT**：以对话形式向公众提供生成式语言模型，强化了自然语言交互这一入口。")
    link(title="Introducing ChatGPT（2022-11-30）", url="https://openai.com/index/chatgpt/")
    text("**2023｜GPT-4、LLaVA**：前者报告图文输入到文本输出的能力；后者研究视觉指令微调。")
    link(title="GPT-4 Technical Report", url="https://arxiv.org/abs/2303.08774")
    link(title="LLaVA：Visual Instruction Tuning", url="https://arxiv.org/abs/2304.08485")
    text("**2025｜DeepSeek-R1、Qwen2.5-Omni**：分别提供强化学习提升推理能力、多模态输入与语音输出的研究实例。")
    link(title="DeepSeek-R1 技术报告", url="https://arxiv.org/abs/2501.12948")
    link(title="Qwen2.5-Omni 技术报告", url="https://arxiv.org/abs/2503.20215")
    text("今天的视频让我们看到产品的表达方式；这些公开论文帮助我们理解能力背后的技术路径。")
    text("### 这些变化依赖什么？")
    text("**数据**提供学习材料，**算力**支持更大规模训练与使用，**算法**组织表示和学习，**反馈与评测**帮助调整行为。")
    text("模型规模是因素之一；数据质量、训练目标、推理方式和系统设计同样会改变结果。")


def future_directions():
    text("## 06 · 下一步往哪里走？")
    image("images/lecture_01/future-directions.png", width=600, style={"maxWidth": "calc(100vw - 120px)"})
    text("### Thinking with image/video")
    text("### 具身智能")
    text("### 端侧模型")
    text("### 生成理解统一")
    text("### 多语言/多文化公平")


def limitations_and_risks():
    text("## 07 · 能生成之后：局限性与风险")
    text("### 可靠性：流畅、逼真，就一定正确吗？")
    text("模型可能编造文献、误译否定与数字，或生成不符合物理关系的画面；一次成功演示不能代表所有语言和场景。")
    text("外语任务尤其要核查：**原文信息是否保留、引文能否追溯、文化含义是否被改写**。评价既看表达质量，也看事实与证据。")

    text("### 版权与授权：能生成，就能使用吗？")
    text("把问题分开：**训练素材从哪里来？输入素材能否上传或改编？生成结果能否传播和商用？** 这些问题不能只靠『由 AI 生成』来回答。")
    text("回看多语言歌曲：歌词、曲谱、录音可能涉及不同权利；模仿真人声音还涉及本人授权与人格权益。标注来源不能替代所需的授权。")
    text("课程项目应记录素材来源、许可与使用范围；具体权利判断取决于适用地区、使用情境和人工创作贡献。")

    text("### 隐私：输入给模型的信息去了哪里？")
    text("访谈录音、同学照片、聊天记录与未公开译稿，都可能包含本人或他人的敏感信息。上传之前先确认授权、必要性，以及服务的数据保存和使用设置。")
    text("课堂实践：使用公开且允许使用的材料，或以虚构数据替代；去掉姓名也未必能匿名，声音、面孔与上下文仍可能识别人。端侧运行可减少上传，但仍需管理日志、同步与访问权限。")

    text("### 水印与来源标识：怎样知道内容从哪里来？")
    text("**显式标识**：让人看到『AI 生成』提示。**隐式水印**：在内容信号中嵌入可检测的标记。**来源凭证**：用签名记录内容的来源与编辑过程，例如 C2PA。")
    text("水印的挑战是兼顾质量、可检测性与抗变换能力；压缩、裁剪或重新编码可能影响检测效果，具体取决于方案。来源元数据也可能在传播中丢失。")
    text("**没有检测到水印，不代表一定是真实拍摄；验证了来源凭证，也不等于画面中的事件真实。** 标识、来源核查与事实核查需要配合使用。")
    link(title="延伸阅读：NIST 合成内容透明度技术综述（2024）", url="https://www.nist.gov/publications/reducing-risks-posed-synthetic-content-overview-technical-approaches-digital-content")
    link(title="C2PA：来源凭证能记录什么？", url="https://c2pa.org/faqs/")

    text("### 安全与诈骗：当声音和视频不再足以证明身份")
    text("声音克隆、换脸视频与多语言文本生成，可能被用于冒充亲友或同事、伪造通知和钓鱼信息。熟悉的声音、流利的措辞和看似真实的画面，都不能单独作为身份凭据。")
    text("遇到催促转账或索要验证码的信息，先暂停，使用自己已有的可信联系方式独立核实；不要只沿用对方消息提供的电话或链接。")
    link(title="真实风险案例：FTC 关于 AI 声音克隆冒充亲友的提醒（2023）", url="https://consumer.ftc.gov/consumer-alerts/2023/03/scammers-use-ai-enhance-their-family-emergency-schemes")
    text("当模型能调用工具时，还要防止它把网页或文档中的恶意文字当作指令；发消息、转账、删除文件等操作，需要权限边界与关键步骤确认。")

    text("### 回到本课程：能力之外，还要评价什么？")
    text("**内容准确吗？素材获准使用吗？个人信息受到保护吗？生成与编辑过程可追溯吗？会不会误导或伤害他人？**")
    link(title="风险全景：NIST 生成式 AI 风险管理框架（2024）", url="https://nvlpubs.nist.gov/nistpubs/ai/NIST.AI.600-1.pdf")


def course_information():
    text("**课程大纲介绍**")


if __name__ == "__main__":
    main()
