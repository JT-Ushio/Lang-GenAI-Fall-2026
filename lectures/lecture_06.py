"""第六讲：多模态理解与生成统一，以及视觉智能体。"""
import json
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F
from execute_util import image,interactive,link,text
from demos.lecture_06_unified import (
    IDS,TOKENS,COLORS,scene,image_tokens,make_example,pack,
    loss_for,block_attention,caption_image,edit_image,build_experiments,
)
from demos.lecture_06_unified_assets import prepare_assets

FIGURE_STYLE={"maxWidth":"min(100%, calc(100vw - 125px))","height":"auto"}
LAB_STYLE={"maxWidth":"min(100%, calc(100vw - 125px))"}
ASSETS='images/lecture_06/unified/'


def main():
    text('# 第六讲 · 多模态理解与生成统一')
    experiments=build_experiments()  # @stepover
    prepare_assets(experiments)  # @stepover
    from_conditioning_to_unification()
    tasks_as_sequences()
    visual_representations()
    janus_case()
    showo_case()
    bagel_case()
    joint_training(experiments)
    one_model_many_tasks(experiments)
    language_and_context()
    model_and_agent()
    feedback_loop(experiments)
    evaluation()


def from_conditioning_to_unification():
    text('## 01 · 从“按提示画图”，到“理解之后再生成”')
    image(ASSETS+'from-conditioning.svg',width=760,style=FIGURE_STYLE)
    text('**文字条件生成**：文字经过编码，影响图像采样。这条生成管线本身不等于能围绕原图开展问答、规划和多轮修改的系统。')
    text('**多模态理解与生成统一**：在同一模型框架中联合学习图文理解与视觉生成，让图像既能被读取，也能成为输出，并让两种能力利用共同上下文。')
    text('**图像编辑**：理解原图、识别修改对象、执行变化并维持其他约束，是统一能力的重要应用。')
    text('例：先说明海报里有哪些元素，再把标题对应的背景改亮；继续提出“保留这个主体，改成水彩风格”。后一句依赖前面的图像和指称。')
    text('上一讲的文本编码器已经表示语义；这里新增的是理解、生成、上下文与任务流程怎样共同组织。')


def tasks_as_sequences():
    text('## 02 · 同一上下文，选择回答、生成或编辑')
    image(ASSETS+'interleaved-sequence.svg',width=760,style=FIGURE_STYLE)
    text('**图文交错（interleaved）**：文字与图像片段按顺序交替出现。图像可以是用户提供的证据，也可以是模型刚生成、随后继续讨论的结果。')
    text('| 条件输入 | 待预测输出 | 任务 |\n|---|---|---|\n| 图像＋问题 | 文字 | 描述、问答与解释 |\n| 文字描述 | 图像 | 文生图 |\n| 原图＋编辑要求 | 新图 | 图像编辑 |\n| 多轮文字与图像 | 后续文字／图像 | 交错生成与多轮修改 |')
    text('**模态边界标记**：区分文字区、图像区和任务类型。序列里虽然都是数值表示，但这些位置具有不同作用。')
    text('**共享上下文**：当前输出可利用前面的语言与视觉信息；如何读取历史还由注意力规则和上下文长度决定。')
    text('统一可以指共享主干、联合训练与统一接口；不要求所有模态使用完全相同的编码器、输出头或损失。')
    interactive(ASSETS+'index.html',height=830,fragment='sequence',style=LAB_STYLE)


def visual_representations():
    text('## 03 · 理解图像的表示，与生成图像的表示')
    image(ASSETS+'visual-representations.svg',width=760,style=FIGURE_STYLE)
    text('**语义视觉特征**：把对象、属性和关系交给语言模型，适合“画面里有什么”的问题。')
    text('**离散图像词元**：把图像压缩成一组视觉编号。模型预测编号，再由图像解码器还原画面；编号不等于一个汉字或一个物体。')
    text('**连续图像潜变量**：保留连续数值表示，通过流匹配或去噪过程生成；不必把每个位置变成离散词表类别。')
    image(ASSETS+'palette-tokens.svg',width=760,style=FIGURE_STYLE)
    text('### 代码：图像数组 → 视觉编号')
    palette_code()
    text('这里的调色板固定且可逆。真实视觉量化器通常需要训练，能把较大的图像块编码成视觉词元。')
    image(ASSETS+'three-routes.svg',width=760,style=FIGURE_STYLE)


def palette_code():
    pixels=scene(left=0,right=2)
    tokens=image_tokens(pixels)  # @stepover
    pixel_shape=str(tuple(pixels.shape))  # @inspect pixel_shape
    first_row_tokens=tokens[1:5]  # @inspect first_row_tokens
    token_count=len(tokens)  # @inspect token_count
    print('像素编号：\n',pixels)
    print('图像序列：',tokens)
    print('16 个像素编号＋2 个边界标记；红色像素词元 ID=',IDS['像素_红'])
    assert tokens[1:17]==[14+int(x) for x in pixels.flatten()]


def janus_case():
    text('## 04 · Janus：视觉编码解耦，核心模型统一')
    image(ASSETS+'janus-architecture.svg',width=760,style=FIGURE_STYLE)
    text('**Janus／Janus-Pro**：理解路径使用语义视觉编码，生成路径使用可还原图像的视觉量化表示；两路进入共同的 Transformer。')
    text('**解耦（decoupling）**：不强迫一种视觉编码同时满足识别语义与还原细节的要求。统一发生在核心序列处理，而不是删除所有专用模块。')
    text('Janus-Pro-7B 的公开配置中，理解特征由 1024 维投影到 4096 维；图像输出头预测 16384 个视觉编号之一。经典 384×384 生成示例包含 576 个编号。')
    text('### 代码：理解投影与视觉输出头的接口')
    janus_shape_code()
    text('**自回归图像生成**：每次产生一个视觉编号，把它的嵌入接回上下文，继续预测，最后解码整张图。')
    text('联合支持理解和文生图，不表示该版本自动支持任意编辑指令；任务能力还要看训练数据和具体接口。')
    link(title='Janus：解耦视觉编码的统一模型',url='https://arxiv.org/abs/2410.13848')
    link(title='Janus 系列：作者实现与 Janus-Pro 配置',url='https://github.com/deepseek-ai/Janus')


def janus_shape_code():
    config=json.loads((Path(__file__).parent/'demos/lecture_06_configs/janus_pro_7b.json').read_text())
    source_width=config['aligner_config']['params']['input_dim']
    hidden_width=config['aligner_config']['params']['n_embed']
    codebook=config['gen_head_config']['params']['image_token_size']
    semantic_features=torch.empty(1,576,source_width,device='meta')
    projected=nn.Linear(source_width,hidden_width,device='meta')(semantic_features)
    last_state=torch.empty(1,hidden_width,device='meta')
    image_logits=nn.Linear(hidden_width,codebook,device='meta')(last_state)
    understanding_shape=str(tuple(projected.shape))  # @inspect understanding_shape
    generation_shape=str(tuple(image_logits.shape))  # @inspect generation_shape
    print('理解路径：',tuple(semantic_features.shape),'→',understanding_shape)
    print('一步视觉预测：',tuple(last_state.shape),'→',generation_shape)
    text('meta 张量验证接口形状，不执行 Janus 预训练权重。')


def showo_case():
    text('## 05 · Show-o：统一不等于全部逐词生成')
    image(ASSETS+'showo-architecture.svg',width=760,style=FIGURE_STYLE)
    text('**文字输出**：自回归地预测下一个文字词元。')
    text('**视觉输出**：对被遮盖的离散图像词元进行预测，迭代恢复图像；同一个核心模型采用不同的建模顺序。')
    text('**混合注意力**：文字仍遵守因果顺序，视觉块内部可采用双向注意力。未来答案必须被屏蔽，否则训练会偷看标准答案。')
    image(ASSETS+'attention-mask.svg',width=760,style=FIGURE_STYLE)
    attention_code()
    text('右图只演示已知条件块的分块读取规则；下面的微型自回归模型使用完全因果掩码。不同路线不能直接互换注意力和采样方式。')
    link(title='Show-o：统一自回归与离散扩散',url='https://arxiv.org/abs/2408.12528')


def attention_code():
    allowed=block_attention([0,0,1,1,2,2])  # @stepover
    image_reads_image=bool(allowed[2,3])  # @inspect image_reads_image
    earlier_reads_future=bool(allowed[0,4])  # @inspect earlier_reads_future
    print('T1 T2 | I1 I2 | T3 T4；行读取列：\n',allowed.int())
    assert allowed[2,3] and allowed[3,2]
    assert not allowed[0,4] and not allowed[4,5]


def bagel_case():
    text('## 06 · BAGEL：理解与连续视觉生成进入共同上下文')
    image(ASSETS+'bagel-architecture.svg',width=760,style=FIGURE_STYLE)
    text('**两类视觉信息**：ViT 提供语义特征，VAE 提供更接近像素细节的连续潜变量。原图的对象信息与视觉细节可以共同参与后续生成。')
    text('**MoT（Mixture of Transformer Experts，Transformer 专家混合）**：保持共同上下文与注意力交互，同时让部分参数按模态分工。不能把 MoT 理解成所有词元都经过完全相同的权重。')
    text('**Rectified Flow（整流流）**：在连续潜空间学习图像生成的更新方向；文字仍以自回归方式产生。')
    text('### 代码：连续潜变量怎样变成视觉位置')
    latent_patch_code()
    text('**先理解再生成**：模型可以结合图像与语言要求产生说明和视觉结果，支持编辑及交错生成。输出的说明有助于交流，但不能代替对图像结果的检查。')
    text('BAGEL 的完整注意力还区分干净、加噪与语义视觉块，防止目标信息泄漏。这里的结构图省略了完整训练日程与缓存细节。')
    link(title='BAGEL：统一多模态预训练与组合能力',url='https://arxiv.org/abs/2505.14683')
    link(title='BAGEL：作者实现、编辑与交错生成案例',url='https://github.com/ByteDance-Seed/Bagel')


def latent_patch_code():
    latent=torch.empty(1,16,64,64,device='meta')
    patches=latent.unfold(2,2,2).unfold(3,2,2)
    sequence=patches.permute(0,2,3,1,4,5).reshape(1,1024,64)
    latent_shape=str(tuple(latent.shape))  # @inspect latent_shape
    visual_sequence_shape=str(tuple(sequence.shape))  # @inspect visual_sequence_shape
    print('512×512 图像的 VAE 潜变量示例：',latent_shape)
    print('2×2 潜块：16×2×2=64 个数；位置数 32×32=1024：',visual_sequence_shape)


def joint_training(data):
    text('## 07 · 联合训练：让两类任务更新同一套参数')
    image(ASSETS+'joint-training.svg',width=760,style=FIGURE_STYLE)
    text('**理解数据**：给图像与问题，监督答案文字。')
    text('**生成数据**：给文字描述，监督视觉编号，或连续视觉更新方向。')
    text('**编辑与交错数据**：提供前后图像、修改要求和连续上下文，学习哪些内容变化、哪些内容延续。只有互不相关的图文对，不足以直接保证多轮编辑能力。')
    text('### 代码：同一个 Transformer，三种序列')
    shared_forward_code(data)
    text('### 代码：共享权重确实接收两类梯度')
    shared_gradient_code(data)
    text('**损失加权与任务配比**：理解、生成和编辑都要有足够训练信号；联合优化也可能产生任务竞争，需要分别检查各项能力。')
    image(ASSETS+'training-results.svg',width=760,style=FIGURE_STYLE)
    text('微型模型为两层、宽度 48 的 Transformer，使用固定四颜色图像和有限词表。72 个样本均参与训练；图中逐步生成分数检验拟合，不是未见样本泛化。')


def shared_forward_code(data):
    model=data['_model']
    rows=[make_example('理解',0,2),make_example('生成',0,2),make_example('编辑',0,2,0,1)]  # @stepover
    inputs,labels=pack(rows)  # @stepover
    logits=model(inputs)  # @stepover
    input_shape=str(tuple(inputs.shape))  # @inspect input_shape
    prediction_shape=str(tuple(logits.shape))  # @inspect prediction_shape
    supervised_counts=(labels!=-100).sum(1).tolist()  # @inspect supervised_counts
    print('输入：',input_shape,'；输出：',prediction_shape)
    print('理解／生成／编辑监督词元数：',supervised_counts)
    loss=F.cross_entropy(logits.reshape(-1,len(TOKENS)),labels.reshape(-1),ignore_index=-100)
    print('联合序列交叉熵：',float(loss.detach()))
    text('答案区参与损失；条件前缀与补齐位置设为 -100。训练时看标准前缀，推理时必须接回自己生成的词元。')


def shared_gradient_code(data):
    model=data['_model']
    understand_loss=loss_for(model,[make_example('理解',0,2)])  # @stepover
    generate_loss=loss_for(model,[make_example('生成',0,2)])  # @stepover
    shared_weight=model.backbone.layers[0].linear1.weight
    gradient_u=torch.autograd.grad(understand_loss,shared_weight,retain_graph=True)[0]
    gradient_g=torch.autograd.grad(generate_loss,shared_weight)[0]
    understanding_grad=round(float(gradient_u.norm()),6)  # @inspect understanding_grad
    generation_grad=round(float(gradient_g.norm()),6)  # @inspect generation_grad
    print('同一权重矩阵的梯度范数：',understanding_grad,generation_grad)
    assert gradient_u.norm()>0 and gradient_g.norm()>0


def one_model_many_tasks(data):
    text('## 08 · 同一模型的理解、生成与编辑输出')
    image(ASSETS+'learned-tasks.svg',width=760,style=FIGURE_STYLE)
    text('**理解**：读取左红右蓝的图像，输出左右色块的颜色。')
    text('**生成**：读取“左绿、右蓝”的固定词元描述，逐步生成图像编号。')
    text('**编辑**：读取原图和“左、绿”的固定指令词元，生成新图；需要读取原图右侧颜色，才能保留它。')
    learned_output_code(data)
    text('这里的选项直接映射到固定词元，不包含开放中文分词、真实照片语义或自由表达理解。图像输出来自训练后网络，没有用正确目标覆盖模型输出。')
    interactive(ASSETS+'index.html',height=860,fragment='model',style=LAB_STYLE)


def learned_output_code(data):
    example=next(r for r in data['records'] if r['task']=='编辑' and r['left']==0 and r['right']==2 and r['side']==0 and r['color']==1)
    generated_first_row=example['prediction'][1:5]  # @inspect generated_first_row
    exact_match=example['correct']  # @inspect exact_match
    print('模型生成的图像编号：\n',torch.tensor(example['predicted_image']))
    print('与目标序列精确匹配：',exact_match)
    print('各任务的训练样本自由生成结果：',data['report']['training_set_free_running'])


def language_and_context():
    text('## 09 · 语言把视觉上下文变成可执行的任务')
    image(ASSETS+'language-context.svg',width=760,style=FIGURE_STYLE)
    text('**指称与共指**：“把它也改成这种风格”依赖前文中的对象和参考图。模型需要把语言指称绑定到具体视觉内容。')
    text('**变化与不变**：“把左边改绿，右边保持蓝色”包含目标和保留约束。理解指令之后，还要检验视觉输出是否满足这两项要求。')
    text('**预设**：要求“换掉红色汽车”之前，应确认原图中确实存在该对象；目标不存在时，不能悄悄改别的对象来交差。')
    text('**跨语言等值**：把 left 翻成 right 会改变任务；表面流畅不等于指令等值。多语言比较要固定图像、对象、约束和参数。')
    text('**上下文记忆**：原图、最新图、参考图和修改历史各有角色。只保留最后一张图，可能丢失“其余保持原样”所需要的原始证据。')


def model_and_agent():
    text('## 10 · 从统一模型到多模态智能体')
    image(ASSETS+'model-and-agent.svg',width=760,style=FIGURE_STYLE)
    text('**智能体（agent）**：围绕目标维护状态，选择并执行动作，读取环境返回的观察，再决定下一步或停止。')
    text('**统一模型**提供看图、回答、生成等能力；**智能体系统**还包含规划策略、工具接口、状态管理、结果验收和预算。')
    text('| 组成方式 | 模型是否统一 | 是否形成智能体闭环 |\n|---|---|---|\n| 统一模型只生成一次图像 | 可以统一 | 未必有反馈循环 |\n| VLM 规划＋外部图像工具＋结果检查 | 模型可以分离 | 可以形成闭环 |\n| 统一模型承担观察与编辑＋状态与验收 | 理解与生成可统一 | 可以形成闭环 |')
    image(ASSETS+'agent-loop.svg',width=760,style=FIGURE_STYLE)
    text('**动作（action）**：可表示为结构化参数，例如 `{tool: edit, object: left, color: green}`。文字计划需要通过实际执行才能改变图像或环境。')
    text('**观察（observation）**：应来自新图、工具返回或环境状态，而不是把“我已经改好了”当成完成证据。')
    text('**停止条件**：目标与保留约束通过验收，或出现工具错误、无进展、预算耗尽等情况。')
    link(title='ReAct：交替组织推理、动作与环境观察',url='https://arxiv.org/abs/2210.03629')


def feedback_loop(data):
    text('## 11 · 可执行闭环：改错对象以后怎样纠正')
    text('目标：把左侧红色块改绿，右侧保持最初的蓝色。状态包括原图、当前图、目标和剩余动作次数。')
    text('规划策略和像素验收采用明确规则；看图回答与编辑都调用同一个训练后 Transformer。第一次把工具目标左右交换，用来演示反馈，不表示模型的自然错误率。')
    agent_code(data)
    image(ASSETS+'agent-trajectory.svg',width=760,style=FIGURE_STYLE)
    text('**第一步**：工具实际改了右侧；重新观察发现，左侧没完成，右侧保留约束也被破坏。')
    text('**第二步**：根据原图记录恢复右侧蓝色。')
    text('**第三步**：把左侧改绿，独立读取图像像素，确认目标和保留条件同时成立。')
    text('本例的历史保存在程序状态中，每次模型读取当前图与固定任务词元；真实多轮统一模型还可以直接读取交错图文历史。')
    interactive(ASSETS+'index.html',height=930,fragment='agent',style=LAB_STYLE)


def agent_code(data):
    model=data['_model']
    original=scene(0,2)
    current=original.clone()
    goal=(1,2)
    for turn in range(3):
        observed,_=caption_image(model,current)  # @stepover
        observed_colors='左'+COLORS[observed[0]]+'、右'+COLORS[observed[1]]  # @inspect observed_colors
        side,color=(1,goal[1]) if observed[1]!=goal[1] else (0,goal[0])
        executed_side=1-side if turn==0 else side
        action='改'+('左' if executed_side==0 else '右')+'为'+COLORS[color]  # @inspect action
        current,_=edit_image(model,current,executed_side,color)  # @stepover
        target_ok=torch.equal(current[1:3,:2],scene(*goal)[1:3,:2])
        preserve_ok=torch.equal(current[:,2:],original[:,2:])  # @inspect preserve_ok
        verified=torch.equal(current,scene(*goal))  # @inspect verified
        print(f'观察：{observed_colors}；实际执行：{action}；目标={target_ok}，保留={preserve_ok}')
        if verified:break
    assert verified
    short=data['episodes']['short']
    print('只有两步预算：',short['status'])
    assert not short['success']


def evaluation():
    text('## 12 · 理解正确、生成正确与任务完成')
    image(ASSETS+'unified-evaluation.svg',width=760,style=FIGURE_STYLE)
    text('**理解能力**：图像中的对象、属性、关系是否读对；参考图和原图有没有混淆。')
    text('**生成与编辑能力**：是否满足语言要求、是否改对对象、是否保留必要内容，画面是否自然。')
    text('**联合一致性**：模型说“右侧保持不变”，图像中右侧却变色，是跨输出不一致；文字解释正确不能抵消图像错误。')
    text('**智能体任务成功率**：检查真实结果、调用次数、失败恢复、耗时与成本；预算耗尽必须报告未完成。')
    text('**验收独立性**：同一模型可能反复确认自己的错误。这里用精确像素规则核查；真实开放场景可用外部工具、可靠标注或人工检查。')
    text('统一模型让语言与视觉在共同上下文中协作；智能体循环进一步把生成结果变成新观察，让下一步动作由反馈决定。')


if __name__=='__main__':main()
