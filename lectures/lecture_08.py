"""第八讲：音乐生成——从乐谱到歌曲。备课说明见 notes/lecture_08.md。"""
import inspect
import json
from pathlib import Path
import torch
from execute_util import audio, image, link, text
from demos.lecture_08_audio import (
    delay_pattern, TinyMelodyLM, train_melody_model, generate_melody,
    render_melody, flow_matching_example, multitrack_demo,
)
from demos.lecture_08_music import (
    midi_to_hz, render_score, beat_clicks, music_features, render_chord,
)
STYLE = {'maxWidth': '100%', 'height': 'auto'}
CODE_FENCE = chr(96) * 3


def music_shape_code():
    codes = torch.empty(1, 4, 250, dtype=torch.long, device='meta')
    hidden = torch.empty(1, 253, 1024, device='meta')
    heads = torch.nn.ModuleList([
        torch.nn.Linear(1024, 2048, device='meta') for _ in range(4)])
    logits = torch.stack([head(hidden) for head in heads], dim=1)
    logits_shape = tuple(logits.shape)  # @inspect logits_shape
    print('五秒音频编号', tuple(codes.shape))
    print('四码本延迟排列长度 250+3=253，分数形状', logits_shape)
    assert logits_shape == (1, 4, 253, 2048)


def delay_code():
    codes = torch.arange(24).reshape(4, 6)
    delayed = delay_pattern(codes)  # @stepover
    delayed_values = delayed.tolist()  # @inspect delayed_values
    valid_count = int((delayed >= 0).sum())  # @inspect valid_count
    restored = torch.stack([delayed[q, q:q + 6] for q in range(4)])
    print('原始编号', codes.tolist(), '延迟排列', delayed_values)
    print('有效监督位置', valid_count)
    assert torch.equal(restored, codes) and valid_count == 24
    text(CODE_FENCE + 'python\n' + inspect.getsource(delay_pattern) + '\n' + CODE_FENCE)


def train_music_code():
    text(CODE_FENCE + 'python\n' + inspect.getsource(TinyMelodyLM) + '\n' + CODE_FENCE)
    text(CODE_FENCE + 'python\n' + inspect.getsource(train_melody_model) + '\n' + CODE_FENCE)
    model, report = train_melody_model()  # @stepover
    parameter_count = report['parameters']  # @inspect parameter_count
    loss_history = report['history']  # @inspect loss_history
    generated_notes = report['phrases']  # @inspect generated_notes
    stored = json.loads((Path(__file__).parent / 'demos/lecture_08_result.json').read_text())
    print('参数数量', parameter_count, '更新次数', report['steps'])
    print('给定首音后的交叉熵记录', loss_history)
    print('实际生成的音符编号', generated_notes)
    assert generated_notes == stored['phrases']
    assert report['history'][-1]['loss'] < report['history'][0]['loss']


def generated_listening():
    text('从宫开始前进')
    audio('videos/lecture_08/melody-forward.wav')
    text('从羽开始后退')
    audio('videos/lecture_08/melody-backward.wav')
    text(CODE_FENCE + 'python\n' + inspect.getsource(generate_melody) + '\n' + CODE_FENCE)
    text(CODE_FENCE + 'python\n' + inspect.getsource(render_melody) + '\n' + CODE_FENCE)


def official_music_listening():
    audio('videos/lecture_08/musicgen-official.wav')
    link(title='MusicGen 作者样例页', url='https://ai.honu.io/papers/musicgen/')


def continuous_shape_code():
    latent = torch.empty(1, 250, 64, device='meta')
    patches = latent.reshape(1, 125, 128)
    planning_ids = torch.empty(1, 50, dtype=torch.long, device='meta')
    continuous_shape = tuple(latent.shape)  # @inspect continuous_shape
    print('十秒连续潜变量', continuous_shape)
    print('两个时间位置合成一块', tuple(patches.shape), '5 Hz 规划码', tuple(planning_ids.shape))


def flow_code():
    noise, target, state, velocity, final = flow_matching_example()  # @stepover
    intermediate_state = state.tolist()  # @inspect intermediate_state
    target_velocity = velocity.tolist()  # @inspect target_velocity
    final_error = float((final - target).abs().max())  # @inspect final_error
    print('t=0.4 时的插值状态', intermediate_state, '直线目标速度', target_velocity)
    print('十次 Euler 更新后的误差', final_error)
    assert final_error < 1e-6
    text(CODE_FENCE + 'python\n' + inspect.getsource(flow_matching_example) + '\n' + CODE_FENCE)


def edit_score_code():
    notes = torch.tensor([0, 2, 1, 4, 3, 2, 0, 1])
    edited = notes.clone()
    edited[2] = 2
    original_notes = notes.tolist()  # @inspect original_notes
    edited_notes = edited.tolist()  # @inspect edited_notes
    changed = (notes != edited).nonzero().flatten().tolist()  # @inspect changed
    print('修改位置', changed, '商改为角')
    assert changed == [2]
    audio('videos/lecture_08/motif-80.wav')
    audio('videos/lecture_08/motif-edited.wav')


def muse_supervision_code():
    sequence = torch.tensor([7, 8, 100, 103, 109])
    is_audio = torch.tensor([0, 0, 1, 1, 1]).bool()
    inputs = sequence[:-1]
    labels = sequence[1:].clone()
    labels[~is_audio[1:]] = -100
    input_values = inputs.tolist()  # @inspect input_values
    target_values = labels.tolist()  # @inspect target_values
    print('教学编号输入', input_values, '监督标签', target_values)
    assert target_values == [-100, 100, 103, 109]
    text(CODE_FENCE + 'python\n' + inspect.getsource(muse_supervision_code) + CODE_FENCE)


def multitrack_code():
    stems, gains, mixture, no_drums = multitrack_demo()  # @stepover
    stems_shape = tuple(stems.shape)  # @inspect stems_shape
    without_drums = (torch.tensor([1., 0., 1.])[:, None] * stems).sum(0)
    assert torch.equal(without_drums, no_drums)
    codes = torch.arange(1 * 3 * 4 * 6).reshape(1, 3, 4, 6)
    interleaved = codes.permute(0, 3, 1, 2).reshape(1, 6, 12)
    restored = interleaved.reshape(1, 6, 3, 4).permute(0, 2, 3, 1)
    interleaved_shape = tuple(interleaved.shape)  # @inspect interleaved_shape
    print('分轨波形', stems_shape, '混音', tuple(mixture.shape))
    print('教学编号 [B,轨,码本,T]', tuple(codes.shape), '交错排列', interleaved_shape)
    assert torch.equal(restored, codes)
    text(CODE_FENCE + 'python\n' + inspect.getsource(multitrack_demo) + CODE_FENCE)
    text('低音轨')
    audio('videos/lecture_08/stem-bass.wav')
    text('鼓点轨')
    audio('videos/lecture_08/stem-drums.wav')
    text('旋律轨')
    audio('videos/lecture_08/stem-melody.wav')
    text('三轨混音')
    audio('videos/lecture_08/stem-mix.wav')
    text('鼓点静音')
    audio('videos/lecture_08/stem-no-drums.wav')

def music_meter_code():
    bpm = 120
    seconds_per_beat = 60 / bpm  # @inspect seconds_per_beat
    bar_seconds = {meter: meter * seconds_per_beat for meter in [3, 4]}
    print('每拍秒数', seconds_per_beat, '每小节秒数', bar_seconds)
    text('4/4 拍：两小节，共八拍')
    audio('videos/lecture_08/music-meter-4.wav')
    text('3/4 拍：两小节，共六拍')
    audio('videos/lecture_08/music-meter-3.wav')


def music_rhythm_code():
    notes, beats, starts, shifted, intervals = music_features()
    seconds = beats * 60 / 120
    onset_seconds = starts * 60 / 120
    event_lengths = seconds.tolist()  # @inspect event_lengths
    event_onsets = onset_seconds.tolist()  # @inspect event_onsets
    print('每个音符秒数', event_lengths, '起音位置', event_onsets)
    assert float(seconds.sum()) == 4
    text('等时值版本：每个音符一拍')
    audio('videos/lecture_08/music-rhythm-even.wav')
    text('改变时值版本：音高顺序与总拍数不变')
    audio('videos/lecture_08/music-rhythm-varied.wav')
    text(CODE_FENCE + 'python\n' + inspect.getsource(render_score) + CODE_FENCE)


def music_transpose_code():
    notes, beats, starts, shifted, intervals = music_features()
    semitone_steps = intervals.tolist()  # @inspect semitone_steps
    original_hz = midi_to_hz(notes)
    transposed_hz = midi_to_hz(shifted)
    ratio = transposed_hz / original_hz
    print('原音高', notes.tolist(), '移调后', shifted.tolist(), '相邻半音距离', semitone_steps)
    assert torch.equal(intervals, shifted.diff())
    assert torch.allclose(ratio, torch.full_like(ratio, 2 ** (2 / 12)))
    text('原旋律')
    audio('videos/lecture_08/music-rhythm-even.wav')
    text('整体提高两个半音，节奏不变')
    audio('videos/lecture_08/music-transposed.wav')
    text(CODE_FENCE + 'python\n' + inspect.getsource(midi_to_hz) + CODE_FENCE)


def music_chord_code():
    major = torch.tensor([60, 64, 67])
    minor = torch.tensor([60, 63, 67])
    major_intervals = (major - major[0]).tolist()  # @inspect major_intervals
    minor_intervals = (minor - minor[0]).tolist()  # @inspect minor_intervals
    print('大三和弦相对半音', major_intervals, '小三和弦相对半音', minor_intervals)
    assert major_intervals == [0, 4, 7] and minor_intervals == [0, 3, 7]
    text('C 大三和弦：C–E–G')
    audio('videos/lecture_08/music-major.wav')
    text('C 小三和弦：C–E♭–G')
    audio('videos/lecture_08/music-minor.wav')
    text(CODE_FENCE + 'python\n' + inspect.getsource(render_chord) + CODE_FENCE)


def music_timbre_code():
    text('较柔和的合成音色')
    audio('videos/lecture_08/music-timbre-soft.wav')
    text('较明亮的合成音色：只改变谐波配比')
    audio('videos/lecture_08/music-timbre-bright.wav')


def music_score_code():
    notes, beats, starts, _, _ = music_features()
    events = torch.stack([notes.float(), starts, beats], dim=1)
    score_shape = tuple(events.shape)  # @inspect score_shape
    print('每行 [MIDI 音高, 起音拍数, 持续拍数]', events.tolist())
    waveform = render_score(notes.tolist(), beats.tolist(), bpm=120)
    waveform_shape = tuple(waveform.shape)  # @inspect waveform_shape
    print('事件形状', score_shape, '16 kHz 四秒波形', waveform_shape)
    assert score_shape == (8, 3) and waveform_shape == (64000,)


def slide_01():
    text('## 一段音乐包含哪些可以控制的内容？')
    text('旋律组织音高的先后；节奏组织音符的长短与位置；和声组织同时响起的音；音色体现声音质感。')
    text('曲式把片段组织成主歌、副歌和桥段；编配决定何时出现人声、鼓、低音和其他乐器。')
    text('本课使用原创旋律“宫—商—角—徵—羽—徵—角—宫”，逐项改变音乐条件，再连接到生成模型。')


def slide_02():
    text('## 音乐产业：创作、制作、发行与消费')
    text('一首歌连接词曲创作、编曲、演唱、录音混音、版权授权、发行传播和现场演出。模型可介入多个环节。')
    text('IFPI 披露：2025 年全球录制音乐收入 317 亿美元，同比增长 6.4%；付费流媒体订阅账户用户为 8.37 亿。')
    text('中国传媒大学团队报告：2025 年中国音乐产业总规模约 5190.35 亿元，包含演出、乐器与教育等。它与全球“录制音乐收入”的范围不同。')
    link(title="参考资料", url='https://www.ifpi.org/global-music-report-2026-global-recorded-music-revenues-grow-6-4-as-record-companies-drive-innovation/')
    link(title="参考资料", url='https://www.cflac.org.cn/new_zixun/202606/t20260617_1366277.html')


def slide_03():
    text('## Suno：从歌曲生成模型到创作平台')
    text('2026-06-03，Suno 公告 D 轮融资超过 4 亿美元，投后估值为 54 亿美元。估值属于公司，不是某个模型版本。')
    text('官方 Careers 页面披露：累计超过 1 亿人曾用 Suno 创作音乐，2026 年初付费订阅者超过 200 万，年化收入运行率为 3 亿美元。')
    link(title="参考资料", url='https://suno.com/blog/series-d-announcement')
    link(title="参考资料", url='https://about.suno.com/careers')


def slide_04():
    text('## 拍、速度与拍号')
    text('拍（beat）是计量音乐时间的参照。速度 BPM 表示每分钟有多少个指定单位拍；120 BPM 下，一拍是 0.5 秒。')
    text('4/4 拍通常每小节有四个四分音符拍；3/4 拍有三个。拍号不能只靠“每秒敲几下”决定。')
    text('试听在相同速度下，四拍一组与三拍一组的重音。程序只模拟小节首拍较强的简单模式。')
    image('images/lecture_08/music-meter.png', width=800, style=STYLE)
    music_meter_code()


def slide_05():
    text('## 同一旋律可以有不同节奏')
    text('两段保持相同的八个音高与总拍数。第一段每音一拍；第二段使用半拍、一拍和两拍的时值。')
    text('时值（duration）描述音持续多久；起音位置（onset）描述音何时开始。节奏包含声音与休止的时间组织。')
    text('保持 120 BPM，两段都是八拍、四秒。总长度相同，音乐的推进与停顿仍会改变。')
    image('images/lecture_08/music-rhythm.png', width=800, style=STYLE)
    music_rhythm_code()


def slide_06():
    text('## 音高、音程、音阶与移调')
    text('在十二平均律中，一个八度分为十二个半音。MIDI 音高编号每增加 1，频率乘以 2^(1/12)；69 对应 A4 的 440 Hz。')
    text('音程描述两个音的距离。把整段旋律升高两个半音，保留相邻音程与时值，就是一次移调。')
    text('本例使用 C、D、E、G、A 五个音级；这只是便于演示的五声音阶，不代表所有音乐使用同一音阶体系。')
    image('images/lecture_08/music-transpose.png', width=800, style=STYLE)
    music_transpose_code()


def slide_07():
    text('## 和声：同时发声的音怎样组合？')
    text('旋律强调音的先后，和声关注同时响起及相继连接的音。三和弦是三个音构成的常见和声单位。')
    text('C 大三和弦为 C–E–G；C 小三和弦为 C–E♭–G。两者保持根音与五度，只改变中间的三度音。')
    text('试听这一个半音的变化。大小和弦的感受还取决于节奏、编配与文化经验，不能固定等同于快乐或悲伤。')
    image('images/lecture_08/music-chords.png', width=800, style=STYLE)
    music_chord_code()


def slide_08():
    text('## 同一旋律、同一节奏，改变音色')
    text('音色（timbre）使相同音高的钢琴、笛子与人声仍能被区分。频谱、起音、衰减和演奏方式都会影响它。')
    text('本例保持音高、时值与包络，只改变谐波比例，比较较柔和与较明亮的合成音。它们不是对真实乐器的精确模拟。')
    text('“换成古筝”不仅是改变一个频率；模型还需学习拨弦音头、共鸣、装饰音和具体演奏法。')
    music_timbre_code()


def slide_09():
    text('## 曲式：重复、变化、对比与回归')
    text('主歌（verse）常推动叙述，副歌（chorus）常承载反复出现的核心乐句，桥段（bridge）可提供对比与过渡。')
    text('A–B–A′–B–C–B′ 是一种示意结构。相同字母表示材料相关，撇号表示变化；真实歌曲没有唯一模板。')
    text('段落级控制需要让歌词、旋律、乐器密度与能量变化相互配合。这正是 Muse 分段条件和音乐规划模型的任务。')
    image('images/lecture_08/music-form.png', width=800, style=STYLE)


def slide_10():
    text('## 乐谱与录音：两种音乐表示')
    text('符号表示记录音高、起音、时值、速度和乐器等事件。钢琴卷帘图用横轴表示时间，纵轴表示音高。')
    text('音频表示记录实际声波，保留音色、演唱、空间与制作细节。相同乐谱可对应多种演奏和录音。')
    text('MIDI 是事件与控制信息的协议，不是声音文件。课堂用整数数组模拟事件，再由简单合成器生成波形。')
    image('images/lecture_08/music-rhythm.png', width=800, style=STYLE)
    music_score_code()


def slide_11():
    text('## 音乐 codec：把录音压缩成可预测的表示')
    text('音乐生成模型常先用神经音频编解码器压缩录音：编码器得到潜表示，量化器产生编号，解码器恢复声音。')
    text('多个残差码本可以从粗到细补充录音信息；它们不分别对应钢琴、鼓或人声。音轨与码本是不同轴。')
    text('MusicGen 预测多码本离散编号；ACE-Step 更新连续声学潜变量。两者都需要把表示还原为可听音乐。')
    link(title="参考资料", url='https://arxiv.org/abs/2306.05284')
    link(title="参考资料", url='https://arxiv.org/abs/2602.00744')


def slide_12():
    text('## 把音乐意图写成分层条件')
    text('全局条件：“中文校园民谣，4/4 拍，96 BPM，木吉他与轻鼓点”。歌词：“晚风经过窗前，灯光照亮书页”。')
    text('段落条件：“主歌疏朗，副歌增加和声与鼓点”。保留条件：“修改副歌歌词，保持原旋律与伴奏”。')
    text('文字只是表达意图的一种接口。精确旋律可用乐谱或哼唱，精确节奏可用事件序列，局部编配可用参考音轨。')


def slide_13():
    text('## 音乐生成模型的四种建模路线')
    text('离散编号自回归：MusicGen、Muse、SongGen、MusicGen-Stem。像续写词元一样生成音乐编码。')
    text('连续潜变量生成：早期 ACE-Step 使用流匹配；ACE-Step 1.5 进一步加入语言模型规划与 DiT 声学渲染。')
    text('符号与音频统一：YuE2 将可读乐谱、语义词元与连续声学生成连接。Suno 的内部架构不据产品演示推定。')
    link(title="参考资料", url='https://arxiv.org/abs/2306.05284')
    link(title="参考资料", url='https://arxiv.org/abs/2601.03973')
    link(title="参考资料", url='https://arxiv.org/abs/2506.00045')
    link(title="参考资料", url='https://arxiv.org/abs/2602.00744')
    link(title="参考资料", url='https://arxiv.org/abs/2609.33757')


def slide_14():
    text('## 时间线与研究团队：2023—2025')
    text('2023-06-08｜MusicGen｜Meta AI：多码本音乐语言模型，文本与旋律条件。')
    text('2025-01-03｜MusicGen-Stem｜Meta、IRCAM–CNRS–索邦等：鼓、低音和其他分轨。')
    text('2025-02-18｜SongGen｜上海人工智能实验室及北航、港中文等：歌曲与人声／伴奏双轨。')
    text('2025-05-28｜ACE-Step｜ACE Studio：连续潜变量音乐生成与编辑。')
    link(title="参考资料", url='https://arxiv.org/abs/2306.05284')
    link(title="参考资料", url='https://arxiv.org/html/2501.01757')
    link(title="参考资料", url='https://arxiv.org/html/2502.13128')
    link(title="参考资料", url='https://arxiv.org/html/2506.00045')


def slide_15():
    text('## 时间线与研究团队：2026')
    text('2026-01-07｜Muse｜复旦 NLP：中英歌曲数据、分段控制与可复现训练。')
    text('2026-01-31｜ACE-Step 1.5｜ACE Studio、StepFun：规划、声学渲染与多任务编辑。')
    text('2026-09-09｜Suno v6｜Suno：商业产品发布，多模态条件与局部编辑。')
    text('2026-09-27｜YuE2｜HKUST／HKGAI、M-A-P 等：符号乐谱与歌曲音频统一。')
    link(title="参考资料", url='https://arxiv.org/html/2601.03973')
    link(title="参考资料", url='https://arxiv.org/html/2602.00744')
    link(title="参考资料", url='https://suno.com/release-notes/introducing-v6')
    link(title="参考资料", url='https://arxiv.org/html/2609.33757')


def slide_16():
    text('## MusicGen：音频词元语言模型')
    text('音频先经 EnCodec 压缩成多条码本编号序列。文字经 T5 编码，作为生成条件。')
    text('Transformer 逐步预测音频编号，再由 EnCodec 解码器恢复音乐波形。')
    text('本课固定讲经典单声道四码本配置。其旋律条件版本还可以读取 chromagram（色度图）。')
    image('images/lecture_08/musicgen-architecture.png', width=800, style=STYLE)
    link(title="参考资料", url='https://arxiv.org/abs/2306.05284')
    link(title="参考资料", url='https://github.com/facebookresearch/audiocraft/blob/main/docs/MUSICGEN.md')


def slide_17():
    text('## MusicGen small 的维度')
    text('32 kHz 单声道的一秒波形约有 32,000 点，编码为 50 个时间位置、每处四个编号。')
    text('小型 LM 使用 24 层、1024 隐藏维、16 个注意力头。每个码本有 2048 个编号。')
    text('五秒的码本张量为 [B,4,250]，预测分数可写成 [B,4,S,2048]，S 取决于排列方式。')
    music_shape_code()
    link(title="参考资料", url='https://github.com/facebookresearch/audiocraft/blob/main/docs/MUSICGEN.md')
    link(title="参考资料", url='https://github.com/facebookresearch/audiocraft/blob/main/config/model/lm/model_scale/small.yaml')


def slide_18():
    text('## 代码实验：码本的延迟排列')
    text('四条码本流使用 [0,1,2,3] 的延迟。不同流错开后，在同一个生成步预测各自当前位置。')
    text('六个原始时间位置变成九个排列位置，填充位置用 −1 表示。')
    text('有效位置参与损失，空位置需要掩蔽。总生成步数与四条流直接串联的长度不同。')
    image('images/lecture_08/delay-pattern.png', width=800, style=STYLE)
    delay_code()
    link(title="参考资料", url='https://arxiv.org/abs/2306.05284')
    link(title="参考资料", url='https://github.com/facebookresearch/audiocraft/blob/main/config/model/lm/musicgen_lm.yaml')


def slide_19():
    text('## MusicGen 的数据与训练流程')
    text('先训练或选定音频 codec，并为训练音乐提取目标编号。')
    text('整理音乐片段与描述，抽取时间窗口，用冻结文本编码器提供条件，训练音频 LM。')
    text('训练时读取正确的历史编号，推理时接回模型已生成的编号。通过条件丢弃还可训练无条件分支。')
    link(title="参考资料", url='https://github.com/facebookresearch/audiocraft/blob/main/docs/MUSICGEN.md')
    link(title="参考资料", url='https://github.com/facebookresearch/audiocraft/blob/main/config/conditioner/text2music.yaml')


def slide_20():
    text('## 代码实验：训练一个条件旋律模型')
    text('两层、宽度 32 的因果 Transformer，词表含五个音级和起止标记。')
    text('十条五音级循环乐句，条件指定前进或后退。给定首音后，学习后续四音和结束标记。')
    text('模型训练与自由生成真实执行。数据规模用于检查流程，音符编号随后由解析合成器变成波形。')
    image('images/lecture_08/tiny-training.png', width=800, style=STYLE)
    train_music_code()


def slide_21():
    text('## 试听模型自己生成的音符')
    text('条件一为“从宫开始，按五声音阶前进”，条件二为“从羽开始，按五声音阶后退”。')
    text('生成过程中每次读取自己刚预测的音符，直到输出结束标记。')
    text('这里直接建模可读音级。完整 MusicGen 则建模含有复杂音色与录音细节的 codec 编号。')
    image('images/lecture_08/tiny-pianoroll.png', width=800, style=STYLE)
    generated_listening()


def slide_22():
    text('## MusicGen 官方生成样例')
    text('中文释义为“旋律抓耳、带热带打击乐、节奏轻快的流行舞曲，适合海滩”。')
    text('作者展示页采用英文提示。本课节选其 MusicGen 音频前八秒，模型组别为页面所述 3.3B。')
    text('听旋律、节奏和音色的共同出现，再回到编号预测与解码的结构。')
    official_music_listening()
    link(title="参考资料", url='https://ai.honu.io/papers/musicgen/')


def slide_23():
    text('## ACE-Step 1.5：规划与声学生成')
    text('语言模型组织歌词、描述和音乐信息，并可生成低频率离散音乐码。')
    text('条件进入 DiT（Diffusion Transformer，扩散 Transformer），在连续潜空间生成声学细节。')
    text('1D VAE 将潜变量解码为 48 kHz 立体声。本课对应 1.5 报告的基础架构。')
    image('images/lecture_08/ace15-official.png', width=800, style=STYLE)
    link(title="参考资料", url='https://arxiv.org/abs/2602.00744')


def slide_24():
    text('## ACE-Step 1.5 的时间尺度')
    text('报告中的 VAE 每秒用 25 个 64 维潜向量表示音频。十秒对应 [B,250,64]。')
    text('规划侧量化码为 5 Hz，十秒约 50 个编号。DiT patchify 还会将潜序列长度减半。')
    text('低频码提供结构条件，连续潜变量保留更细的声学信息。')
    continuous_shape_code()
    link(title="参考资料", url='https://arxiv.org/abs/2602.00744')


def slide_25():
    text('## 代码实验：流匹配的训练目标')
    text('采样噪声 z₀ 与目标 z₁，构造 zₜ=(1−t)z₀+t z₁，直线路径的目标速度是 z₁−z₀。')
    text('网络学习根据当前状态、时间和条件预测速度。积分得到生成轨迹。')
    text('小例子用已知端点验证直线路径与 Euler 更新，便于理解目标的来源。')
    image('images/lecture_08/flow-path.png', width=800, style=STYLE)
    flow_code()
    link(title="参考资料", url='https://arxiv.org/abs/2210.02747')


def slide_26():
    text('## ACE-Step 的数据加工与训练阶段')
    text('构建音乐—描述样本，增加歌词与语言标注，并过滤内容、标注和音频之间的不一致。')
    text('训练阶段从基础文生音乐扩展到带来源与遮罩的多任务，再进行高质量微调和蒸馏。')
    text('歌声与伴奏分离等加工可以产生更多训练条件。各阶段的具体规则需结合版本代码核对。')
    link(title="参考资料", url='https://arxiv.org/abs/2602.00744')


def slide_27():
    text('## 中文歌词与旋律的对齐')
    text('原创短句“晚风经过窗前，灯光照亮书页”可以分成两句，各安排若干音符。')
    text('一个字可以跨多个音符，也可以在一个音符内快速带过多个字，时长需要结合节奏安排。')
    text('汉语字调、旋律走向、押韵与句末落音会互相影响，需要同时听内容和音乐组织。')


def slide_28():
    text('## YuE2：可读乐谱参与歌曲生成')
    text('条件包括文字风格和歌词。模型先产生可编辑的符号乐谱，再产生语义词元和连续声学潜变量。')
    text('乐谱采用 ABC 文本形式，表达调性、节拍、速度、旋律与和弦。')
    text('AR–NAR MoT 将自回归离散预测和非自回归声学生成放入同一模型框架。')
    image('images/lecture_08/yue2-official.png', width=800, style=STYLE)
    link(title="参考资料", url='https://arxiv.org/abs/2609.33757')


def slide_29():
    text('## YuE2 的训练材料与信息流')
    text('录音经转写与音乐理解工具产生符号和语义监督，再与声学目标对齐。')
    text('自回归流预测乐谱和语义词元，连续流学习声学潜变量的更新。两流共享注意力交互，并保留专用参数。')
    text('可读乐谱让修改落到具体旋律或和弦，再生成新的录音。')
    link(title="参考资料", url='https://arxiv.org/abs/2609.33757')


def slide_30():
    text('## 代码实验：把音乐反馈变成符号修改')
    text('给定一段原创音级序列，只把第三个音从“商”改为“角”，其余音级与时长保持。')
    text('先检查编号，再听修改前后的合成旋律。')
    text('本例直接编辑符号数组，演示可检查的控制接口。完整歌曲模型还要学习如何把计划实现为声音。')
    edit_score_code()


def slide_31():
    text('## Muse：复旦的分段歌曲生成模型')
    text('Muse 在 Qwen3 上扩充 MuCodec 音频词表，用同一自回归序列连接风格、歌词与音频编号。')
    text('MuCodec 的音频码本容量为 16,384。模型逐段预测音频编号，再用 codec 解码为混合人声与伴奏的波形。')
    text('技术报告主表使用 Muse-0.6B；仓库还提供其他规模。公开推理分为 vLLM 生成编号与 MuCodec 解码两步。')
    image('images/lecture_08/muse-official.png', width=800, style=STYLE)
    link(title="参考资料", url='https://arxiv.org/abs/2601.03973')
    link(title="参考资料", url='https://github.com/yuhui1038/Muse')
    link(title="参考资料", url='https://github.com/yuhui1038/Muse/tree/main/infer')


def slide_32():
    text('## Muse 的数据怎样构建？')
    text('作者构建 116,489 首中英合成歌曲：文字模型拟定风格与分段歌词，Suno V5 生成完整歌曲。')
    text('再由音频理解模型核查并重写风格描述，补充段落级标注；生成提示与实际听感可能存在差异。')
    text('训练将分段条件与目标音频编号配对，采用因果语言建模交叉熵。数据、权重及训练评测流程均公开，使用遵循发布许可。')
    image('images/lecture_08/muse-data-official.png', width=800, style=STYLE)
    link(title="参考资料", url='https://arxiv.org/abs/2601.03973')
    link(title="参考资料", url='https://github.com/yuhui1038/Muse')


def slide_33():
    text('## 分段控制与下一词元监督')
    text('原创中文条件：全曲为轻柔校园民谣；主歌“晚风经过窗前”配木吉他；副歌“灯光照亮书页”加入鼓点。')
    text('每轮给出段落风格与歌词，随后预测该段音频；此前音频进入上下文，帮助段落衔接。')
    text('代码用教学编号演示标签右移与条件掩蔽；真实输入格式、音素提示和特殊标记应按官方样例构造。')
    muse_supervision_code()
    link(title="参考资料", url='https://github.com/yuhui1038/Muse/blob/main/infer/test.jsonl')
    link(title="参考资料", url='https://github.com/yuhui1038/Muse/tree/main/train')


def slide_34():
    text('## ACE-Step 与 Muse：先核对版本和评测')
    text('Muse 论文 Table 1 中，0.6B 的 Muse 在这三项指标上优于早期 ACE-Step 3.5B；Muse 还获得分段风格条件，基线仅用全局条件。')
    text('该基线引用 2025 年 ACE-Step 论文，不是 ACE-Step 1.5。1.5 报告没有 Muse，测试样本与条件也不同，无法从两篇表格得出双方胜负。')
    text('PER 检查歌词音素错误；MuLan-T 检查文字与音乐的关联；PQ 估计制作质量。数值均为作者报告，未在本课复测。')
    text('| Muse 论文同一评测表 | PER ↓ | MuLan-T ↑ | 制作质量 PQ ↑ |\n|---|---|---|---|\n| ACE-Step 3.5B（早期） | 0.39 | 0.28 | 7.31 |\n| Muse 0.6B | 0.16 | 0.33 | 8.14 |')
    link(title="参考资料", url='https://arxiv.org/abs/2601.03973')
    link(title="参考资料", url='https://arxiv.org/abs/2506.00045')
    link(title="参考资料", url='https://arxiv.org/abs/2602.00744')


def slide_35():
    text('## Suno：产品能力与公开技术证据')
    text('Suno 在 2026-09-09 发布 v6，官方介绍文字、音频、图像和视频条件，以及局部改词、改编与多来源创作。')
    text('截至 2026-11-16，本课未检索到可核验的 Suno v5／v6 完整公开架构与训练技术报告；不把第三方推测画成确定的网络结构。')
    text('Suno 团队早期开放的 Bark 是 Transformer 文本到音频模型，支持语音及简单音乐音效；Bark 的实现不能视为当前 Suno 歌曲模型的架构。')
    link(title="参考资料", url='https://suno.com/release-notes/introducing-v6')
    link(title="参考资料", url='https://github.com/suno-ai/bark')


def slide_36():
    text('## 声道、音轨与码本是三个维度')
    text('声道（channel）描述播放布局，如左、右声道；音轨或分轨（track／stem）描述可独立编辑的声音来源，如人声、鼓和低音。')
    text('一条鼓轨也可以是立体声；一个单声道混音也可用多个 RVQ 码本编码。多码本不等于多音轨。')
    text('波形可写成 [批次,音轨,声道,采样点]，编码可写成 [批次,音轨,码本,时间帧]。具体轴顺序以实现为准。')
    link(title="参考资料", url='https://arxiv.org/abs/2502.13128')
    link(title="参考资料", url='https://arxiv.org/abs/2501.01757')


def slide_37():
    text('## SongGen：人声与伴奏的联合建模')
    text('SongGen 提供混合输出与双轨输出两种模式。双轨版本分别编码人声和伴奏，让同一个自回归模型学习两轨的依赖。')
    text('公开 Interleaving A–V 权重按伴奏与人声流交错建模，解码得到两段同步波形，可以分别调节增益后相加。')
    text('同一时间的演唱、节奏与和声需要协调；独立随机生成两首声音再叠加，不会自动学到这种关系。')
    image('images/lecture_08/songgen-official.png', width=800, style=STYLE)
    link(title="参考资料", url='https://arxiv.org/abs/2502.13128')
    link(title="参考资料", url='https://github.com/LiuZH-19/SongGen')


def slide_38():
    text('## 分轨生成、条件补轨与音源分离')
    text('MusicGen-Stem 研究低音、鼓、其他三条分轨：各轨先编码为离散编号，模型联合建模，并能依据保留轨生成待修改轨。')
    text('中文条件示例：“保留鼓与低音，把其他轨改成钢琴”。SongGen 的人声／伴奏双轨是另一种轨道粒度。')
    text('音源分离从已有混音估计各轨，可能有串音；原生分轨生成直接建模目标轨。Suno 的分轨导出功能本身不能证明内部原生联合建模。')
    link(title="参考资料", url='https://arxiv.org/abs/2501.01757')
    link(title="参考资料", url='https://simonrouard.github.io/musicgenstem/')
    link(title="参考资料", url='https://github.com/LiuZH-19/SongGen')


def slide_39():
    text('## 代码与试听：独立分轨怎样混音？')
    text('原创解析合成的低音、鼓点、旋律各四秒，组成 [3,64000] 张量。保持同一采样率与时间轴，用增益加权求和。')
    text('依次听三条独立音轨、完整混音、鼓点静音后的版本。五个文件共用振幅标尺，便于比较。')
    text('代码还将 [B,音轨,码本,T] 转成按时间交错的流并精确还原，解释接口；这里没有训练分轨生成模型。')
    image('images/lecture_08/multitrack.png', width=800, style=STYLE)
    multitrack_code()
    link(title="参考资料", url='https://github.com/LiuZH-19/SongGen')


def slide_40():
    text('## 音乐模型的公开材料与许可')
    text('MusicGen 提供推理与训练代码、权重和论文。代码 MIT，权重 CC BY-NC 4.0。')
    text('ACE-Step 1.5 提供代码、权重和技术报告，项目按 MIT 发布，具体组件另看其说明。')
    text('YuE2 提供代码、权重和报告。代码 Apache-2.0，权重采用 CC BY-NC 4.0 并附个人创作等额外许可。 Muse 提供数据与训练评测流程，使用前分别核对模型和数据许可。')
    link(title="参考资料", url='https://github.com/facebookresearch/audiocraft/blob/main/model_cards/MUSICGEN_MODEL_CARD.md')
    link(title="参考资料", url='https://github.com/ace-step/ACE-Step-1.5')
    link(title="参考资料", url='https://github.com/multimodal-art-projection/YuE/blob/main/MODEL_LICENSE')
    link(title="参考资料", url='https://github.com/yuhui1038/Muse')


def slide_41():
    text('## 生成质量与条件控制的评价')
    text('先固定提示、时长和采样预算，分别比较音质、旋律与节奏、歌词清晰度、结构和条件符合程度。')
    text('**FAD（Fréchet Audio Distance，弗雷歇音频距离）** 比较两组音频嵌入分布，需要足够样本与明确的特征模型。')
    text('**CLAP（Contrastive Language-Audio Pretraining，语言—音频对比预训练）** 相似度可辅助检查文字关联，主观听评还需覆盖音乐性。')
    link(title="参考资料", url='https://arxiv.org/abs/2306.05284')
    link(title="参考资料", url='https://arxiv.org/abs/2602.00744')
    link(title="参考资料", url='https://arxiv.org/abs/2609.33757')


def slide_42():
    text('## 声音来源、授权与创作记录')
    text('参考声音需要有明确来源和使用权限，录音也可能包含身份线索。')
    text('复现时记录模型版本、输入文字、歌词、参考音频、随机种子与采样设置。')
    text('公开代码、模型权重与训练材料分别提供不同程度的可复现性，课程报告应逐项说明。')


def slide_43():
    text('## 回顾与阅读')
    text('音乐生成需要同时组织旋律、节奏、和声、音色与曲式，并协调歌词、人声和伴奏。')
    text('音符事件适合精确编辑，录音编码保留演奏细节，分轨表示支持独立控制与混音。')
    text('比较模型时，区分建模路线、条件输入、输出轨道、模型版本与评测协议。')
    link(title="参考资料", url='https://arxiv.org/abs/2412.10117')
    link(title="参考资料", url='https://arxiv.org/abs/2306.05284')
    link(title="参考资料", url='https://arxiv.org/abs/2602.00744')
    link(title="参考资料", url='https://arxiv.org/abs/2609.33757')
    link(title="参考资料", url='https://arxiv.org/abs/2410.00037')
    link(title="参考资料", url='https://arxiv.org/abs/2601.03973')
    link(title="参考资料", url='https://arxiv.org/abs/2502.13128')


def main():
    text("# 第八讲 · 音乐生成：从乐谱到歌曲")
    text("节奏、旋律、和声、曲式与音乐生成模型")
    slide_01()
    slide_02()
    slide_03()
    slide_04()
    slide_05()
    slide_06()
    slide_07()
    slide_08()
    slide_09()
    slide_10()
    slide_11()
    slide_12()
    slide_13()
    slide_14()
    slide_15()
    slide_16()
    slide_17()
    slide_18()
    slide_19()
    slide_20()
    slide_21()
    slide_22()
    slide_23()
    slide_24()
    slide_25()
    slide_26()
    slide_27()
    slide_28()
    slide_29()
    slide_30()
    slide_31()
    slide_32()
    slide_33()
    slide_34()
    slide_35()
    slide_36()
    slide_37()
    slide_38()
    slide_39()
    slide_40()
    slide_41()
    slide_42()
    slide_43()
