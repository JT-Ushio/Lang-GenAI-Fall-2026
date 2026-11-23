"""第八讲延伸阅读：Moshi。独立编译 python3 execute.py -m lecture_08_moshi_appendix。"""
import torch
from execute_util import text, image, link


def moshi_shape_code():
    audio_ids = torch.empty(1, 2, 8, 50, dtype=torch.long)
    text_ids = torch.empty(1, 50, dtype=torch.long)
    frame_ms = 1000 / 12.5  # @inspect frame_ms
    streams_shape = tuple(audio_ids.shape)  # @inspect streams_shape
    print('四秒、两说话流、八码本', streams_shape, '帧长 / ms', frame_ms)
    print('另行对齐的文字槽位', tuple(text_ids.shape))
    assert streams_shape == (1, 2, 8, 50) and frame_ms == 80


def moshi_architecture():
    text('## Moshi：边听边说的全双工模型')
    text('Moshi 将用户语音、系统语音分成两条同步流，并为系统语音预测对齐的文字流。允许插话、重叠和短回应。')
    text('Mimi 流式编码器将 24 kHz 语音变成每秒 12.5 帧；Moshi 使用八个码本，每个码本容量为 2048。')
    text('时间 Transformer 聚合历史，深度 Transformer 建模同一帧的码本关系。文字“内心独白”帮助组织要说的内容。')
    image('images/lecture_08/moshi-official.png', width=800, style=STYLE)
    link(title="参考资料", url='https://arxiv.org/abs/2410.00037')
    link(title="参考资料", url='https://github.com/kyutai-labs/moshi')


def moshi_streams():
    text('## 两条对话流与多个码本')
    text('中文接口示意：用户说“明天……不，改成后天”，系统可在听的同时发出“嗯”，随后修正回答。')
    text('四秒对应 50 个编码帧。教学张量 [B,2,8,50] 的轴依次为批次、说话流、码本、时间；文字另有一条对齐流。')
    text('两条说话流描述谁在说；八个码本描述每条流的编码层级。原始 Moshi 的中文能力不由这个形状演示验证。')
    moshi_shape_code()
    link(title="参考资料", url='https://arxiv.org/abs/2410.00037')

STYLE = {'maxWidth': '100%', 'height': 'auto'}

def main():
    text('# 延伸阅读 · Moshi 的全双工语音交互')
    moshi_architecture()
    moshi_streams()
