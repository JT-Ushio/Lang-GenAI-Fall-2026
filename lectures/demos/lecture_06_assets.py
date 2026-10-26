"""Original SVG figures and actual numerical outputs for lecture 06."""
import base64
from html import escape
import io
import json
from pathlib import Path
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "images/lecture_06"


def txt(x, y, value, size=22, color="#17334a", weight=400):
    content = escape(str(value))
    for glyph, letter in {'ₜ':'t', '₀':'0', '₁':'1', '₋':'−', 'ᵤ':'u'}.items():
        content = content.replace(glyph, f'<tspan baseline-shift="sub" font-size="70%">{letter}</tspan>')
    return f'<text x="{x}" y="{y}" font-size="{size}" fill="{color}" font-weight="{weight}">{content}</text>'


def box(x, y, width, height, lines, fill="#e2f1f0", size=21):
    return (f'<rect x="{x}" y="{y}" width="{width}" height="{height}" rx="13" fill="{fill}" stroke="#9ab9c1"/>' +
            ''.join(txt(x+14, y+30+i*29, line, size) for i, line in enumerate(lines)))


def arrow(x1, y1, x2, y2):
    return f'<path d="M{x1} {y1}L{x2} {y2}" fill="none" stroke="#307f82" stroke-width="3" marker-end="url(#a)"/>'


def save(name, title, body, foot, height=500):
    OUT.mkdir(parents=True, exist_ok=True)
    svg = (f'<svg xmlns="http://www.w3.org/2000/svg" width="1000" height="{height}" viewBox="0 0 1000 {height}" role="img"><title>{escape(title)}</title>'
           '<defs><marker id="a" markerWidth="8" markerHeight="8" refX="7" refY="3" orient="auto"><path d="M0 0L0 6L8 3Z" fill="#307f82"/></marker></defs>'
           f'<rect width="1000" height="{height}" rx="18" fill="#f5f9fc"/><g font-family="PingFang SC,Microsoft YaHei,Arial,sans-serif">' +
           txt(30,45,title,28,weight=600) + body + txt(30,height-24,foot,17,"#536c80") + '</g></svg>')
    (OUT/name).write_text(svg, encoding="utf-8")


def to_image(values, bar=False):
    array = np.asarray(values)
    if array.ndim == 3:
        array = array.transpose(1, 2, 0) if array.shape[0] == 3 else array[0]
    if bar:
        array = (array + 1) / 2
    return Image.fromarray(np.round(np.clip(array, 0, 1) * 255).astype(np.uint8))


def embed(values, x, y, width, bar=False):
    stream = io.BytesIO()
    to_image(values, bar).resize((width, width), Image.Resampling.NEAREST).save(stream, format="PNG")
    data = base64.b64encode(stream.getvalue()).decode()
    return f'<image x="{x}" y="{y}" width="{width}" height="{width}" href="data:image/png;base64,{data}"/>'


def prepare_assets(data):
    OUT.mkdir(parents=True, exist_ok=True)
    for key in ("source", "candidate", "target", "mask"):
        to_image(data[key]).save(OUT/(key+'.png'))
    chosen = [c for c in data['bar_cases'] if c['seed']==42 and c['label']==1]
    body = txt(30,93,'固定竖条源图、种子 42 和横条条件；MLP 实际预测，DDPM 实际更新。',21)
    body += txt(30,132,'先看加噪起点',21,weight=600)
    for i,c in enumerate(chosen):
        x=30+i*163
        body+=embed(c['noisy'],x,150,132,True)+txt(x,310,f"强度 {c['strength']:g}",20)
    body += txt(30,367,'再看条件去噪结果',21,weight=600)
    for i,c in enumerate(chosen):
        x=30+i*163
        body+=embed(c['result'],x,386,132,True)+txt(x,548,f"更新 {c['updates']} 次",19)
    save('strength-results.svg','实际图生图小实验：从不同噪声水平开始',body,
         '训练数据：六张 8×8 模板；条件：横／竖类别；强度 0 直接返回原图。',610)

    body=''
    for i,(key,title) in enumerate([('source','输入：红方块＋蓝圆形'),('mask','硬遮罩：白色允许编辑'),('local','硬合成：只替换方块'),('candidate','候选图：故意改变其他区域'),('soft_mask','柔化遮罩：边界有中间值'),('soft','柔化合成：边缘也发生变化')]):
        row,col=divmod(i,3);x=30+col*330;y=125+row*300
        value=data['rgb_results'][key] if key in ('local','soft') else data[key]
        body+=txt(x,y-20,title,19)+embed(value,x,y,220)
    save('mask-results.svg','实际数组合成：遮罩范围决定哪些像素被替换',body,
         '数组合成：I_out = M · I_edit + (1 − M) · I_source。',730)

    body=''
    for i,(key,title) in enumerate([('bar_source','原图：竖条'),('bar_mask','允许编辑的中间四行'),('bar_unrestricted','无区域约束'),('bar_local','逐步约束＋最终保留')]):
        x=30+i*245;body+=txt(x,100,title,19)+embed(data[key],x,125,200,key!='bar_mask')
    body+=txt(30,384,'遮罩外像素最大误差 = 0；这是显式保留规则的保证，不代表模型懂了“只修改”。',21)
    save('toy-inpaint.svg','训练后的小模型：在允许区域生成，并保留已知区域',body,
         '8×8 条纹模型；横条条件；强度 0.9；种子 42；允许编辑中间四行。',450)

    body=''
    for i,(key,title) in enumerate([('source','原图'),('edges','Sobel 边缘（二值）'),('target','改色图：结构相同')]):
        x=30+i*330;body+=txt(x,102,title,21)+embed(data[key],x,125,235)
    body+=txt(30,414,'相同轮廓可以对应不同颜色；结构输入只约束一部分信息，文字仍需补足要求。',21)
    save('edge-guide.svg','实际提取结构条件：边缘不等于完整图像',body,
         '灰度图 → Sobel 卷积 → 梯度幅值 → 阈值二值化。',480)

    body=''
    names=[('unchanged','不编辑'),('global','整图改变'),('local','仅目标区域'),('soft','柔化合成')]
    for i,(key,title) in enumerate(names):
        x=30+i*245;m=data['report']['rgb_metrics'][key]
        body+=txt(x,100,title,23,weight=600)+embed(data['rgb_results'][key],x,123,200)
        body+=txt(x,366,f"区域外误差 {m['outside_mae']:.4f}",19)+txt(x,402,f"区域内目标误差 {m['inside_target_mae']:.4f}",19)
    save('evaluation-cases.svg','真实算例：仅靠“保留得很好”不能证明编辑成功',body,
         '已知对齐目标时，可用像素误差分别检查编辑区域与保留区域。',475)
    lab=OUT/'edit-lab';lab.mkdir(exist_ok=True)
    keys=['report','source','candidate','target','mask','bar_cases','bar_source','bar_mask','bar_local','alpha_bar','shapes']
    payload={key:data[key] for key in keys}
    (lab/'data.js').write_text('window.EDIT_DATA='+json.dumps(payload,ensure_ascii=False,separators=(',',':'))+';\n',encoding='utf-8')
    (ROOT/'demos/lecture_06_result.json').write_text(json.dumps(data['report'],ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
