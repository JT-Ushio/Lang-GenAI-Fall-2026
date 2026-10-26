"""A learned, shared autoregressive model plus an explicitly rule-based agent.

The finite vocabulary describes two colored blocks. Pixels are encoded with a
fixed palette, not a learned VQ encoder. All 72 examples are training examples;
free-running exact match measures fitting, not held-out generalization.
"""
import hashlib
import json
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F

TOKENS = ['<pad>', '<bos>', '<eos>', '<image>', '</image>', '<answer>',
          '理解', '生成', '编辑', '左', '右', '红', '绿', '蓝',
          '像素_背景', '像素_红', '像素_绿', '像素_蓝']
IDS = {word:i for i,word in enumerate(TOKENS)}
COLORS = ['红', '绿', '蓝']
PALETTE = [[241,245,249], [225,69,67], [51,175,99], [52,111,219]]
ROOT = Path(__file__).resolve().parents[1]


def scene(left=0, right=2):
    pixels = torch.zeros(4,4,dtype=torch.long)
    pixels[1:3,:2] = left+1
    pixels[1:3,2:] = right+1
    return pixels


def image_tokens(pixels):
    return [IDS['<image>']] + [14+int(x) for x in pixels.flatten()] + [IDS['</image>']]


def make_example(task, left, right, side=0, color=1):
    source = scene(left,right)
    description = [IDS['左'], IDS[COLORS[left]], IDS['右'], IDS[COLORS[right]]]
    if task == '理解':
        prompt = [IDS['<bos>'],IDS[task]] + image_tokens(source) + [IDS['<answer>']]
        answer = description + [IDS['<eos>']]
        target = None
    elif task == '生成':
        prompt = [IDS['<bos>'],IDS[task]] + description + [IDS['<answer>']]
        target = source
        answer = image_tokens(target) + [IDS['<eos>']]
    else:
        prompt = ([IDS['<bos>'],IDS[task]] + image_tokens(source)
                  + [IDS['左' if side==0 else '右'],IDS[COLORS[color]],IDS['<answer>']])
        target = scene(color,right) if side==0 else scene(left,color)
        answer = image_tokens(target) + [IDS['<eos>']]
    return {'task':task,'left':left,'right':right,'side':side,'color':color,
            'prompt':prompt,'answer':answer,'source':source.tolist(),
            'target':target.tolist() if target is not None else None}


def dataset():
    rows=[]
    for left in range(3):
        for right in range(3):
            rows.extend([make_example('理解',left,right),make_example('生成',left,right)])
            rows.extend(make_example('编辑',left,right,side,color)
                        for side in range(2) for color in range(3))
    return rows


def pack(rows):
    length=max(len(r['prompt'])+len(r['answer'])-1 for r in rows)
    inputs=torch.full((len(rows),length),IDS['<pad>'],dtype=torch.long)
    labels=torch.full_like(inputs,-100)
    for i,row in enumerate(rows):
        sequence=row['prompt']+row['answer']
        n=len(sequence)-1
        inputs[i,:n]=torch.tensor(sequence[:-1])
        labels[i,len(row['prompt'])-1:n]=torch.tensor(row['answer'])
    return inputs,labels


class TinyUnified(nn.Module):
    def __init__(self, width=48):
        super().__init__()
        self.width=width
        self.embedding=nn.Embedding(len(TOKENS),width)
        self.position=nn.Embedding(64,width)
        block=nn.TransformerEncoderLayer(width,4,width*2,dropout=0.,
                                        activation='gelu',batch_first=True,norm_first=True)
        self.backbone=nn.TransformerEncoder(block,2,enable_nested_tensor=False)
        self.norm=nn.LayerNorm(width)
        self.head=nn.Linear(width,len(TOKENS))

    def forward(self, ids):
        length=ids.shape[1]
        hidden=self.embedding(ids)+self.position(torch.arange(length))
        blocked=torch.ones(length,length,dtype=torch.bool).triu(1)
        hidden=self.backbone(hidden,mask=blocked,
                             src_key_padding_mask=ids.eq(IDS['<pad>']),is_causal=True)
        return self.head(self.norm(hidden))


def loss_for(model, rows):
    ids,labels=pack(rows)
    logits=model(ids)
    return F.cross_entropy(logits.reshape(-1,len(TOKENS)),labels.reshape(-1),ignore_index=-100)


@torch.no_grad()
def generate_many(model, rows, max_tokens=24):
    model.eval()
    assert len({len(row['prompt']) for row in rows})==1
    prompt_length=len(rows[0]['prompt'])
    ids=torch.tensor([row['prompt'] for row in rows])
    finished=torch.zeros(len(rows),dtype=torch.bool)
    for _ in range(max_tokens):
        next_id=model(ids)[:,-1].argmax(-1)
        next_id=torch.where(finished,IDS['<pad>'],next_id)
        ids=torch.cat([ids,next_id[:,None]],dim=1)
        finished |= next_id.eq(IDS['<eos>'])
        if finished.all():break
    results=[]
    for sequence in ids[:,prompt_length:].tolist():
        if IDS['<eos>'] in sequence:sequence=sequence[:sequence.index(IDS['<eos>'])+1]
        results.append(sequence)
    return results


def parse_image(ids):
    if len(ids)!=19 or ids[0]!=IDS['<image>'] or ids[-2:]!=[IDS['</image>'],IDS['<eos>']]:
        return None
    if any(x<14 or x>17 for x in ids[1:17]):return None
    return torch.tensor([x-14 for x in ids[1:17]]).reshape(4,4)


def parse_caption(ids):
    if len(ids)!=5 or ids[0]!=IDS['左'] or ids[2]!=IDS['右'] or ids[-1]!=IDS['<eos>']:return None
    if ids[1] not in (11,12,13) or ids[3] not in (11,12,13):return None
    return (ids[1]-11,ids[3]-11)


def observe_pixels(pixels):
    """Independent exact verifier for the fixed geometry, not a neural judge."""
    for left in range(3):
        for right in range(3):
            if torch.equal(pixels,scene(left,right)):return (left,right)
    return None


def caption_image(model,pixels):
    row=make_example('理解',0,0)
    row['prompt']=[IDS['<bos>'],IDS['理解']]+image_tokens(pixels)+[IDS['<answer>']]
    predicted=generate_many(model,[row])[0]
    return parse_caption(predicted),predicted


def edit_image(model,pixels,side,color):
    row=make_example('编辑',0,0,side,color)
    row['prompt']=([IDS['<bos>'],IDS['编辑']]+image_tokens(pixels)
                   +[IDS['左' if side==0 else '右'],IDS[COLORS[color]],IDS['<answer>']])
    predicted=generate_many(model,[row])[0]
    return parse_image(predicted),predicted


def run_agent(model, fault=False, budget=3):
    """Rule policy + actual neural perception/editing + independent verifier.

    Goal: make left green and preserve the initial blue right block. The injected
    fault executes the first edit on the wrong side; no pixels are patched to
    manufacture a successful final result.
    """
    original=scene(0,2); current=original.clone(); goal=(1,2); history=[]
    for step in range(budget):
        perceived,caption=caption_image(model,current)
        if perceived is None:return {'status':'观察格式错误','success':False,'steps':history}
        # First restore a damaged preservation constraint, then reach the target.
        if perceived[1]!=goal[1]:side,color=1,goal[1]
        elif perceived[0]!=goal[0]:side,color=0,goal[0]
        else:
            ok=torch.equal(current,scene(*goal))
            return {'status':'完成' if ok else '独立验收失败','success':ok,'steps':history}
        executed_side=1-side if fault and step==0 else side
        proposed,output=edit_image(model,current,executed_side,color)
        if proposed is None:return {'status':'图像格式错误','success':False,'steps':history}
        actual=observe_pixels(proposed)
        target_ok=actual is not None and actual[0]==goal[0]
        preserve_ok=torch.equal(proposed[:,2:],original[:,2:])
        success=torch.equal(proposed,scene(*goal))
        history.append({'step':step+1,'before':current.tolist(),'observed':list(perceived),
                        'caption_tokens':caption,'planned_side':side,'executed_side':executed_side,
                        'color':color,'after':proposed.tolist(),'output_tokens':output,
                        'measured':list(actual) if actual is not None else None,
                        'target_ok':target_ok,'preserve_ok':preserve_ok,'success':success})
        current=proposed
        if success:return {'status':'完成','success':True,'steps':history}
    return {'status':'预算耗尽，未完成','success':False,'steps':history}


def block_attention(groups):
    """Simplified modality-block causality, not full BAGEL training attention."""
    group_ids=torch.tensor(groups)
    order=torch.arange(len(groups))
    previous=group_ids[None,:] < group_ids[:,None]
    same=group_ids[None,:] == group_ids[:,None]
    visual=group_ids[:,None].eq(1)  # Middle image block has full internal attention.
    return previous | (same & (visual | (order[None,:]<=order[:,None])))


def joint_gradient_demo():
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(67)
        model=TinyUnified()
        text_loss=loss_for(model,[make_example('理解',0,2)])
        image_loss=loss_for(model,[make_example('生成',0,2)])
        weight=model.backbone.layers[0].linear1.weight
        grad_text=torch.autograd.grad(text_loss,weight,retain_graph=True)[0]
        grad_image=torch.autograd.grad(image_loss,weight)[0]
        result={'understanding_grad_norm':float(grad_text.norm()),
                'generation_grad_norm':float(grad_image.norm())}
        assert min(result.values())>0
        return result


def train_demo(updates=1000):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(61)
        model=TinyUnified();rows=dataset()
        groups={task:[r for r in rows if r['task']==task] for task in ('理解','生成','编辑')}
        optim=torch.optim.AdamW(model.parameters(),lr=.003,weight_decay=.01)
        rng=torch.Generator().manual_seed(62);history=[]
        for step in range(updates+1):
            if step%100==0:
                model.eval()
                with torch.no_grad():
                    metrics={task:float(loss_for(model,group)) for task,group in groups.items()}
                history.append({'step':step,**metrics})
            if step==updates:break
            model.train();optim.zero_grad()
            losses=[]
            for task,group in groups.items():
                indices=torch.randint(len(group),(9,),generator=rng).tolist()
                losses.append(loss_for(model,[group[i] for i in indices]))
            loss=torch.stack(losses).mean();loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(),1.)
            optim.step()
        records=[];scores={}
        for task,group in groups.items():
            outputs=generate_many(model,group)
            exact=0
            for row,prediction in zip(group,outputs):
                ok=prediction==row['answer'];exact+=ok
                pixels=parse_image(prediction) if task!='理解' else None
                caption=parse_caption(prediction) if task=='理解' else None
                records.append({**row,'prediction':prediction,'correct':ok,
                                'predicted_image':pixels.tolist() if pixels is not None else None,
                                'predicted_caption':list(caption) if caption is not None else None})
            scores[task]={'correct':exact,'total':len(group),'exact_match':exact/len(group)}
        digest=hashlib.sha256()
        for key,value in sorted(model.state_dict().items()):
            digest.update(key.encode());digest.update(value.detach().numpy().tobytes())
        report={'parameters':sum(p.numel() for p in model.parameters()),'updates':updates,
                'width':48,'layers':2,'heads':4,'vocab':len(TOKENS),'history':history,
                'training_set_free_running':scores,'model_sha256':digest.hexdigest(),
                'shared_gradients':joint_gradient_demo(),
                'scope':'All 72 cases used for training; finite symbolic language and fixed palette. Rule-based planning; neural captioning and editing share one model.'}
        episodes={'normal':run_agent(model),'fault':run_agent(model,True,3),
                  'short':run_agent(model,True,2)}
        assert episodes['normal']['success'] and len(episodes['normal']['steps'])==1
        assert episodes['fault']['success'] and len(episodes['fault']['steps'])==3
        assert not episodes['fault']['steps'][0]['preserve_ok']
        assert not episodes['short']['success'] and len(episodes['short']['steps'])==2
        return model,{'report':report,'records':records,'episodes':episodes,
                      'tokens':TOKENS,'palette':PALETTE,'attention':block_attention([0,0,1,1,2,2]).int().tolist()}


def build_experiments():
    previous=torch.get_num_threads();torch.set_num_threads(1)
    try:
        model,data=train_demo()
        data['_model']=model
        return data
    finally:torch.set_num_threads(previous)


if __name__=='__main__':
    result=build_experiments()
    print(json.dumps({'report':result['report'],'episodes':result['episodes']},ensure_ascii=False,indent=2))
