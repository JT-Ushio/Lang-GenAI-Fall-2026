"""Real multilingual CLIP fine-tuning and full-candidate retrieval on fixed subsets."""
import argparse
import json
import math
import random
import time
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from safetensors.torch import load_file
from torch import nn
from torch.nn import functional as F
from transformers import AutoModel, AutoTokenizer, CLIPImageProcessor, CLIPVisionModelWithProjection

LANGS=['en','de','fr','cs']

class MultilingualCLIP(nn.Module):
    def __init__(self, model_root):
        super().__init__()
        image_path=model_root/'clip-vit-base-patch32'
        text_path=model_root/'clip-ViT-B-32-multilingual-v1'
        self.vision=CLIPVisionModelWithProjection.from_pretrained(image_path,local_files_only=True)
        self.text=AutoModel.from_pretrained(text_path,local_files_only=True)
        self.text_projection=nn.Linear(768,512,bias=False)
        weights=load_file(str(text_path/'2_Dense/model.safetensors'))
        self.text_projection.weight.data.copy_(weights['linear.weight'])
        self.logit_scale=nn.Parameter(torch.tensor(math.log(1/0.07)))

    def encode_image(self,pixels):
        return F.normalize(self.vision(pixel_values=pixels).image_embeds.float(),dim=-1)

    def encode_text(self,tokens):
        hidden=self.text(**tokens).last_hidden_state
        mask=tokens['attention_mask'].unsqueeze(-1).to(hidden.dtype)
        pooled=(hidden*mask).sum(1)/mask.sum(1).clamp_min(1)
        return F.normalize(self.text_projection(pooled).float(),dim=-1)


def load_split(path,limit,seed):
    records=[json.loads(line) for line in path.read_text().splitlines()]
    if limit and len(records)>limit:
        records=random.Random(seed).sample(records,limit)
    return records


def load_pixels(records,processor):
    images=[]
    for row in records:
        with Image.open(row['image_path']) as image:
            images.append(image.convert('RGB'))
    return processor(images=images,return_tensors='pt')['pixel_values']


def tokenize(records,tokenizer):
    return {lang:tokenizer([r['texts'][lang] for r in records],padding='max_length',truncation=True,max_length=64,return_tensors='pt') for lang in records[0]['texts']}


def metrics(scores):
    # rows = images, columns = texts, identities paired in manifest order
    n=scores.shape[0]
    labels=torch.arange(n)[:,None]
    result={}
    for name,matrix in [('i2t',scores),('t2i',scores.T)]:
        ranks=matrix.argsort(dim=1,descending=True,stable=True)
        for k in (1,5,10):
            result[f'{name}_R@{k}']=100*(ranks[:,:min(k,n)]==labels).any(1).float().mean().item()
    result['CLIP_Score']=100*scores.diag().clamp_min(0).mean().item()
    return result


@torch.inference_mode()
def evaluate(model,data,batch,device):
    model.eval()
    results={}
    for split,(records,pixels,tokens) in data.items():
        image_features=[]
        for start in range(0,len(records),batch):
            with torch.autocast('cuda',dtype=torch.bfloat16):
                features=model.encode_image(pixels[start:start+batch].to(device))
            image_features.append(features.cpu())
        image_features=torch.cat(image_features)
        results[split]={'candidates':len(records),'languages':{}}
        print('evaluate',split,'candidates=',len(records),flush=True)
        for lang,encoded in tokens.items():
            texts=[]
            for start in range(0,len(records),batch):
                chunk={k:v[start:start+batch].to(device) for k,v in encoded.items()}
                with torch.autocast('cuda',dtype=torch.bfloat16):
                    features=model.encode_text(chunk)
                texts.append(features.cpu())
            scores=image_features@torch.cat(texts).T
            results[split]['languages'][lang]=metrics(scores)
    return results


def validation_score(result):
    return np.mean([r[k] for r in result['val']['languages'].values() for k in ('i2t_R@1','t2i_R@1')]).item()


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--train-limit',type=int,default=256)
    p.add_argument('--eval-limit',type=int,default=256)
    p.add_argument('--epochs',type=int,default=4)
    p.add_argument('--batch-size',type=int,default=32)
    p.add_argument('--lr',type=float,default=1e-5)
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(4)
    torch.manual_seed(args.seed); np.random.seed(args.seed); random.seed(args.seed)
    assert torch.cuda.is_available(),'A CUDA GPU is required'
    device=torch.device('cuda')
    free_memory,_=torch.cuda.mem_get_info()
    assert free_memory > 16*1024**3, 'Selected GPU has less than 16 GiB free; choose an idle GPU'
    start=time.perf_counter()
    print('GPU:',torch.cuda.get_device_name(),flush=True)
    model=MultilingualCLIP(args.root/'models').to(device)
    processor=CLIPImageProcessor.from_pretrained(args.root/'models/clip-vit-base-patch32',local_files_only=True)
    tokenizer=AutoTokenizer.from_pretrained(args.root/'models/clip-ViT-B-32-multilingual-v1',local_files_only=True)
    data={}
    for split in ['train','val','test_2016_flickr','test_2017_flickr','test_2017_mscoco']:
        limit=args.train_limit if split=='train' else args.eval_limit
        records=load_split(args.root/'data/manifests'/f'{split}.jsonl',limit,args.seed)
        data[split]=(records,load_pixels(records,processor),tokenize(records,tokenizer))
        print('prepared',split,len(records),'images',flush=True)
    (args.output/'subset_ids.json').write_text(json.dumps({s:[r['image_id'] for r in d[0]] for s,d in data.items()},indent=2))
    config={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()}
    config.update({'gpu':torch.cuda.get_device_name(),'torch':torch.__version__,'precision':'bfloat16','training':'both encoders and projections','max_length':64})
    (args.output/'config.json').write_text(json.dumps(config,indent=2))
    baseline=evaluate(model,{s:d for s,d in data.items() if s!='train'},args.batch_size,device)
    (args.output/'baseline.json').write_text(json.dumps(baseline,indent=2))
    best_score=validation_score(baseline)
    torch.save(model.state_dict(),args.output/'best.pt')
    best_epoch=0
    optimizer=torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=0.01)
    records,pixels,tokens=data['train']
    history=[]
    training_start=time.perf_counter()
    for epoch in range(args.epochs):
        model.train()
        order=torch.randperm(len(records))
        losses=[]
        for step,indices in enumerate(order.split(args.batch_size)):
            if len(indices)<2:
                continue
            # A fixed per-image offset rotates languages each epoch. Over 4 epochs
            # every training image is used with all four captions, with no duplicate image in a batch.
            chosen=[LANGS[(int(i)+epoch)%4] for i in indices]
            encoded={k:torch.stack([tokens[lang][k][int(i)] for i,lang in zip(indices,chosen)]).to(device) for k in tokens['en']}
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast('cuda',dtype=torch.bfloat16):
                images=model.encode_image(pixels[indices].to(device))
                texts=model.encode_text(encoded)
                logits=model.logit_scale.exp().clamp(max=100)*(images@texts.T)
                target=torch.arange(len(indices),device=device)
                loss=(F.cross_entropy(logits,target)+F.cross_entropy(logits.T,target))/2
            assert torch.isfinite(loss), 'non-finite loss'
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(),1.0)
            optimizer.step()
            losses.append(loss.item())
            print(f'epoch={epoch+1} step={step+1} loss={loss.item():.4f}',flush=True)
        val=evaluate(model,{'val':data['val']},args.batch_size,device)
        score=validation_score(val)
        history.append({'epoch':epoch+1,'mean_train_loss':float(np.mean(losses)),'val_R1_macro':score,'val':val})
        if score>best_score:
            best_score=score; best_epoch=epoch+1
            torch.save(model.state_dict(),args.output/'best.pt')
        print(f'epoch={epoch+1} val_R1_macro={score:.3f} best_epoch={best_epoch}',flush=True)
        (args.output/'history.json').write_text(json.dumps(history,indent=2))
    train_seconds=time.perf_counter()-training_start
    torch.save(model.state_dict(),args.output/'last.pt')
    model.load_state_dict(torch.load(args.output/'best.pt',map_location=device,weights_only=True))
    final=evaluate(model,{s:d for s,d in data.items() if s!='train'},args.batch_size,device)
    test_rows=[v['languages'] for k,v in final.items() if k.startswith('test_')]
    final['summary']={'best_epoch':best_epoch,'val_R1_macro':best_score,'training_and_validation_seconds':train_seconds,'total_seconds':time.perf_counter()-start,'peak_gpu_memory_gb':torch.cuda.max_memory_allocated()/1e9,'test_retrieval_macro':float(np.mean([np.mean([v[k] for v in rows.values() for k in v if k!='CLIP_Score']) for rows in test_rows])),'test_clip_score_macro':float(np.mean([np.mean([v['CLIP_Score'] for v in rows.values()]) for rows in test_rows])),'scope':'fixed subsets for pipeline smoke test, not full assignment ranking'}
    (args.output/'results.json').write_text(json.dumps(final,indent=2))
    print('COMPLETE',json.dumps(final['summary']),flush=True)

if __name__=='__main__':
    main()
