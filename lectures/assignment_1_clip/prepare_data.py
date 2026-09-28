"""Join official four-language text with image archives by filename, never row guessing."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
from zipfile import ZipFile
from PIL import Image

SPLITS = {'train': (29000, ['en','de','fr','cs']), 'val': (1014, ['en','de','fr','cs']), 'test_2016_flickr': (1000, ['en','de','fr','cs']), 'test_2017_flickr': (1000, ['en','de','fr']), 'test_2017_mscoco': (461, ['en','de','fr'])}

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    args=p.parse_args()
    root=args.root
    images=root/'images'; images.mkdir(exist_ok=True)
    archives=[root/'flickr30k-images.zip', root/'multi30k_test2017_task1.zip']
    report={'archives':{},'splits':{}}
    for archive in archives:
        if not archive.exists():
            raise FileNotFoundError(archive)
        with ZipFile(archive) as z:
            for member in z.infolist():
                name=Path(member.filename).name
                if not name.lower().endswith(('.jpg','.jpeg','.png')):
                    continue
                dest=images/name
                blob=z.read(member)  # Also checks ZIP CRC.
                if dest.exists():
                    if hashlib.sha256(dest.read_bytes()).digest()!=hashlib.sha256(blob).digest():
                        raise ValueError(f'Conflicting image filename: {name}')
                else:
                    dest.write_bytes(blob)
        report['archives'][archive.name]={'bytes':archive.stat().st_size}
        print('Extracted and CRC checked:',archive,flush=True)
    all_ids={}
    manifests=root/'manifests'; manifests.mkdir(exist_ok=True)
    for split,(expected,languages) in SPLITS.items():
        ids=[line.split('#')[0] for line in (root/'official-task1/image_splits'/f'{split}.txt').read_text().splitlines()]
        assert len(ids)==expected and len(set(ids))==expected
        captions={lang:gzip.open(root/'official-task1/raw'/f'{split}.{lang}.gz','rt',encoding='utf8').read().splitlines() for lang in languages}
        assert all(len(lines)==expected and all(x.strip() for x in lines) for lines in captions.values())
        records=[]
        for index,name in enumerate(ids):
            path=images/name
            with Image.open(path) as im:
                im.verify()
            records.append({'image_id':name,'image_path':str(path.resolve()),'texts':{lang:captions[lang][index] for lang in languages}})
        (manifests/f'{split}.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in records))
        all_ids[split]=set(ids)
        report['splits'][split]={'images':len(records),'languages':languages,'pairs':len(records)*len(languages)}
        print(split,report['splits'][split],flush=True)
    report['overlaps']={f'{a}/{b}':len(all_ids[a]&all_ids[b]) for i,a in enumerate(SPLITS) for b in list(SPLITS)[i+1:]}
    assert all(v==0 for v in report['overlaps'].values()), report['overlaps']
    (root/'data_report.json').write_text(json.dumps(report,indent=2))

if __name__=='__main__':
    main()
