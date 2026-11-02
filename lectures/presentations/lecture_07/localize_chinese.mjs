import fs from 'node:fs/promises';
const ROOT='/Users/taoji/Github/课程/lectures';
const clean=s=>s.replaceAll('**','');
export async function localize({presentation,input={}}){
 const replacements=JSON.parse(await fs.readFile(ROOT+'/presentations/lecture_07/chinese_replacements.json','utf8'));
 const content=JSON.parse(await fs.readFile(ROOT+'/presentations/lecture_07/content.json','utf8'));
 const changes=[];
 const before=presentation.slides.items.map(s=>({id:s.id,shapes:s.shapes.items.map(v=>v.id),images:s.images.items.map(v=>v.id)}));
 for(let i=0;i<presentation.slides.items.length;i++){
  const s=presentation.slides.getItem(i);
  for(const sh of s.shapes.items){
   let v=sh.text.toString(),next=v;
   for(const [old,repl] of Object.entries(replacements))next=next.replaceAll(clean(old),clean(repl));
   if(v.includes("reference = 'she sees six ships'.split()"))next=content[29].snippet;
   if(next!==v){sh.text=next;changes.push(i+1);}
  }
  let note=s.speakerNotes.text.toString();
  for(const [old,repl]of Object.entries(replacements))note=note.replaceAll(clean(old),clean(repl));
  if(i===21)note+='\n中文教学图依据 tiny 官方配置，AI 辅助生成，图中波形和频谱为示意，非模型运行结果。';
  if(i===29)note+='\n中文 WER 采用显式分词：[我, 今天, 学习, 语音] 与 [我, 学习, 语音]，删除“今天”，WER=1/4。';
  s.speakerNotes.text=note;
 }
 // Replace only the nine illustrations whose examples or source audio changed.
 for(const i of (input.imageSlides || [])){
  const s=presentation.slides.getItem(i);const fig=content[i].figure;
  for(const im of s.images.items)im.replace({path:ROOT+'/images/lecture_07/'+fig,alt:content[i].title,fit:'contain'});
  changes.push(i+1);
 }
 const after=presentation.slides.items.map(s=>({id:s.id,shapes:s.shapes.items.map(v=>v.id),images:s.images.items.map(v=>v.id)}));
 if(JSON.stringify(before)!==JSON.stringify(after))throw Error('Unexpected structural change');
 const changed=[...new Set(changes)].sort((a,b)=>a-b);
 if(input.render) await presentation.render({slide:changed.map(i=>i-1)});
 return {changedSlides:changed,count:presentation.slides.items.length,structurePreserved:true};
}
