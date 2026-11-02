import fs from 'node:fs/promises';
const ROOT='/Users/taoji/Github/课程/lectures';
const strip=s=>s.replaceAll('**','');
function tb(slide,text,x,y,w,h,size=26,color='#153553',bold=false){
 const sh=slide.shapes.add({geometry:'textbox',position:{left:x,top:y,width:w,height:h},fill:'none',line:{fill:'none',width:0}});
 sh.text=text;sh.text.style={typeface:'PingFang SC',fontSize:size,color,bold};return sh;
}
export async function build({presentation,input}){
 const content=JSON.parse(await fs.readFile(ROOT+'/presentations/lecture_07/content.json','utf8'));
 for(let i=input.start;i<input.end;i++){
  const d=content[i],s=presentation.slides.add({width:1280,height:720});s.background.fill='#F8FAFC';
  tb(s,d.title,64,38,1150,82,38,'#153553',true);
  const hasFig=!!d.figure || !!d.snippet;
  if(['whisper-official.png','whisper-chinese.png'].includes(d.figure)){
   s.images.add({path:ROOT+'/images/lecture_07/'+d.figure,fit:'contain',position:{left:65,top:133,width:1140,height:355},alt:'Whisper 架构教学图'});
   tb(s,d.paragraphs.map(strip).join('\n\n'),65,510,1140,155,21);
  }else{
   const width=hasFig?560:1110;
   const textSize=hasFig?24:28;
   let y=155;
   for(const p of d.paragraphs){
    const plain=strip(p);const cols=hasFig?22:38;
    const lines=Math.ceil([...plain].reduce((a,c)=>a+(c.charCodeAt(0)>255?1:.52),0)/cols);
    const h=Math.max(textSize*1.4,lines*textSize*1.4)+18;
    tb(s,plain,70,y,width,h,textSize);y+=h;
   }
   if(d.snippet){ const code=tb(s,d.snippet,660,165,550,440,20,'#167D8D');code.text.style={typeface:'Menlo',fontSize:20,color:'#167D8D'};}
   else if(hasFig)s.images.add({path:ROOT+'/images/lecture_07/'+d.figure,fit:'contain',position:{left:650,top:155,width:570,height:430},alt:d.title+'：实际数据或说明图'});
  }
  tb(s,String(i+1).padStart(2,'0'),1150,665,65,28,17,'#6B7D91');
  const extra=d.code?'\n可执行演示：lectures/lecture_07.py 中的 '+d.code+'。课堂运行产生的实际输出见 trace。':'';
  s.speakerNotes.text=d.paragraphs.map(strip).join('\n')+extra+'\n'+(d.source?'资料：'+d.source:'本课原创解释与本地数值演示。')+(['whisper-official.png','whisper-chinese.png'].includes(d.figure)?'\n图源：https://raw.githubusercontent.com/openai/whisper/main/approach.png，MIT 许可。':'');
 }
 return {count:presentation.slides.items.length};
}
