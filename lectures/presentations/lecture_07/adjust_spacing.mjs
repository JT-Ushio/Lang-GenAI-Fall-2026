/** Preserve every slide, shape, image, and text string. Change typography and position only. */
export async function adjust({presentation}){
 const before=presentation.slides.items.map(s=>({id:s.id,shapes:s.shapes.items.map(v=>[v.id,v.text.toString()]),images:s.images.items.map(v=>v.id)}));
 for(let i=0;i<presentation.slides.items.length;i++){
  const s=presentation.slides.getItem(i);const shapes=s.shapes.items;
  for(let j=0;j<shapes.length;j++){
   const sh=shapes[j],txt=sh.text.toString();
   const isCode=txt.includes('\n')&&(txt.includes('torch.')||txt.includes('weights =')||txt.includes('def collapse')||txt.includes('with wave')||txt.includes('reference ='));
   sh.text.typeface=isCode?'Menlo':'PingFang SC';
  }
  if(i===0||i===21)continue;
  const body=shapes.filter((v,j)=>j>0&&v.position.left===70);
  const row=480/body.length;
  for(let j=0;j<body.length;j++){
   const sh=body[j];sh.position={...sh.position,top:150+j*row,height:row-8};
  }
 }
 const after=presentation.slides.items.map(s=>({id:s.id,shapes:s.shapes.items.map(v=>[v.id,v.text.toString()]),images:s.images.items.map(v=>v.id)}));
 if(JSON.stringify(before)!==JSON.stringify(after))throw Error('Content or IDs changed');
 await presentation.render();return {count:presentation.slides.items.length,contentPreserved:true};
}
