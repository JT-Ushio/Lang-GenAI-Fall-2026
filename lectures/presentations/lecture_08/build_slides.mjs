import fs from 'node:fs/promises';
const ROOT = '/Users/taoji/Github/课程/lectures';
const clean = text => text.replaceAll('**', '');

function textbox(slide, text, position, size, color = '#153553', bold = false, typeface = 'PingFang SC') {
    const shape = slide.shapes.add({
        geometry: 'textbox', position, fill: 'none',
        line: {fill: 'none', width: 0},
    });
    shape.text = text;
    shape.text.style = {typeface, fontSize: size, color, bold, lineSpacing: 1.15};
    return shape;
}

export async function build({presentation, input}) {
    const content = JSON.parse(await fs.readFile(ROOT + '/presentations/lecture_08/content.json', 'utf8'));
    const indices = input.indices ?? Array.from({length: input.end - input.start}, (_, j) => input.start + j);
    for (const i of indices) {
        const data = content[i];
        const slide = presentation.slides.add({width: 1280, height: 720});
        slide.background.fill = '#F8FAFC';
        textbox(slide, data.title, {left: 64, top: 38, width: 1150, height: 90}, 38, '#153553', true);
        if (data.table) {
            const table = slide.tables.add({rows: data.table.length, columns: data.table[0].length,
                left: 70, top: 165, width: 1140, height: 210, values: data.table,
                columnWidths: [400, 245, 245, 250]});
            for (let row = 0; row < data.table.length; row++) {
                for (let col = 0; col < data.table[0].length; col++) {
                    const cell = table.getCell(row, col);
                    cell.fill = row === 0 ? '#153553' : '#E8F1F4';
                    cell.text.style = {typeface: 'PingFang SC', fontSize: 23,
                        color: row === 0 ? '#FFFFFF' : '#153553', bold: row === 0};
                }
            }
            textbox(slide, data.paragraphs.map(clean).join('\n\n'),
                {left: 70, top: 415, width: 1140, height: 235}, 24);
        } else if (data.layout === 'figure') {
            slide.images.add({
                path: ROOT + '/images/lecture_08/' + data.figure,
                fit: 'contain', alt: data.title,
                position: {left: 64, top: 132, width: 1152, height: 350},
            });
            textbox(slide, data.paragraphs.map(clean).join('\n\n'),
                {left: 70, top: 500, width: 1140, height: 160}, 22);
        } else {
            const divided = Boolean(data.figure || data.snippet);
            textbox(slide, data.paragraphs.map(clean).join('\n\n'),
                {left: 70, top: 155, width: divided ? 550 : 1110, height: 480},
                divided ? 24 : 28);
            if (data.snippet) {
                textbox(slide, data.snippet,
                    {left: 660, top: 165, width: 550, height: 465},
                    20, '#167D8D', false, 'Menlo');
            } else if (data.figure) {
                slide.images.add({
                    path: ROOT + '/images/lecture_08/' + data.figure,
                    fit: 'contain', alt: data.title,
                    position: {left: 650, top: 160, width: 570, height: 440},
                });
            }
        }
        textbox(slide, String(i + 1).padStart(2, '0'),
            {left: 1150, top: 665, width: 65, height: 28}, 17, '#6B7D91');
        slide.speakerNotes.text = data.paragraphs.map(clean).join('\n') +
            '\n' + data.notes + '\n' + data.sources.join('\n') +
            (data.demo ? '\n配套网页可执行演示 ' + data.demo + '。音频在网页中试听。' : '');
    }
    return {count: presentation.slides.items.length};
}
