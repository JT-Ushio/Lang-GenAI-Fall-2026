import fs from 'node:fs/promises';
const ROOT = '/Users/taoji/Github/课程/lectures';

// Run once after the 11 extension slides have been appended to the original deck.
// A second run is harmless: pages are matched by their final unique titles.
export async function sync({presentation}) {
    const content = JSON.parse(await fs.readFile(ROOT + '/presentations/lecture_08/content.json', 'utf8'));
    if (presentation.slides.items.length !== content.length) throw new Error('Incomplete deck');
    const slides = presentation.slides.items;
    const find = title => slides.find(s => s.shapes.items[0]?.text.toString() === title);
    for (const [key, oldTitle] of [['old_39', '三个音乐模型的公开材料'], ['old_42', '回顾与阅读']]) {
        const data = content.find(r => r.key === key);
        const slide = find(data.title) ?? find(oldTitle);
        slide.shapes.items[0].text = data.title;
        slide.shapes.items[1].text = data.paragraphs.map(x => x.replaceAll('**', '')).join('\n\n');
        slide.speakerNotes.text = data.paragraphs.join('\n') + '\n' + data.notes + '\n' + data.sources.join('\n');
    }
    const ids = content.map(data => {
        const slide = find(data.title);
        if (!slide) throw new Error('Missing slide: ' + data.title);
        return slide.id;
    });
    presentation.slides.reorder(ids);
    for (let i = 0; i < presentation.slides.items.length; i++) {
        const slide = presentation.slides.getItem(i);
        const footer = slide.shapes.items.find(s => /^\d{2}$/.test(s.text.toString()));
        if (footer) footer.text = String(i + 1).padStart(2, '0');
    }
    return {count: presentation.slides.items.length, titles: content.map(r=>r.title)};
}
