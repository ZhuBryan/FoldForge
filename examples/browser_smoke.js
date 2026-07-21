// Headless-browser smoke test for studio/index.html.
// Catches page-level failures that unit tests can't (dead handlers, runtime
// throws in the upload path, CDN/offline issues) and exercises the photo
// pipeline: subject fold, bilateral Symmetry (Auto), the pipeline strip and
// the showcase gallery.
//
// Setup:  npm install puppeteer-core
//         set CHROME to a Chrome/Chromium binary (or install puppeteer for a bundled one)
// Run:    CHROME=/path/to/chrome node examples/browser_smoke.js [image.jpg]
//         (defaults to the baked butterfly sample so Symmetry=Auto can fire)
const puppeteer = require('puppeteer-core');
const path = require('path');
const fs = require('fs');
(async () => {
  const sample = path.join(__dirname, 'samples', 'butterfly.jpg');
  const img = process.argv[2] || (fs.existsSync(sample) ? sample : path.join(__dirname, '..', 'zebra.jpg'));
  const html = 'file://' + path.join(__dirname, '..', 'studio', 'index.html');
  const shot = path.join(__dirname, 'output', 'studio_pipeline_ui.png');
  const browser = await puppeteer.launch({
    executablePath: process.env.CHROME || '/usr/bin/google-chrome',
    args: ['--no-sandbox', '--disable-gpu', '--allow-file-access-from-files']});
  const page = await browser.newPage();
  await page.setViewport({width: 1280, height: 800});
  const errs = [];
  page.on('pageerror', e => errs.push(e.message));
  await page.goto(html);
  await new Promise(r => setTimeout(r, 2000));
  const status = async () => page.evaluate(() => document.getElementById('status').textContent);
  const fail = async (msg) => { console.error('FAIL:', msg, '| status:', await status(), '| pageerrors:', errs); await browser.close(); process.exit(1); };

  if (!/Ready/.test(await status())) await fail('page did not become Ready');

  // ---- photo upload with Symmetry = Auto (default) on the butterfly sample ----
  await page.select('#symmode', 'auto');
  const input = await page.$('#imgfile');
  await input.uploadFile(img);
  await new Promise(r => setTimeout(r, 3500));
  const folded = await status();
  if (!/^Folded /.test(folded)) await fail('image upload did not fold');
  if (!/mirror-IoU\s+[0-9.]+/.test(folded)) await fail('symmetry mirror score not reported in status');
  const mirror = (folded.match(/mirror-IoU\s+([0-9.]+)/) || [])[1];
  const applied = /symmetrized/.test(folded);

  // ---- pipeline strip populated (input photo -> height field [-> symmetrized]) ----
  const pipe = await page.evaluate(() => {
    const vis = el => el && !el.classList.contains('hidden') && getComputedStyle(el).display !== 'none';
    return {
      strip: getComputedStyle(document.getElementById('pipeline')).display,
      input: vis(document.getElementById('pstage-input')),
      height: vis(document.getElementById('pstage-height')),
      sym: vis(document.getElementById('pstage-sym')),
    };
  });
  if (pipe.strip === 'none') await fail('pipeline strip not shown');
  if (!pipe.input || !pipe.height) await fail('pipeline strip stages missing (input/height)');

  // ---- rough-detail rebuild still works ----
  await page.select('#detail', '10');
  await new Promise(r => setTimeout(r, 1800));
  if (!/9 folds/.test(await status())) await fail('rough detail rebuild failed');

  // ---- showcase gallery: cards present + load a baked sample ----
  await page.click('#galleryBtn');
  await new Promise(r => setTimeout(r, 300));
  const cards = await page.evaluate(() => document.querySelectorAll('#gallerylist .gcard').length);
  if (cards < 4) await fail('showcase gallery has too few sample cards: ' + cards);
  // load the symmetrized butterfly bake (carries a mirror-IoU from the offline pipeline)
  const gsym = await page.evaluate(() => {
    if (typeof loadBaked !== 'function' || !DATA.shapes['butterfly_sym']) return null;
    loadBaked('butterfly_sym');
    const s = DATA.shapes['butterfly_sym'].sym;
    return s ? {iou: s.iou, applied: s.applied, engine: DATA.shapes['butterfly_sym'].engine} : null;
  });
  if (!gsym) await fail('baked butterfly_sym / gallery load unavailable');
  await new Promise(r => setTimeout(r, 1500));

  // ---- step-fold the loaded sample, then screenshot the new UI ----
  await page.select('#foldmode', 'step');
  await page.evaluate(() => { const s = document.getElementById('slider'); s.value = 1; s.dispatchEvent(new Event('input')); });
  await new Promise(r => setTimeout(r, 1200));
  await page.screenshot({path: shot});

  if (errs.length) await fail('uncaught page errors');
  console.log('OK — studio folds', path.basename(img),
    '| live mirror-IoU', mirror, applied ? '(symmetrized)' : '(left as-is)',
    '| baked butterfly_sym IoU', gsym.iou, '| gallery cards', cards,
    '| pipeline sym-stage', pipe.sym, '| screenshot', shot, '|', await status());
  await browser.close();
})().catch(e => { console.error('FAIL:', e.message); process.exit(1); });
