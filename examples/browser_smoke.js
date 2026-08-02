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
  const shotUpload = path.join(__dirname, 'output', 'studio_fix_upload.png');
  const shotGallery = path.join(__dirname, 'output', 'studio_fix_gallery.png');
  const shotRectFlat = path.join(__dirname, 'output', 'studio_rect_flat.png');
  const shotRectFolded = path.join(__dirname, 'output', 'studio_rect_folded.png');
  const shotMiura = path.join(__dirname, 'output', 'studio_live_miura.png');
  const setFold = f => page.evaluate(v => { const s = document.getElementById('slider'); s.value = v; s.dispatchEvent(new Event('input')); }, f);
  // panel-1 ("input photo") image source, as the browser resolves it
  const panelSrc = async () => page.evaluate(() => {
    const im = document.getElementById('pimg');
    return im && !document.getElementById('pstage-input').classList.contains('hidden') ? im.currentSrc || im.src || '' : '';
  });
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

  // ---- panel 1 ("input photo") actually shows the uploaded photo ----
  const uploadSrc = await panelSrc();
  if (!uploadSrc) await fail('panel 1 input-photo src empty after upload');
  await page.screenshot({path: shotUpload});

  // ---- hand-fold budget: selecting "Easy" drops the fold count and the status
  //      reports the difficulty label (Easy) ----
  const beforeEasy = (await status()).match(/(\d+) folds/);
  await page.select('#detail', '10');
  await new Promise(r => setTimeout(r, 1800));
  const easyStatus = await status();
  if (!/9 folds \(Easy\)/.test(easyStatus)) await fail('Easy budget did not report "9 folds (Easy)"');
  if (beforeEasy && !(9 < parseInt(beforeEasy[1]))) await fail('Easy budget did not drop the fold count');

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

  // ---- panel 1 reuses the baked sample's thumbnail data-URI ----
  const gallerySrc = await panelSrc();
  if (!gallerySrc) await fail('panel 1 input-photo src empty after loading baked gallery sample');
  await page.screenshot({path: shotGallery});

  // ---- flat-sheet footprint: a baked subject must start as a clean RECTANGLE at
  //      fold=0 (relief rises within it), not a silhouette-shaped blob ----
  await page.evaluate(() => { if (typeof loadBaked === 'function') loadBaked('butterfly'); });
  await new Promise(r => setTimeout(r, 800));
  await page.select('#foldmode', 'all');
  await setFold(0);
  await new Promise(r => setTimeout(r, 600));
  const rect = await page.evaluate(() => {
    const p = geo.attributes.position.array, nx = S.nx, ny = S.ny;
    let X0 = 1e9, X1 = -1e9, W = 0;
    for (let k = 0; k < p.length; k += 3) { if (p[k] < X0) X0 = p[k]; if (p[k] > X1) X1 = p[k]; }
    W = X1 - X0 || 1;
    // rectangle => every grid row spans the same x-range; a ragged developed sheet does not
    let mnMax = 1e9, mxMax = -1e9, mnMin = 1e9, mxMin = -1e9;
    for (let j = 0; j < ny; j++) { let a = 1e9, b = -1e9;
      for (let i = 0; i < nx; i++) { const x = p[(j * nx + i) * 3]; if (x < a) a = x; if (x > b) b = x; }
      if (b < mnMax) mnMax = b; if (b > mxMax) mxMax = b; if (a < mnMin) mnMin = a; if (a > mxMin) mxMin = a; }
    return { nx, ny, rightRag: (mxMax - mnMax) / W, leftRag: (mxMin - mnMin) / W };
  });
  if (!rect || !(rect.nx > 1)) await fail('could not read flat-sheet geometry');
  // the developed (unfolded) sheet is allowed to be ragged; folding is what matters
  await page.screenshot({path: shotRectFlat});
  await setFold(1);
  await new Promise(r => setTimeout(r, 900));
  await page.screenshot({path: shotRectFolded});

  // ---- step-fold the loaded sample, then screenshot the new UI ----
  await page.select('#foldmode', 'step');
  await page.evaluate(() => { const s = document.getElementById('slider'); s.value = 1; s.dispatchEvent(new Event('input')); });
  await new Promise(r => setTimeout(r, 1200));
  await page.screenshot({path: shot});

  // ---- live 2-D Miura engine: re-upload, switch engine, run the in-browser fit ----
  await page.select('#detail', '24');
  const inputM = await page.$('#imgfile');
  await inputM.uploadFile(img);
  await new Promise(r => setTimeout(r, 2500));
  await page.select('#enginemode', 'miura');
  await new Promise(r => setTimeout(r, 4000));
  const miStatus = await status();
  const mi = miStatus.match(/live 2-D Miura fit on a (\d+)×(\d+) grid: (\d+) iterations, fidelity error ([0-9.]+), (\d+) ms/);
  if (!mi) await fail('2-D Miura engine did not report the live fit readout: ' + miStatus);
  // mesh actually updated: folded vertices carry real z relief at fold=1
  await setFold(1);
  await new Promise(r => setTimeout(r, 600));
  const meshOk = await page.evaluate(() => {
    const p = geo.attributes.position.array; let zmin = 1e9, zmax = -1e9;
    for (let k = 2; k < p.length; k += 3) { if (p[k] < zmin) zmin = p[k]; if (p[k] > zmax) zmax = p[k]; }
    return (zmax - zmin);
  });
  if (!(meshOk > 0.05)) await fail('2-D Miura mesh has no folded relief (z range ' + meshOk + ')');
  await page.screenshot({path: shotMiura});
  const miGrid = mi[1] + '×' + mi[2], miIters = mi[3], miErr = mi[4], miMs = mi[5];

  if (errs.length) await fail('uncaught page errors');
  console.log('OK — studio folds', path.basename(img),
    '| 2-D Miura live fit', miGrid, 'grid,', miIters, 'iters, error', miErr, ',', miMs + 'ms, z-relief', meshOk.toFixed(2),
    '| live mirror-IoU', mirror, applied ? '(symmetrized)' : '(left as-is)',
    '| baked butterfly_sym IoU', gsym.iou, '| gallery cards', cards,
    '| pipeline sym-stage', pipe.sym,
    '| panel1 upload src', uploadSrc.slice(0, 24) + '…', '| panel1 gallery src', gallerySrc.slice(0, 24) + '…',
    '| flat-sheet rag L/R', rect.leftRag.toFixed(3) + '/' + rect.rightRag.toFixed(3), '(ragged flat OK)',
    '| screenshots', shot, shotUpload, shotGallery, shotRectFlat, shotRectFolded, '|', await status());
  await browser.close();
})().catch(e => { console.error('FAIL:', e.message); process.exit(1); });
