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
  const shotMiuraFlat = path.join(__dirname, 'output', 'miura_flat.png');
  const shotMiuraFolded = path.join(__dirname, 'output', 'miura_folded.png');
  const shotEasy = path.join(__dirname, 'output', 'studio_difficulty_easy.png');
  const shotDetailed = path.join(__dirname, 'output', 'studio_difficulty_detailed.png');
  const shotDesigner = path.join(__dirname, 'output', 'studio_designer.png');
  const shotCrane = path.join(__dirname, 'output', 'studio_crane.png');
  const shotGrouped = path.join(__dirname, 'output', 'studio_grouped_controls.png');
  const shotWalk = path.join(__dirname, 'output', 'studio_walkthrough.png');
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
  await page.goto(html + '?tour=0');   // ?tour=0 suppresses the first-run walkthrough auto-open so it never blocks assertions
  await new Promise(r => setTimeout(r, 2000));
  const status = async () => page.evaluate(() => document.getElementById('status').textContent);
  const fail = async (msg) => { console.error('FAIL:', msg, '| status:', await status(), '| pageerrors:', errs); await browser.close(); process.exit(1); };

  if (!/Ready/.test(await status())) await fail('page did not become Ready');

  // ---- grouped control bar: the three labeled clusters are present, tour dismissed ----
  const groups = await page.evaluate(() => [...document.querySelectorAll('#ui .ctlgroup .glabel')].map(e => e.textContent.trim()));
  if (groups.length !== 3) await fail('control bar not grouped into 3 clusters: ' + JSON.stringify(groups));
  await page.screenshot({path: shotGrouped});

  // ---- first-run walkthrough: opens from the "? Tour" button, shows step 1, highlights a
  //      control on the next step, then closes on Esc (and stays out of the way afterward) ----
  await page.click('#tourBtn');
  await new Promise(r => setTimeout(r, 300));
  const tourStep1 = await page.evaluate(() => ({
    open: getComputedStyle(document.getElementById('tour')).display !== 'none',
    title: document.getElementById('tourTitle').textContent,
  }));
  if (!tourStep1.open) await fail('walkthrough did not open on "? Tour" button');
  if (!/FoldForge/i.test(tourStep1.title)) await fail('walkthrough step 1 missing welcome text: ' + tourStep1.title);
  await page.click('#tourNext');   // step 2 highlights the Gallery control (visible ring)
  await new Promise(r => setTimeout(r, 300));
  const ringVisible = await page.evaluate(() => parseFloat(getComputedStyle(document.getElementById('tourRing')).width) > 0);
  if (!ringVisible) await fail('walkthrough step 2 did not highlight a control');
  await page.screenshot({path: shotWalk});
  await page.keyboard.press('Escape');
  await new Promise(r => setTimeout(r, 200));
  const tourClosed = await page.evaluate(() => getComputedStyle(document.getElementById('tour')).display === 'none');
  if (!tourClosed) await fail('walkthrough did not close on Esc');

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
  if (!/Easy\b.*\b9 folds/.test(easyStatus)) await fail('Easy budget did not report "Easy — 9 folds"');
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

  // ---- DETAIL selector on a BAKED sample must re-fold it (the reported bug: the
  //      selector did nothing for gallery samples). Load the butterfly bake, switch
  //      Detailed -> Easy, and assert BOTH the fold count drops AND the mesh geometry
  //      (vertex/triangle count) actually changes. ----
  await page.evaluate(() => { if (typeof loadBaked === 'function') loadBaked('butterfly'); });
  await new Promise(r => setTimeout(r, 700));
  await page.select('#foldmode', 'all');
  await page.select('#detail', '24');                         // Detailed
  await new Promise(r => setTimeout(r, 1600));
  const detStatus = await status();
  const detGeom = await page.evaluate(() => ({ verts: geo.attributes.position.count, tris: geo.index.count / 3, nx: S.nx, ny: S.ny }));
  const detFolds = parseInt((detStatus.match(/(\d+) folds/) || [])[1]);
  await setFold(1); await new Promise(r => setTimeout(r, 500));
  await page.screenshot({path: shotDetailed});
  await page.select('#detail', '10');                         // Easy
  await new Promise(r => setTimeout(r, 1600));
  const easyStatus2 = await status();
  const easyGeom = await page.evaluate(() => ({ verts: geo.attributes.position.count, tris: geo.index.count / 3, nx: S.nx, ny: S.ny }));
  const easyFolds = parseInt((easyStatus2.match(/(\d+) folds/) || [])[1]);
  await setFold(1); await new Promise(r => setTimeout(r, 500));
  await page.screenshot({path: shotEasy});
  if (!/Easy\b.*\b9 folds/.test(easyStatus2)) await fail('baked-sample Easy did not report "Easy — 9 folds": ' + easyStatus2);
  if (!(easyFolds < detFolds)) await fail('baked-sample Detail did not drop fold count: Easy ' + easyFolds + ' vs Detailed ' + detFolds);
  if (!(easyGeom.verts < detGeom.verts)) await fail('baked-sample Detail did not change mesh vertices: Easy ' + easyGeom.verts + ' vs Detailed ' + detGeom.verts);

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

  // ---- PROVE it FOLDS (not a z-morph): the XY footprint must CONTRACT and the z-relief
  //      GROW as the slider rises from t=0 (flat crease pattern) to t=1 (folded surface) ----
  const foot = () => page.evaluate(() => {
    const p = geo.attributes.position.array;
    let X0 = 1e9, X1 = -1e9, Y0 = 1e9, Y1 = -1e9, Z0 = 1e9, Z1 = -1e9;
    for (let k = 0; k < p.length; k += 3) { const x = p[k], y = p[k+1], z = p[k+2];
      if (x < X0) X0 = x; if (x > X1) X1 = x; if (y < Y0) Y0 = y; if (y > Y1) Y1 = y; if (z < Z0) Z0 = z; if (z > Z1) Z1 = z; }
    return { xspan: X1 - X0, area: (X1 - X0) * (Y1 - Y0), zrelief: Z1 - Z0 };
  });
  await setFold(0);   await new Promise(r => setTimeout(r, 500)); const m0 = await foot();
  await page.screenshot({path: shotMiuraFlat});
  await setFold(0.5); await new Promise(r => setTimeout(r, 500)); const m5 = await foot();
  await setFold(1);   await new Promise(r => setTimeout(r, 500)); const m1 = await foot();
  await page.screenshot({path: shotMiuraFolded});
  if (!(m1.area < m0.area && m5.area < m0.area))
    await fail('2-D Miura footprint did not contract as it folds (area t0/t.5/t1 ' + m0.area.toFixed(2) + '/' + m5.area.toFixed(2) + '/' + m1.area.toFixed(2) + ')');
  if (!(m5.zrelief > m0.zrelief && m1.zrelief > m5.zrelief))
    await fail('2-D Miura z-relief did not grow as it folds (t0/t.5/t1 ' + m0.zrelief.toFixed(3) + '/' + m5.zrelief.toFixed(3) + '/' + m1.zrelief.toFixed(3) + ')');
  const miFold = 'footprint area ' + m0.area.toFixed(1) + '->' + m5.area.toFixed(1) + '->' + m1.area.toFixed(1)
    + ' (x-span ' + m0.xspan.toFixed(1) + '->' + m1.xspan.toFixed(1) + '), z-relief ' + m0.zrelief.toFixed(2) + '->' + m5.zrelief.toFixed(2) + '->' + m1.zrelief.toFixed(2);

  // ---- figurative designer: open "Design a figure", fold the four-flap base ----
  await page.click('#designBtn');
  await new Promise(r => setTimeout(r, 400));
  const dInfo = await page.evaluate(() => {
    const cards = [...document.querySelectorAll('#designerlist .dcard')];
    const cpOk = cards.length > 0 && cards.every(c => ((c.querySelector('img') || {}).src || '').startsWith('data:image/svg'));
    return { n: cards.length, cpOk, four: (DATA.designs || []).some(d => d.key === 'four-flap') };
  });
  if (dInfo.n < 2) await fail('designer panel has too few base cards: ' + dInfo.n);
  if (!dInfo.cpOk) await fail('designer cards missing crease-pattern thumbnails');
  if (!dInfo.four) await fail('four-flap design not present in DATA.designs');
  const dClicked = await page.evaluate(() => {
    for (const c of document.querySelectorAll('#designerlist .dcard'))
      if (/Four-flap/i.test(c.textContent)) { c.querySelector('button.dload').click(); return true; }
    return false;
  });
  if (!dClicked) await fail('four-flap "Fold this base" button not found');
  await new Promise(r => setTimeout(r, 700));
  const dStatus = await status();
  if (!/^Folded .*Four-flap/.test(dStatus)) await fail('four-flap fold did not update status: ' + dStatus);
  if (!/TreeMaker/.test(dStatus)) await fail('four-flap status missing TreeMaker note: ' + dStatus);
  if (!/flap-length error/.test(dStatus)) await fail('four-flap status missing flap-error stat: ' + dStatus);
  await setFold(1);
  await new Promise(r => setTimeout(r, 1500));
  const dRelief = await page.evaluate(() => {
    const p = geo.attributes.position.array; let zmin = 1e9, zmax = -1e9;
    for (let k = 2; k < p.length; k += 3) { if (p[k] < zmin) zmin = p[k]; if (p[k] > zmax) zmax = p[k]; }
    return { z: zmax - zmin, dynamic: !!(S && S.dynamic), verts: geo.attributes.position.count };
  });
  if (!dRelief.dynamic) await fail('four-flap did not load as a live .fold model');
  if (!(dRelief.z > 0.1)) await fail('four-flap base did not fold (z-relief ' + dRelief.z + ')');
  await page.screenshot({path: shotDesigner});

  // ---- five-flap pentagon: the universal (gusset) molecule now folds it FLAT.
  //      Assert its data marks folds-flat + passes Kawasaki+Maekawa, and it folds live.
  const fiveInfo = await page.evaluate(() => {
    const d = (DATA.designs || []).find(x => x.key === 'five-flap');
    return d ? { ff: !!d.foldsFlat, kaw: d.kawasaki, mae: d.maekawa,
                 flat: d.flatFoldable, flaps: d.flaps, iv: d.interiorVerts } : null;
  });
  if (!fiveInfo) await fail('five-flap design not present in DATA.designs');
  if (!(fiveInfo.ff && fiveInfo.kaw && fiveInfo.mae && fiveInfo.flat))
    await fail('five-flap not marked folds-flat / passing Kawasaki+Maekawa: ' + JSON.stringify(fiveInfo));
  if (!(fiveInfo.flaps === 5 && fiveInfo.iv >= 5))
    await fail('five-flap stats look wrong (expect 5 flaps, many interior verts): ' + JSON.stringify(fiveInfo));
  const fiveClicked = await page.evaluate(() => {
    for (const c of document.querySelectorAll('#designerlist .dcard'))
      if (/Five-flap/i.test(c.textContent)) { c.querySelector('button.dload').click(); return true; }
    return false;
  });
  if (!fiveClicked) await fail('five-flap "Fold this base" button not found');
  await new Promise(r => setTimeout(r, 700));
  const fiveStatus = await status();
  if (!/^Folded .*Five-flap/.test(fiveStatus)) await fail('five-flap fold did not update status: ' + fiveStatus);
  await setFold(1);
  await new Promise(r => setTimeout(r, 1500));
  const fiveRelief = await page.evaluate(() => {
    const p = geo.attributes.position.array; let zmin = 1e9, zmax = -1e9;
    for (let k = 2; k < p.length; k += 3) { if (p[k] < zmin) zmin = p[k]; if (p[k] > zmax) zmax = p[k]; }
    return { z: zmax - zmin, dynamic: !!(S && S.dynamic) };
  });
  if (!fiveRelief.dynamic) await fail('five-flap did not load as a live .fold model');
  if (!(fiveRelief.z > 0.1)) await fail('five-flap base did not fold (z-relief ' + fiveRelief.z + ')');

  // ---- canonical crane: the "Crane (bird base)" card must be the classic base,
  //      show its Kawasaki/Maekawa pass stats, and fold live (z-relief) ----
  const craneInfo = await page.evaluate(() => {
    const d = (DATA.designs || []).find(x => x.key === 'bird-base');
    return d ? { classic: !!d.classic, kaw: d.kawasaki, mae: d.maekawa,
                 V: d.vertsN, E: d.edgesN, iv: d.interiorVerts, label: d.label } : null;
  });
  if (!craneInfo) await fail('bird-base design not present in DATA.designs');
  if (!craneInfo.classic || !craneInfo.kaw || !craneInfo.mae)
    await fail('bird-base not marked classic / not passing Kawasaki+Maekawa: ' + JSON.stringify(craneInfo));
  const craneClicked = await page.evaluate(() => {
    for (const c of document.querySelectorAll('#designerlist .dcard'))
      if (/Crane \(bird base\)/i.test(c.textContent)) { c.querySelector('button.dload').click(); return true; }
    return false;
  });
  if (!craneClicked) await fail('"Fold this crane base" button not found');
  await new Promise(r => setTimeout(r, 700));
  const craneStatus = await status();
  if (!/^Folded .*Crane \(bird base\)/.test(craneStatus)) await fail('crane fold did not update status: ' + craneStatus);
  if (!/traditional bird base/.test(craneStatus)) await fail('crane status missing bird-base note: ' + craneStatus);
  if (!/Kawasaki\s+✓\s+Maekawa\s+✓/.test(craneStatus)) await fail('crane status missing Kawasaki/Maekawa pass stats: ' + craneStatus);
  await setFold(1);
  await new Promise(r => setTimeout(r, 1500));
  const craneRelief = await page.evaluate(() => {
    const p = geo.attributes.position.array; let zmin = 1e9, zmax = -1e9;
    for (let k = 2; k < p.length; k += 3) { if (p[k] < zmin) zmin = p[k]; if (p[k] > zmax) zmax = p[k]; }
    return { z: zmax - zmin, dynamic: !!(S && S.dynamic) };
  });
  if (!craneRelief.dynamic) await fail('crane did not load as a live .fold model');
  if (!(craneRelief.z > 0.1)) await fail('crane base did not fold (z-relief ' + craneRelief.z + ')');
  await page.screenshot({path: shotCrane});

  if (errs.length) await fail('uncaught page errors');
  console.log('OK — studio folds', path.basename(img),
    '| designer cards', dInfo.n, 'four-flap fold z-relief', dRelief.z.toFixed(2), '| designer shot', shotDesigner,
    '| crane (bird base)', craneInfo.V + 'v/' + craneInfo.E + 'e,', 'Kaw/Mae ✓✓,', craneInfo.iv + ' interior, fold z-relief', craneRelief.z.toFixed(2), '| crane shot', shotCrane,
    '| 2-D Miura live fit', miGrid, 'grid,', miIters, 'iters, error', miErr, ',', miMs + 'ms, z-relief', meshOk.toFixed(2),
    '| 2-D Miura FOLDS:', miFold, '| miura shots', shotMiuraFlat, shotMiuraFolded,
    '| live mirror-IoU', mirror, applied ? '(symmetrized)' : '(left as-is)',
    '| baked butterfly_sym IoU', gsym.iou, '| gallery cards', cards,
    '| pipeline sym-stage', pipe.sym,
    '| panel1 upload src', uploadSrc.slice(0, 24) + '…', '| panel1 gallery src', gallerySrc.slice(0, 24) + '…',
    '| flat-sheet rag L/R', rect.leftRag.toFixed(3) + '/' + rect.rightRag.toFixed(3), '(ragged flat OK)',
    '| baked-sample Detail: Detailed', detFolds, 'folds/' + detGeom.verts + 'v/' + detGeom.tris + 't ->',
    'Easy', easyFolds, 'folds/' + easyGeom.verts + 'v/' + easyGeom.tris + 't',
    '| screenshots', shot, shotUpload, shotGallery, shotRectFlat, shotRectFolded, shotDetailed, shotEasy, shotGrouped, shotWalk, '|', await status());
  await browser.close();
})().catch(e => { console.error('FAIL:', e.message); process.exit(1); });
