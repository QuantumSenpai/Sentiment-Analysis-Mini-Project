/* ── Sakura Canvas Particles ── */
(function(){
  const cv  = document.getElementById('sakura-canvas');
  const ctx = cv.getContext('2d');
  let W, H, petals = [];

  function resize(){ W = cv.width = window.innerWidth; H = cv.height = window.innerHeight; }
  resize(); window.addEventListener('resize', resize);

  function mkPetal(){
    const s = Math.random()*6+3;
    return{
      x: Math.random()*W, y: -20, s,
      rot: Math.random()*Math.PI*2,
      scaleY: Math.random()*.5+.4,
      vx: (Math.random()-.5)*.7,
      vy: Math.random()*.65+.35,
      vrot: (Math.random()-.5)*.035,
      op: Math.random()*.4+.15,
      hue: Math.round(Math.random()*22+338)
    };
  }

  for(let i=0;i<42;i++){
    const p = mkPetal(); p.y = Math.random()*H; petals.push(p);
  }

  (function loop(){
    ctx.clearRect(0,0,W,H);
    for(const p of petals){
      p.x += p.vx + Math.sin(p.y*.011)*.3;
      p.y += p.vy; p.rot += p.vrot;
      if(p.y>H+30||p.x<-50||p.x>W+50){ Object.assign(p,mkPetal()); p.y=-20; }
      ctx.save();
      ctx.translate(p.x,p.y); ctx.rotate(p.rot); ctx.scale(1,p.scaleY);
      ctx.globalAlpha = p.op;
      ctx.beginPath();
      ctx.ellipse(0,0,p.s,p.s*.62,0,0,Math.PI*2);
      ctx.fillStyle = `hsl(${p.hue},78%,83%)`;
      ctx.fill();
      ctx.restore();
    }
    requestAnimationFrame(loop);
  })();
})();

/* ── File Handling ── */
const fileInput  = document.getElementById('fileInput');
const fileLabel  = document.querySelector('.file-label');
const fileStatus = document.getElementById('fileStatus');
const analyzeBtn = document.getElementById('analyzeBtn');

function handleFile(file){
  if(!file) return;
  if(!file.name.endsWith('.csv')){
    fileStatus.textContent = 'Only .csv files are supported.';
    fileStatus.className   = 'file-status error';
    analyzeBtn.disabled    = true;
    return;
  }
  fileStatus.textContent = '✓ ' + file.name;
  fileStatus.className   = 'file-status';
  analyzeBtn.disabled    = false;
}

fileInput.addEventListener('change', e => handleFile(e.target.files[0]));

fileLabel.addEventListener('dragover',  e => { e.preventDefault(); fileLabel.classList.add('drag-over'); });
fileLabel.addEventListener('dragleave', ()  => fileLabel.classList.remove('drag-over'));
fileLabel.addEventListener('drop', e => {
  e.preventDefault(); fileLabel.classList.remove('drag-over');
  const f = e.dataTransfer.files[0];
  if(f){ handleFile(f); }
});

/* ── Form Submit ── */
document.getElementById('uploadForm').addEventListener('submit', async function(e){
  e.preventDefault();
  const file = fileInput.files[0];
  if(!file){ return; }

  document.getElementById('loading').style.display = 'block';
  document.getElementById('results').style.display = 'none';

  const formData = new FormData();
  formData.append('file', file);

  try{
    const res  = await fetch('/upload', { method:'POST', body:formData });
    const data = await res.json();
    if(res.ok){ displayResults(data); }
    else{ alert('Error: ' + data.error); }
  } catch(err){
    alert('Error: ' + err.message);
  } finally{
    document.getElementById('loading').style.display = 'none';
  }
});

/* ── Display Results ── */
function displayResults(data){
  const total = data.total_reviews;
  const icons = { positive:'😊', neutral:'😐', negative:'😞' };
  const badgeClass = { positive:'badge-pos', neutral:'badge-neu', negative:'badge-neg' };

  // Metrics row
  const mRow = document.getElementById('metricsRow');
  if(mRow){
    mRow.innerHTML = [
      ['Total Reviews', total],
      ['Accuracy',      (data.accuracy*100).toFixed(1)+'%'],
      ['Positive',      ((data.sentiment_distribution.positive/total)*100).toFixed(0)+'%'],
      ['Negative',      ((data.sentiment_distribution.negative/total)*100).toFixed(0)+'%'],
    ].map(([l,v],i) => `
      <div class="metric-box" style="animation-delay:${i*.07}s">
        <div class="metric-label">${l}</div>
        <div class="metric-value">${v}</div>
      </div>`).join('');
  }

  // Sentiment table
  const sentTable = document.getElementById('sentTable');
  if(sentTable){
    sentTable.innerHTML =
      '<thead><tr><th>Sentiment</th><th>Count</th><th>Share</th></tr></thead><tbody>' +
      Object.entries(data.sentiment_distribution).map(([k,v]) => `
        <tr>
          <td><span class="badge ${badgeClass[k]}">${icons[k]} ${k}</span></td>
          <td>${v}</td>
          <td>${((v/total)*100).toFixed(1)}%</td>
        </tr>`).join('') + '</tbody>';
  }

  // Aspects table
  const aspectTable = document.getElementById('aspectTable');
  if(aspectTable && data.aspects){
    aspectTable.innerHTML =
      '<thead><tr><th>Aspect</th><th>Positive</th><th>Neutral</th><th>Negative</th></tr></thead><tbody>' +
      data.aspects.map(a => `
        <tr>
          <td style="font-weight:600">${a.aspect}</td>
          <td style="color:var(--green)">${a.positive}</td>
          <td style="color:var(--violet)">${a.neutral}</td>
          <td style="color:var(--red)">${a.negative}</td>
        </tr>`).join('') + '</tbody>';
  }

  // Legacy support: resultsContent div
  const legacyDiv = document.getElementById('resultsContent');
  if(legacyDiv && !mRow){
    legacyDiv.innerHTML = `
      <div class="metrics-row">
        <div class="metric-box"><div class="metric-label">Total Reviews</div><div class="metric-value">${total}</div></div>
        <div class="metric-box"><div class="metric-label">Accuracy</div><div class="metric-value">${(data.accuracy*100).toFixed(1)}%</div></div>
      </div>
      <div class="section-title">Sentiment Distribution</div>
      <table><thead><tr><th>Sentiment</th><th>Count</th><th>Share</th></tr></thead><tbody>
        ${Object.entries(data.sentiment_distribution).map(([k,v])=>`
          <tr>
            <td><span class="badge ${badgeClass[k]}">${icons[k]} ${k}</span></td>
            <td>${v}</td>
            <td>${((v/total)*100).toFixed(1)}%</td>
          </tr>`).join('')}
      </tbody></table>
      ${data.aspects ? `
        <div class="section-title" style="margin-top:24px">Top Aspects</div>
        <table><thead><tr><th>Aspect</th><th>Positive</th><th>Neutral</th><th>Negative</th></tr></thead><tbody>
          ${data.aspects.map(a=>`
            <tr>
              <td style="font-weight:600">${a.aspect}</td>
              <td style="color:var(--green)">${a.positive}</td>
              <td style="color:var(--violet)">${a.neutral}</td>
              <td style="color:var(--red)">${a.negative}</td>
            </tr>`).join('')}
        </tbody></table>` : ''}`;
  }

  const resultsEl = document.getElementById('results');
  resultsEl.style.display = 'block';
  resultsEl.scrollIntoView({ behavior:'smooth' });
}

/* ── Sample CSV Download ── */
function downloadSample(){
  const csv = `review_text,rating\nGreat product! Really satisfied,5\nTerrible experience,1\nAverage product,3\nExcellent quality,5\nPoor service,2\nGood value,4\nDisappointing,2\nAmazing product,5\nOkay delivery slow,3\nBest buy,5\nNot bad,3\nLoved it,5\nHorrible quality,1\nDecent price,4\nSuperb service,5`;
  const a = document.createElement('a');
  a.href = URL.createObjectURL(new Blob([csv],{type:'text/csv'}));
  a.download = 'sample_reviews.csv'; a.click();
}