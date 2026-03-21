import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import io, sys, os, numpy as np, copy

sys.path.append('.')
from src.model import TeacherCNN, StudentCNN
from src.quantization import apply_ptq
from torchvision import datasets
from torch.utils.data import DataLoader
from torchvision import transforms as T
from flask import Flask, request, jsonify, render_template_string

app = Flask(__name__)

# ── Load models ────────────────────────────────────────────
teacher = TeacherCNN()
teacher.load_state_dict(torch.load('results/teacher.pth', map_location='cpu'))
teacher.eval()

student = StudentCNN()
student.load_state_dict(torch.load('results/student_distilled.pth', map_location='cpu'))
student.eval()

def load_ptq():
    tf = T.Compose([T.ToTensor(), T.Normalize((0.1307,), (0.3081,))])
    dl = DataLoader(datasets.MNIST('./data', train=False, download=True, transform=tf), batch_size=64)
    return apply_ptq(copy.deepcopy(teacher), dl, 'cpu')

print("Loading models...")
ptq_model = load_ptq()
print("All models loaded!\n")

# ── Preprocessing ──────────────────────────────────────────
_tf = transforms.Compose([
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

def preprocess(img):
    img = img.convert('L')
    arr = np.array(img)
    if arr.mean() > 127:
        img = Image.fromarray(255 - arr)
    return _tf(img).unsqueeze(0)

def run_predict(model, tensor):
    with torch.no_grad():
        prob = F.softmax(model(tensor), dim=1)[0]
    return prob.argmax().item(), float(prob.max()) * 100

# ── HTML ───────────────────────────────────────────────────
HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>CompressionLab</title>
<link href="https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600&family=DM+Mono:wght@400;500&family=Playfair+Display:wght@600;700&display=swap" rel="stylesheet">
<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.min.js"></script>
<style>
:root{
  --bg:#F6F5F1;--surf:#FFF;--surf2:#EFEDE8;--brd:#DDD9D2;
  --tx:#17160F;--mt:#7A776E;
  --pu:#4A47A3;--pu2:#7B78D4;
  --am:#B87020;--gn:#1E7A50;--bl:#2060A8;--rd:#B83030;
  --sh:0 1px 3px rgba(0,0,0,.05),0 4px 14px rgba(0,0,0,.06);
  --sh2:0 8px 28px rgba(0,0,0,.11);
}
*{box-sizing:border-box;margin:0;padding:0;}
body{font-family:'DM Sans',sans-serif;background:var(--bg);color:var(--tx);min-height:100vh;font-size:14px;}

/* NAV */
nav{background:var(--surf);border-bottom:1px solid var(--brd);height:58px;
  display:flex;align-items:center;justify-content:space-between;
  padding:0 28px;position:sticky;top:0;z-index:500;box-shadow:0 1px 0 var(--brd);}
.logo{display:flex;align-items:center;gap:10px;}
.logo-ico{width:32px;height:32px;background:var(--pu);border-radius:8px;
  display:flex;align-items:center;justify-content:center;color:#fff;font-size:15px;}
.logo-txt{font-family:'Playfair Display',serif;font-size:16px;font-weight:600;}
.tabs{display:flex;gap:3px;}
.tab{padding:6px 15px;border-radius:7px;border:none;background:transparent;
  font-family:'DM Sans',sans-serif;font-size:13px;font-weight:500;
  color:var(--mt);cursor:pointer;transition:all .18s;}
.tab.on{background:var(--pu);color:#fff;}
.tab:not(.on):hover{background:var(--surf2);color:var(--tx);}
.nbadge{background:var(--surf2);border:1px solid var(--brd);padding:4px 12px;
  border-radius:99px;font-size:11px;color:var(--mt);font-family:'DM Mono',monospace;}

/* PAGES */
.pg{display:none;}
.pg.on{display:block;animation:fi .25s ease;}
@keyframes fi{from{opacity:0;transform:translateY(5px)}to{opacity:1;transform:none}}

/* HERO */
.hero{background:var(--surf);border-bottom:1px solid var(--brd);
  padding:44px 28px 36px;text-align:center;}
.ey{font-family:'DM Mono',monospace;font-size:10px;letter-spacing:2px;
  text-transform:uppercase;color:var(--pu);margin-bottom:12px;}
.hero h1{font-family:'Playfair Display',serif;font-size:34px;
  letter-spacing:-1px;line-height:1.2;margin-bottom:10px;}
.hero h1 em{color:var(--pu);font-style:normal;}
.hero p{color:var(--mt);font-size:14px;max-width:480px;margin:0 auto 28px;}
.sstrip{display:flex;border:1px solid var(--brd);border-radius:12px;
  overflow:hidden;max-width:580px;margin:0 auto;background:var(--surf);}
.ss{flex:1;padding:14px;text-align:center;border-right:1px solid var(--brd);}
.ss:last-child{border-right:none;}
.ssv{font-family:'Playfair Display',serif;font-size:22px;color:var(--pu);font-weight:700;}
.ssl{font-size:10px;color:var(--mt);font-family:'DM Mono',monospace;margin-top:2px;}

/* CONTAINER */
.wrap{max-width:1080px;margin:0 auto;padding:32px 20px 72px;}

/* SECTION LABEL */
.sl{font-family:'DM Mono',monospace;font-size:10px;letter-spacing:2px;
  text-transform:uppercase;color:var(--mt);margin-bottom:10px;}

/* CARD */
.card{background:var(--surf);border:1px solid var(--brd);border-radius:16px;
  padding:24px;box-shadow:var(--sh);transition:box-shadow .2s,transform .2s;}
.card:hover{box-shadow:var(--sh2);transform:translateY(-2px);}
.ct{font-family:'Playfair Display',serif;font-size:15px;margin-bottom:3px;}
.cs{font-size:11px;color:var(--mt);font-family:'DM Mono',monospace;margin-bottom:16px;}

/* GRIDS */
.g2{display:grid;grid-template-columns:1fr 1fr;gap:18px;margin-bottom:18px;}
.g3{display:grid;grid-template-columns:repeat(3,1fr);gap:14px;margin-bottom:18px;}
.g4{display:grid;grid-template-columns:repeat(4,1fr);gap:14px;margin-bottom:24px;}
@media(max-width:860px){.g4{grid-template-columns:1fr 1fr;}.g3{grid-template-columns:1fr 1fr;}}
@media(max-width:600px){.g2,.g3,.g4{grid-template-columns:1fr;}}

/* KPI */
.kpi{background:var(--surf);border:1px solid var(--brd);border-radius:13px;
  padding:18px 20px;box-shadow:var(--sh);position:relative;overflow:hidden;}
.kpi::before{content:'';position:absolute;top:0;left:0;right:0;height:3px;}
.kp{--c:var(--pu);} .kp::before{background:var(--pu);}
.ka{--c:var(--am);} .ka::before{background:var(--am);}
.kg{--c:var(--gn);} .kg::before{background:var(--gn);}
.kb{--c:var(--bl);} .kb::before{background:var(--bl);}
.kl{font-size:10px;color:var(--mt);font-family:'DM Mono',monospace;
  letter-spacing:1px;text-transform:uppercase;margin-bottom:7px;}
.kv{font-family:'Playfair Display',serif;font-size:30px;font-weight:700;color:var(--c);}
.ks{font-size:11px;color:var(--mt);margin-top:3px;}

/* TABLE */
.tbl{width:100%;border-collapse:collapse;font-size:13px;}
.tbl th{text-align:left;padding:9px 13px;font-family:'DM Mono',monospace;
  font-size:10px;letter-spacing:1px;text-transform:uppercase;color:var(--mt);
  border-bottom:2px solid var(--brd);}
.tbl td{padding:11px 13px;border-bottom:1px solid var(--brd);}
.tbl tr:last-child td{border-bottom:none;}
.tbl tr:hover td{background:var(--bg);}
.brow td{background:#F0FFF6 !important;font-weight:500;}
.pill{display:inline-block;padding:2px 9px;border-radius:5px;
  font-size:10px;font-family:'DM Mono',monospace;font-weight:500;}
.pp{background:#EEEEFF;color:var(--pu);border:1px solid #D4D6F5;}
.pa{background:#FFF5E8;color:var(--am);border:1px solid #F0DDB8;}
.pg_{background:#E8F5EF;color:var(--gn);border:1px solid #B8DFD0;}
.pb{background:#E8F0FF;color:var(--bl);border:1px solid #B8C8F0;}

/* BAR ROW */
.brow2{display:flex;align-items:center;gap:7px;}
.bbg{flex:1;background:var(--surf2);border-radius:99px;height:5px;overflow:hidden;}
.bfg{height:5px;border-radius:99px;}
.bnum{font-family:'DM Mono',monospace;font-size:10px;color:var(--mt);min-width:36px;text-align:right;}

/* UPLOAD */
.ucard{background:var(--surf);border:1px solid var(--brd);border-radius:18px;
  padding:36px;text-align:center;box-shadow:var(--sh);margin-bottom:20px;
  position:relative;overflow:hidden;}
.ucard::before{content:'';position:absolute;top:0;left:0;right:0;height:3px;
  background:linear-gradient(90deg,var(--pu),var(--pu2));}
.dz{border:2px dashed var(--brd);border-radius:12px;padding:32px 20px;
  cursor:pointer;transition:all .22s;background:var(--bg);display:block;}
.dz:hover{border-color:var(--pu2);background:#F0EFFE;}
.dz input{display:none;}
.dic{width:48px;height:48px;background:var(--surf2);border:1px solid var(--brd);
  border-radius:12px;display:flex;align-items:center;justify-content:center;
  margin:0 auto 12px;font-size:20px;}
.dt{font-size:14px;font-weight:500;margin-bottom:5px;}
.ds{font-size:11px;color:var(--mt);font-family:'DM Mono',monospace;}
#pvw{display:none;margin:18px auto 0;text-align:center;}
#pvw img{width:110px;height:110px;object-fit:contain;border-radius:10px;
  border:2px solid var(--brd);background:#111;display:block;margin:0 auto 6px;}
#pvw p{font-size:10px;color:var(--mt);font-family:'DM Mono',monospace;}
.rbtn{background:var(--pu);color:#fff;border:none;padding:12px 32px;
  border-radius:9px;font-size:14px;font-weight:600;font-family:'DM Sans',sans-serif;
  cursor:pointer;margin-top:20px;transition:all .2s;
  box-shadow:0 2px 8px rgba(74,71,163,.25);}
.rbtn:hover{background:#3A3790;transform:translateY(-1px);}

/* PRED */
.pgrid{display:none;grid-template-columns:repeat(3,1fr);gap:14px;}
@media(max-width:700px){.pgrid{grid-template-columns:1fr;}}
.pcrd{background:var(--surf);border:1px solid var(--brd);border-radius:16px;
  padding:22px 20px 20px;box-shadow:var(--sh);}
.pacc{height:3px;border-radius:99px;margin-bottom:16px;}
.plbl{font-family:'DM Mono',monospace;font-size:10px;letter-spacing:1.5px;
  text-transform:uppercase;color:var(--mt);margin-bottom:12px;}
.pdig{font-family:'Playfair Display',serif;font-size:76px;font-weight:700;
  line-height:1;margin-bottom:12px;}
.pcf{font-size:12px;color:var(--mt);margin-bottom:7px;}
.pcf span{font-weight:600;color:var(--tx);}
.pbg{background:var(--surf2);border-radius:99px;height:5px;overflow:hidden;margin-bottom:16px;}
.pfg{height:5px;border-radius:99px;transition:width 1s cubic-bezier(.4,0,.2,1);width:0%;}
.badges{display:flex;flex-wrap:wrap;gap:4px;}

/* SUMMARY */
.sbox{background:var(--surf);border:1px solid var(--brd);border-radius:16px;
  padding:22px;margin-top:16px;box-shadow:var(--sh);display:none;}
.sbt{font-family:'DM Mono',monospace;font-size:10px;letter-spacing:2px;
  text-transform:uppercase;color:var(--mt);margin-bottom:14px;}
.sgrid{display:grid;grid-template-columns:repeat(3,1fr);gap:12px;}
@media(max-width:600px){.sgrid{grid-template-columns:1fr;}}
.si{background:var(--bg);border:1px solid var(--brd);border-radius:10px;padding:13px;}
.sil{font-size:10px;color:var(--mt);font-family:'DM Mono',monospace;margin-bottom:4px;}
.siv{font-size:13px;font-weight:600;}
.tok{color:var(--gn);} .tno{color:var(--rd);}

/* SPINNER */
.spw{display:none;text-align:center;padding:26px;}
.sp{width:32px;height:32px;border:3px solid var(--brd);border-top-color:var(--pu);
  border-radius:50%;animation:spin .8s linear infinite;margin:0 auto 10px;}
@keyframes spin{to{transform:rotate(360deg)}}
.spw p{color:var(--mt);font-size:13px;}

/* ERROR / TIP */
.err{background:#FFF0EE;border:1px solid #F5C4BB;color:var(--rd);
  padding:11px 14px;border-radius:9px;margin-bottom:14px;display:none;font-size:13px;}
.tip{background:#EEF0FF;border:1px solid #D4D6F5;border-radius:9px;
  padding:11px 14px;font-size:13px;color:var(--pu);margin-bottom:20px;
  display:flex;gap:9px;align-items:flex-start;}

/* ARCH LAYERS */
.layer{border-radius:7px;padding:9px 13px;font-size:11px;
  font-family:'DM Mono',monospace;margin-bottom:6px;}
.lp{background:#EEEEFF;border:1px solid #D4D6F5;color:var(--pu);}
.lg{background:#E8F5EF;border:1px solid #B8DFD0;color:var(--gn);}

/* OBS CARDS */
.obs{padding:11px 13px;border-radius:9px;border-left:3px solid;margin-bottom:10px;}
.obs-t{font-size:12px;font-weight:600;margin-bottom:3px;}
.obs-d{font-size:12px;color:var(--mt);}

footer{border-top:1px solid var(--brd);padding:18px 28px;text-align:center;
  font-size:11px;color:var(--mt);font-family:'DM Mono',monospace;}
</style>
</head>
<body>

<nav>
  <div class="logo">
    <div class="logo-ico">&#129504;</div>
    <span class="logo-txt">CompressionLab</span>
  </div>
  <div class="tabs">
    <button class="tab on"  onclick="go('dash',this)">Dashboard</button>
    <button class="tab"     onclick="go('infer',this)">Live Inference</button>
    <button class="tab"     onclick="go('comp',this)">Comparison</button>
  </div>
  <div class="nbadge">Pradhi Bobade - Cummins College &middot; DL March-2026</div>
</nav>

<!-- ════════════ DASHBOARD ════════════ -->
<div class="pg on" id="pg-dash">

  <div class="hero">
    <div class="ey">Efficient Deep Learning</div>
    <h1>Model Compression&nbsp;<em>&amp; Quantization</em></h1>
    <p>Pruning &middot; PTQ &middot; QAT &middot; Knowledge Distillation &middot; MNIST</p>
    <div class="sstrip">
      <div class="ss"><div class="ssv">6</div><div class="ssl">Techniques</div></div>
      <div class="ss"><div class="ssv">99.49%</div><div class="ssl">Best accuracy</div></div>
      <div class="ss"><div class="ssv">63.6&times;</div><div class="ssl">Max compression</div></div>
      <div class="ss"><div class="ssv">0.43MB</div><div class="ssl">Smallest model</div></div>
    </div>
  </div>

  <div class="wrap">

    <p class="sl">Key Results</p>
    <div class="g4">
      <div class="kpi kp"><div class="kl">Teacher accuracy</div><div class="kv">99.34%</div><div class="ks">Baseline model</div></div>
      <div class="kpi ka"><div class="kl">Best QAT accuracy</div><div class="kv">99.49%</div><div class="ks">+0.15% vs baseline</div></div>
      <div class="kpi kg"><div class="kl">Max compression</div><div class="kv">63.6&times;</div><div class="ks">Student KD model</div></div>
      <div class="kpi kb"><div class="kl">Smallest model</div><div class="kv">0.43MB</div><div class="ks">vs 27.20MB original</div></div>
    </div>

    <p class="sl">Performance Charts</p>
    <div class="g2">
      <div class="card"><div class="ct">Accuracy Comparison</div><div class="cs">Test accuracy per method</div><canvas id="cAcc" height="210"></canvas></div>
      <div class="card"><div class="ct">Model Size (MB)</div><div class="cs">Saved file size per method</div><canvas id="cSz" height="210"></canvas></div>
    </div>
    <div class="g2">
      <div class="card"><div class="ct">Compression Ratio</div><div class="cs">Times smaller than original</div><canvas id="cRt" height="210"></canvas></div>
      <div class="card"><div class="ct">Accuracy vs Size Trade-off</div><div class="cs">Top-left = best trade-off</div><canvas id="cSc" height="210"></canvas></div>
    </div>

    <p class="sl">Full Results Table</p>
    <div class="card" style="padding:0;overflow:hidden;">
      <table class="tbl">
        <thead><tr><th>Method</th><th>Accuracy</th><th>Size&nbsp;(MB)</th><th>Compression</th><th>Accuracy Bar</th><th>Category</th></tr></thead>
        <tbody>
          <tr><td><strong>Original Teacher</strong></td><td>99.34%</td><td>27.20</td><td>1.0&times;</td><td><div class="brow2"><div class="bbg"><div class="bfg" style="width:99.34%;background:var(--pu)"></div></div><span class="bnum">99.34%</span></div></td><td><span class="pill pp">Baseline</span></td></tr>
          <tr><td><strong>Unstructured Pruning</strong></td><td>99.34%</td><td>27.20</td><td>1.0&times;</td><td><div class="brow2"><div class="bbg"><div class="bfg" style="width:99.34%;background:var(--pu)"></div></div><span class="bnum">99.34%</span></div></td><td><span class="pill pp">Pruning</span></td></tr>
          <tr><td><strong>Structured Pruning</strong></td><td>99.27%</td><td>27.20</td><td>1.0&times;</td><td><div class="brow2"><div class="bbg"><div class="bfg" style="width:99.27%;background:var(--pu)"></div></div><span class="bnum">99.27%</span></div></td><td><span class="pill pp">Pruning</span></td></tr>
          <tr><td><strong>PTQ (INT8)</strong></td><td>99.34%</td><td>7.91</td><td>3.4&times;</td><td><div class="brow2"><div class="bbg"><div class="bfg" style="width:99.34%;background:var(--am)"></div></div><span class="bnum">99.34%</span></div></td><td><span class="pill pa">Quantization</span></td></tr>
          <tr class="brow"><td><strong>QAT (INT8) &#11088;</strong></td><td>99.49%</td><td>7.91</td><td>3.4&times;</td><td><div class="brow2"><div class="bbg"><div class="bfg" style="width:99.49%;background:var(--gn)"></div></div><span class="bnum">99.49%</span></div></td><td><span class="pill pg_">Best Accuracy</span></td></tr>
          <tr><td><strong>Student KD &#128640;</strong></td><td>99.04%</td><td>0.43</td><td>63.6&times;</td><td><div class="brow2"><div class="bbg"><div class="bfg" style="width:99.04%;background:var(--bl)"></div></div><span class="bnum">99.04%</span></div></td><td><span class="pill pb">Best Size</span></td></tr>
        </tbody>
      </table>
    </div>

  </div>
</div>

<!-- ════════════ INFERENCE ════════════ -->
<div class="pg" id="pg-infer">
  <div class="wrap" style="padding-top:30px;">

    <div class="tip"><span>&#128161;</span><span>Upload a <strong>single handwritten digit (0–9)</strong>. White paper photos are auto-inverted. Models run simultaneously.</span></div>

    <p class="sl">Upload Image</p>
    <div class="ucard">
      <label class="dz" for="fi">
        <div class="dic">&#128193;</div>
        <div class="dt">Click to upload a digit image</div>
        <div class="ds">PNG &middot; JPG &middot; Single digit only</div>
        <input type="file" id="fi" accept="image/*">
      </label>
      <div id="pvw"><img id="prev" src="" alt=""><p>Uploaded image</p></div>
      <button class="rbtn" onclick="runModels()">&#9654; Run All 3 Models</button>
    </div>

    <div class="err" id="errBox"></div>
    <div class="spw" id="spnr"><div class="sp"></div><p>Running inference&hellip;</p></div>

    <p class="sl" id="plbl" style="display:none">Predictions</p>
    <div class="pgrid" id="pgrid">

      <div class="pcrd">
        <div class="pacc" style="background:linear-gradient(90deg,#4A47A3,#7B78D4)"></div>
        <div class="plbl">Original Teacher &middot; 27.20 MB</div>
        <div class="pdig" style="color:var(--pu)" id="p0">&mdash;</div>
        <div class="pcf">Confidence: <span id="c0">&mdash;</span></div>
        <div class="pbg"><div class="pfg" id="b0" style="background:linear-gradient(90deg,#4A47A3,#7B78D4)"></div></div>
        <div class="badges"><span class="pill pp">27.20 MB</span><span class="pill pp">99.34% acc</span></div>
      </div>

      <div class="pcrd">
        <div class="pacc" style="background:linear-gradient(90deg,#B87020,#E09B50)"></div>
        <div class="plbl">PTQ &middot; INT8 &middot; 7.91 MB</div>
        <div class="pdig" style="color:var(--am)" id="p1">&mdash;</div>
        <div class="pcf">Confidence: <span id="c1">&mdash;</span></div>
        <div class="pbg"><div class="pfg" id="b1" style="background:linear-gradient(90deg,#B87020,#E09B50)"></div></div>
        <div class="badges"><span class="pill pa">7.91 MB</span><span class="pill pa">3.4&times; smaller</span></div>
      </div>

      <div class="pcrd">
        <div class="pacc" style="background:linear-gradient(90deg,#1E7A50,#4DB87E)"></div>
        <div class="plbl">Student &middot; KD &middot; 0.43 MB</div>
        <div class="pdig" style="color:var(--gn)" id="p2">&mdash;</div>
        <div class="pcf">Confidence: <span id="c2">&mdash;</span></div>
        <div class="pbg"><div class="pfg" id="b2" style="background:linear-gradient(90deg,#1E7A50,#4DB87E)"></div></div>
        <div class="badges"><span class="pill pg_">0.43 MB</span><span class="pill pg_">63.6&times; smaller</span></div>
      </div>

    </div>

    <div class="sbox" id="sbox">
      <div class="sbt">Prediction Summary</div>
      <div class="sgrid">
        <div class="si"><div class="sil">All models agree?</div><div class="siv" id="sa">&mdash;</div></div>
        <div class="si"><div class="sil">Highest confidence</div><div class="siv" id="sb">&mdash;</div></div>
        <div class="si"><div class="sil">Student predicted</div><div class="siv" id="sc">&mdash;</div></div>
      </div>
    </div>

  </div>
</div>

<!-- ════════════ COMPARISON ════════════ -->
<div class="pg" id="pg-comp">
  <div class="wrap" style="padding-top:30px;">

    <p class="sl">Technique Deep Dive</p>
    <div class="g3">

      <div class="card">
        <div class="ct">&#9986; Pruning</div>
        <div class="cs">Structured vs Unstructured</div>
        <table class="tbl">
          <tr><th>Type</th><th>Amount</th><th>Accuracy</th></tr>
          <tr><td>Unstructured</td><td>40%</td><td>99.34%</td></tr>
          <tr><td>Structured</td><td>30%</td><td>99.27%</td></tr>
        </table>
        <div style="margin-top:14px;padding:10px 12px;background:var(--bg);border-radius:8px;font-size:12px;color:var(--mt);">
          <strong style="color:var(--tx)">Key insight:</strong> 40% weights removed — zero accuracy drop. Model is over-parameterized.
        </div>
      </div>

      <div class="card">
        <div class="ct">&#128200; Quantization</div>
        <div class="cs">PTQ vs QAT</div>
        <table class="tbl">
          <tr><th>Type</th><th>Size</th><th>Accuracy</th></tr>
          <tr><td>PTQ (INT8)</td><td>7.91MB</td><td>99.34%</td></tr>
          <tr class="brow"><td>QAT (INT8)</td><td>7.91MB</td><td><strong>99.49%</strong></td></tr>
        </table>
        <div style="margin-top:14px;padding:10px 12px;background:var(--bg);border-radius:8px;font-size:12px;color:var(--mt);">
          <strong style="color:var(--tx)">Key insight:</strong> QAT beats original — quantization noise acts as regularization.
        </div>
      </div>

      <div class="card">
        <div class="ct">&#127891; Distillation</div>
        <div class="cs">Teacher &rarr; Student</div>
        <table class="tbl">
          <tr><th>Model</th><th>Size</th><th>Accuracy</th></tr>
          <tr><td>Teacher</td><td>27.20MB</td><td>99.34%</td></tr>
          <tr class="brow"><td>Student</td><td>0.43MB</td><td>99.04%</td></tr>
        </table>
        <div style="margin-top:14px;padding:10px 12px;background:var(--bg);border-radius:8px;font-size:12px;color:var(--mt);">
          <strong style="color:var(--tx)">Key insight:</strong> 63.6&times; smaller, only 0.30% accuracy drop.
        </div>
      </div>

    </div>

    <p class="sl">Architecture Comparison</p>
    <div class="g2" style="margin-bottom:18px;">
      <div class="card">
        <div class="ct">Teacher CNN</div>
        <div class="cs">27.20 MB &middot; 6.9M parameters</div>
        <div class="layer lp">Conv2d(1 &rarr; 64) + ReLU + MaxPool</div>
        <div class="layer lp">Conv2d(64 &rarr; 128) + ReLU + MaxPool</div>
        <div class="layer lp">Conv2d(128 &rarr; 256) + ReLU</div>
        <div class="layer lp">Linear(12544 &rarr; 512) + Dropout</div>
        <div class="layer lp">Linear(512 &rarr; 10)</div>
      </div>
      <div class="card">
        <div class="ct">Student CNN</div>
        <div class="cs">0.43 MB &middot; 0.1M parameters &middot; 63.6&times; smaller</div>
        <div class="layer lg">Conv2d(1 &rarr; 16) + ReLU + MaxPool</div>
        <div class="layer lg">Conv2d(16 &rarr; 32) + ReLU + MaxPool</div>
        <div class="layer lg">Linear(1568 &rarr; 64)</div>
        <div class="layer lg">Linear(64 &rarr; 10)</div>
        <div style="margin-top:10px;padding:9px 12px;background:var(--bg);border-radius:7px;font-size:11px;color:var(--mt);">
          4 fewer layers &middot; 63.6&times; fewer parameters &middot; same task
        </div>
      </div>
    </div>

    <p class="sl">Overall Analysis</p>
    <div class="g2">
      <div class="card">
        <div class="ct">Technique Radar</div>
        <div class="cs">Accuracy &middot; Size &middot; Speed &middot; Ease &middot; Effort</div>
        <canvas id="cRdr" height="260"></canvas>
      </div>
      <div class="card">
        <div class="ct">Observations</div>
        <div class="cs">Key findings from experiments</div>
        <div class="obs" style="background:#EEEEFF;border-color:var(--pu)">
          <div class="obs-t" style="color:var(--pu)">Pruning</div>
          <div class="obs-d">40% weights removed with zero accuracy drop. Models are highly redundant.</div>
        </div>
        <div class="obs" style="background:#FFF5E8;border-color:var(--am)">
          <div class="obs-t" style="color:var(--am)">PTQ</div>
          <div class="obs-d">3.4&times; compression instantly. No retraining needed. Best for quick deployment.</div>
        </div>
        <div class="obs" style="background:#E8F5EF;border-color:var(--gn)">
          <div class="obs-t" style="color:var(--gn)">QAT &#11088;</div>
          <div class="obs-d">Beats original by +0.15%. Quantization noise acts as regularization reducing overfitting.</div>
        </div>
        <div class="obs" style="background:#E8F0FF;border-color:var(--bl)">
          <div class="obs-t" style="color:var(--bl)">Knowledge Distillation &#128640;</div>
          <div class="obs-d">63.6&times; smaller with only 0.30% drop. Best choice for mobile/edge deployment.</div>
        </div>
      </div>
    </div>

  </div>
</div>

<footer>Efficient Deep Learning &middot; Compression &amp; Quantization &middot; Pruning &middot; PTQ &middot; QAT &middot; Knowledge Distillation &middot; MNIST</footer>

<script>
// ── Navigation ────────────────────────────────────────
function go(id, btn) {
  document.querySelectorAll('.pg').forEach(p => p.classList.remove('on'));
  document.querySelectorAll('.tab').forEach(b => b.classList.remove('on'));
  document.getElementById('pg-' + id).classList.add('on');
  btn.classList.add('on');
  if (id === 'dash') buildDash();
  if (id === 'comp') buildRadar();
}

// ── Data ──────────────────────────────────────────────
const LBL  = ['Original','Unstructured','Structured','PTQ','QAT','Student KD'];
const ACC  = [99.34,99.34,99.27,99.34,99.49,99.04];
const SZ   = [27.20,27.20,27.20,7.91,7.91,0.43];
const RT   = SZ.map(s => parseFloat((27.20/s).toFixed(1)));
const CLR  = ['#4A47A3','#7B78D4','#9B8FD4','#B87020','#1E7A50','#2060A8'];
Chart.defaults.font.family = "'DM Sans',sans-serif";
Chart.defaults.color = '#7A776E';

const GOPTS = {
  plugins:{legend:{display:false}},
  animation:{duration:800},
  responsive:true,
};

let dashDone = false;
function buildDash() {
  if (dashDone) return; dashDone = true;

  new Chart(document.getElementById('cAcc'),{type:'bar',
    data:{labels:LBL,datasets:[{data:ACC,backgroundColor:CLR,borderRadius:5,borderSkipped:false}]},
    options:{...GOPTS,scales:{y:{min:98.5,max:99.8,grid:{color:'#EFEDE8'},ticks:{callback:v=>v+'%'}},x:{grid:{display:false}}}}
  });

  new Chart(document.getElementById('cSz'),{type:'bar',
    data:{labels:LBL,datasets:[{data:SZ,backgroundColor:CLR,borderRadius:5,borderSkipped:false}]},
    options:{...GOPTS,scales:{y:{grid:{color:'#EFEDE8'},ticks:{callback:v=>v+'MB'}},x:{grid:{display:false}}}}
  });

  new Chart(document.getElementById('cRt'),{type:'bar',
    data:{labels:LBL,datasets:[{data:RT,backgroundColor:CLR,borderRadius:5,borderSkipped:false}]},
    options:{...GOPTS,scales:{y:{grid:{color:'#EFEDE8'},ticks:{callback:v=>v+'x'}},x:{grid:{display:false}}}}
  });

  new Chart(document.getElementById('cSc'),{type:'scatter',
    data:{datasets:LBL.map((l,i)=>({
      label:l,data:[{x:SZ[i],y:ACC[i]}],
      backgroundColor:CLR[i],pointRadius:9,pointHoverRadius:12
    }))},
    options:{
      animation:{duration:800},responsive:true,
      plugins:{legend:{position:'bottom',labels:{boxWidth:9,font:{size:10}}}},
      scales:{
        x:{title:{display:true,text:'Model Size (MB)'},grid:{color:'#EFEDE8'}},
        y:{min:98.5,max:99.8,title:{display:true,text:'Accuracy (%)'},
          grid:{color:'#EFEDE8'},ticks:{callback:v=>v+'%'}}
      }
    }
  });
}

let radarDone = false;
function buildRadar() {
  if (radarDone) return; radarDone = true;
  new Chart(document.getElementById('cRdr'),{type:'radar',
    data:{
      labels:['Accuracy','Size\nReduction','Inference\nSpeed','Deploy\nEase','Train\nEffort'],
      datasets:[
        {label:'PTQ',        data:[90,70,85,95,90],borderColor:'#B87020',backgroundColor:'rgba(184,112,32,.1)',pointBackgroundColor:'#B87020'},
        {label:'QAT',        data:[95,70,85,85,60],borderColor:'#1E7A50',backgroundColor:'rgba(30,122,80,.1)', pointBackgroundColor:'#1E7A50'},
        {label:'Student KD', data:[88,99,98,92,55],borderColor:'#2060A8',backgroundColor:'rgba(32,96,168,.1)', pointBackgroundColor:'#2060A8'},
        {label:'Pruning',    data:[92,20,50,80,75],borderColor:'#4A47A3',backgroundColor:'rgba(74,71,163,.1)', pointBackgroundColor:'#4A47A3'},
      ]
    },
    options:{
      animation:{duration:800},responsive:true,
      plugins:{legend:{position:'bottom',labels:{boxWidth:9,font:{size:11}}}},
      scales:{r:{min:0,max:100,grid:{color:'#EFEDE8'},ticks:{display:false},pointLabels:{font:{size:11}}}}
    }
  });
}

window.addEventListener('load', buildDash);

// ── Inference ─────────────────────────────────────────
document.getElementById('fi').onchange = function(e){
  const f = e.target.files[0]; if(!f) return;
  const r = new FileReader();
  r.onload = ev => {
    document.getElementById('prev').src = ev.target.result;
    document.getElementById('pvw').style.display = 'block';
  };
  r.readAsDataURL(f);
};

async function runModels(){
  const f = document.getElementById('fi').files[0];
  if(!f){ showErr('Please select an image first!'); return; }

  document.getElementById('errBox').style.display = 'none';
  document.getElementById('spnr').style.display   = 'block';
  document.getElementById('pgrid').style.display  = 'none';
  document.getElementById('sbox').style.display   = 'none';
  document.getElementById('plbl').style.display   = 'none';

  const fd = new FormData(); fd.append('image', f);
  try {
    const res  = await fetch('/predict',{method:'POST',body:fd});
    const data = await res.json();
    if(data.error){ showErr(data.error); return; }

    const keys  = ['teacher','ptq','student'];
    const names = ['Original Teacher','PTQ (INT8)','Student (KD)'];
    const preds = [], confs = [];

    keys.forEach((m,i)=>{
      preds.push(data[m].prediction);
      confs.push(data[m].confidence);
      document.getElementById('p'+i).textContent = data[m].prediction;
      document.getElementById('c'+i).textContent = data[m].confidence.toFixed(1)+'%';
      setTimeout(()=>{
        document.getElementById('b'+i).style.width = Math.min(data[m].confidence,100)+'%';
      }, 200+i*150);
    });

    const agree  = preds.every(p=>p===preds[0]);
    const bestI  = confs.indexOf(Math.max(...confs));
    document.getElementById('sa').innerHTML = agree
      ? '<span class="tok">&#10003; Yes &mdash; all predicted '+preds[0]+'</span>'
      : '<span class="tno">&#10007; No &mdash; '+preds.join(' / ')+'</span>';
    document.getElementById('sb').textContent = names[bestI]+' \xb7 '+confs[bestI].toFixed(1)+'%';
    document.getElementById('sc').textContent = 'Digit '+preds[2]+' \xb7 '+confs[2].toFixed(1)+'%';

    document.getElementById('spnr').style.display  = 'none';
    document.getElementById('pgrid').style.display = 'grid';
    document.getElementById('sbox').style.display  = 'block';
    document.getElementById('plbl').style.display  = 'block';
  } catch(e){ showErr('Server error: '+e.message); }
}

function showErr(msg){
  document.getElementById('spnr').style.display = 'none';
  const b = document.getElementById('errBox');
  b.textContent = '\u26a0 '+msg; b.style.display = 'block';
}
</script>
</body>
</html>
"""

@app.route('/')
def index():
    return render_template_string(HTML)

@app.route('/predict', methods=['POST'])
def predict_route():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'})
    try:
        img    = Image.open(io.BytesIO(request.files['image'].read()))
        tensor = preprocess(img)
        t_p, t_c = run_predict(teacher,   tensor)
        p_p, p_c = run_predict(ptq_model, tensor)
        s_p, s_c = run_predict(student,   tensor)
        return jsonify({
            'teacher': {'prediction': t_p, 'confidence': t_c},
            'ptq'    : {'prediction': p_p, 'confidence': p_c},
            'student': {'prediction': s_p, 'confidence': s_c},
        })
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    print("\n" + "="*50)
    print("  CompressionLab running!")
    print("  Open: http://127.0.0.1:5000")
    print("="*50 + "\n")

    from pyngrok import ngrok
    url = ngrok.connect(5000)
    print(f"\n  Public URL: {url}\n")


    app.run(debug=False, port=5000)