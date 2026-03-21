import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import io
import sys
import os
import numpy as np
import copy

sys.path.append('.')
from src.model import TeacherCNN, StudentCNN
from src.quantization import apply_ptq, get_model_size_mb
from torchvision import datasets
from torch.utils.data import DataLoader
from torchvision import transforms as T
from flask import Flask, request, jsonify, render_template_string

app = Flask(__name__)

# ── Load all models once at startup ───────────────────────
device = torch.device('cpu')

# Teacher
teacher = TeacherCNN()
teacher.load_state_dict(torch.load('results/teacher.pth', map_location='cpu'))
teacher.eval()

# Student
student = StudentCNN()
student.load_state_dict(torch.load('results/student_distilled.pth', map_location='cpu'))
student.eval()

# PTQ model
def load_ptq():
    transform = T.Compose([T.ToTensor(),
                            T.Normalize((0.1307,), (0.3081,))])
    cal_data   = datasets.MNIST('./data', train=False,
                                download=True, transform=transform)
    cal_loader = DataLoader(cal_data, batch_size=64)
    ptq = apply_ptq(copy.deepcopy(teacher), cal_loader, 'cpu')
    return ptq

print("Loading models...")
ptq_model = load_ptq()
print("All models loaded!\n")

# ── Preprocessing ──────────────────────────────────────────
base_transform = transforms.Compose([
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

def preprocess(img):
    """Auto-invert if white background. MNIST = white digit on black bg."""
    img       = img.convert('L')
    img_array = np.array(img)
    if img_array.mean() > 127:
        img_array = 255 - img_array
        img       = Image.fromarray(img_array)
    return base_transform(img).unsqueeze(0)

# ── Inference ──────────────────────────────────────────────
def run_predict(model, tensor):
    with torch.no_grad():
        out  = model(tensor)
        prob = F.softmax(out, dim=1)[0]
        pred = prob.argmax().item()
    return pred, float(prob[pred]) * 100

# ── HTML ───────────────────────────────────────────────────
HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>DL Model Compression — Live Demo</title>
    <link href="https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600&family=DM+Mono:wght@400;500&family=Playfair+Display:wght@700&display=swap" rel="stylesheet">
    <style>
        :root {
            --bg:        #F7F6F2;
            --surface:   #FFFFFF;
            --surface2:  #F0EEE8;
            --border:    #E2DDD6;
            --text:      #1A1916;
            --muted:     #7A7670;
            --accent:    #4A47A3;
            --accent2:   #7B78D4;
            --green:     #2D8A5E;
            --amber:     #C07C2A;
            --red:       #C0392B;
            --shadow:    0 1px 3px rgba(0,0,0,0.06), 0 4px 16px rgba(0,0,0,0.05);
            --shadow-lg: 0 8px 32px rgba(0,0,0,0.10);
        }

        * { box-sizing: border-box; margin: 0; padding: 0; }

        body {
            font-family: 'DM Sans', sans-serif;
            background: var(--bg);
            color: var(--text);
            min-height: 100vh;
            line-height: 1.6;
        }

        /* ── Header ── */
        .header {
            background: var(--surface);
            border-bottom: 1px solid var(--border);
            padding: 0 40px;
            display: flex;
            align-items: center;
            justify-content: space-between;
            height: 64px;
            position: sticky;
            top: 0;
            z-index: 100;
        }
        .logo {
            display: flex;
            align-items: center;
            gap: 12px;
        }
        .logo-icon {
            width: 36px; height: 36px;
            background: var(--accent);
            border-radius: 10px;
            display: flex; align-items: center; justify-content: center;
            color: white; font-size: 18px;
        }
        .logo-text {
            font-family: 'Playfair Display', serif;
            font-size: 18px;
            color: var(--text);
        }
        .header-badge {
            background: var(--surface2);
            border: 1px solid var(--border);
            padding: 5px 14px;
            border-radius: 99px;
            font-size: 12px;
            color: var(--muted);
            font-family: 'DM Mono', monospace;
        }

        /* ── Hero ── */
        .hero {
            background: var(--surface);
            border-bottom: 1px solid var(--border);
            padding: 56px 40px 48px;
            text-align: center;
        }
        .hero-eyebrow {
            font-family: 'DM Mono', monospace;
            font-size: 11px;
            color: var(--accent);
            letter-spacing: 2px;
            text-transform: uppercase;
            margin-bottom: 16px;
        }
        .hero h1 {
            font-family: 'Playfair Display', serif;
            font-size: 42px;
            font-weight: 700;
            color: var(--text);
            letter-spacing: -1px;
            line-height: 1.2;
            margin-bottom: 16px;
        }
        .hero h1 span { color: var(--accent); }
        .hero p {
            color: var(--muted);
            font-size: 16px;
            max-width: 520px;
            margin: 0 auto;
        }

        /* ── Stats row ── */
        .stats-row {
            display: flex;
            justify-content: center;
            margin-top: 40px;
            border: 1px solid var(--border);
            border-radius: 14px;
            overflow: hidden;
            max-width: 600px;
            margin-left: auto;
            margin-right: auto;
            background: var(--surface);
        }
        .stat {
            flex: 1;
            padding: 18px 20px;
            text-align: center;
            border-right: 1px solid var(--border);
        }
        .stat:last-child { border-right: none; }
        .stat-val {
            font-family: 'Playfair Display', serif;
            font-size: 26px;
            color: var(--accent);
            font-weight: 700;
        }
        .stat-lbl {
            font-size: 11px;
            color: var(--muted);
            margin-top: 2px;
            font-family: 'DM Mono', monospace;
        }

        /* ── Main ── */
        .container {
            max-width: 1000px;
            margin: 0 auto;
            padding: 40px 24px 80px;
        }

        .section-label {
            font-family: 'DM Mono', monospace;
            font-size: 10px;
            letter-spacing: 2px;
            text-transform: uppercase;
            color: var(--muted);
            margin-bottom: 12px;
        }

        /* ── Upload ── */
        .upload-card {
            background: var(--surface);
            border: 1px solid var(--border);
            border-radius: 20px;
            padding: 40px;
            text-align: center;
            box-shadow: var(--shadow);
            margin-bottom: 32px;
            position: relative;
            overflow: hidden;
        }
        .upload-card::before {
            content: '';
            position: absolute;
            top: 0; left: 0; right: 0;
            height: 3px;
            background: linear-gradient(90deg, var(--accent), var(--accent2));
        }

        .drop-zone {
            border: 2px dashed var(--border);
            border-radius: 14px;
            padding: 36px 24px;
            cursor: pointer;
            transition: all 0.25s;
            background: var(--bg);
            display: block;
        }
        .drop-zone:hover {
            border-color: var(--accent2);
            background: #F0EFFE;
        }
        .drop-zone input { display: none; }

        .upload-icon {
            width: 56px; height: 56px;
            background: var(--surface2);
            border: 1px solid var(--border);
            border-radius: 16px;
            display: flex; align-items: center; justify-content: center;
            margin: 0 auto 16px;
            font-size: 24px;
        }
        .drop-title {
            font-size: 15px;
            font-weight: 500;
            color: var(--text);
            margin-bottom: 6px;
        }
        .drop-sub {
            font-size: 12px;
            color: var(--muted);
            font-family: 'DM Mono', monospace;
        }

        #preview-wrap {
            display: none;
            margin: 20px auto 0;
            text-align: center;
        }
        #preview {
            width: 120px; height: 120px;
            object-fit: contain;
            border-radius: 12px;
            border: 2px solid var(--border);
            background: #111;
            display: block;
            margin: 0 auto 8px;
        }
        .preview-label {
            font-size: 11px;
            color: var(--muted);
            font-family: 'DM Mono', monospace;
        }

        .run-btn {
            background: var(--accent);
            color: white;
            border: none;
            padding: 14px 40px;
            border-radius: 10px;
            font-size: 15px;
            font-weight: 600;
            font-family: 'DM Sans', sans-serif;
            cursor: pointer;
            margin-top: 24px;
            transition: all 0.2s;
            box-shadow: 0 2px 8px rgba(74,71,163,0.25);
        }
        .run-btn:hover {
            background: #3A3790;
            transform: translateY(-1px);
            box-shadow: 0 4px 16px rgba(74,71,163,0.35);
        }

        /* ── Tip ── */
        .tip {
            background: #EEF0FF;
            border: 1px solid #D4D6F5;
            border-radius: 10px;
            padding: 12px 16px;
            font-size: 13px;
            color: var(--accent);
            margin-bottom: 28px;
            display: flex;
            align-items: flex-start;
            gap: 10px;
        }

        /* ── Loading ── */
        .loading {
            display: none;
            text-align: center;
            padding: 32px;
        }
        .spinner {
            width: 36px; height: 36px;
            border: 3px solid var(--border);
            border-top-color: var(--accent);
            border-radius: 50%;
            animation: spin 0.8s linear infinite;
            margin: 0 auto 12px;
        }
        @keyframes spin { to { transform: rotate(360deg); } }
        .loading p { color: var(--muted); font-size: 14px; }

        /* ── Error ── */
        .error {
            background: #FFF0EE;
            border: 1px solid #F5C4BB;
            color: var(--red);
            padding: 12px 16px;
            border-radius: 10px;
            margin-bottom: 20px;
            display: none;
            font-size: 14px;
        }

        /* ── Results ── */
        .results {
            display: none;
            grid-template-columns: repeat(3, 1fr);
            gap: 16px;
        }

        .card {
            background: var(--surface);
            border: 1px solid var(--border);
            border-radius: 18px;
            padding: 28px 24px 24px;
            box-shadow: var(--shadow);
            transition: box-shadow 0.2s, transform 0.2s;
        }
        .card:hover {
            box-shadow: var(--shadow-lg);
            transform: translateY(-2px);
        }

        .card-accent {
            height: 3px;
            border-radius: 99px;
            margin-bottom: 20px;
        }
        .ca-purple { background: linear-gradient(90deg, #4A47A3, #7B78D4); }
        .ca-amber  { background: linear-gradient(90deg, #C07C2A, #E09B50); }
        .ca-green  { background: linear-gradient(90deg, #2D8A5E, #4DB87E); }

        .card-label {
            font-family: 'DM Mono', monospace;
            font-size: 10px;
            letter-spacing: 1.5px;
            text-transform: uppercase;
            color: var(--muted);
            margin-bottom: 16px;
        }

        .pred-number {
            font-family: 'Playfair Display', serif;
            font-size: 80px;
            font-weight: 700;
            line-height: 1;
            margin: 0 0 16px;
        }
        .pn-purple { color: var(--accent); }
        .pn-amber  { color: var(--amber);  }
        .pn-green  { color: var(--green);  }

        .conf-label {
            font-size: 13px;
            color: var(--muted);
            margin-bottom: 8px;
        }
        .conf-label span { font-weight: 600; color: var(--text); }

        .conf-track {
            background: var(--surface2);
            border-radius: 99px;
            height: 6px;
            overflow: hidden;
            margin-bottom: 20px;
        }
        .conf-fill {
            height: 6px;
            border-radius: 99px;
            transition: width 1s cubic-bezier(0.4,0,0.2,1);
            width: 0%;
        }
        .cf-purple { background: linear-gradient(90deg, #4A47A3, #7B78D4); }
        .cf-amber  { background: linear-gradient(90deg, #C07C2A, #E09B50); }
        .cf-green  { background: linear-gradient(90deg, #2D8A5E, #4DB87E); }

        .badges { display: flex; flex-wrap: wrap; gap: 6px; }
        .badge {
            padding: 4px 10px;
            border-radius: 6px;
            font-size: 11px;
            font-family: 'DM Mono', monospace;
            font-weight: 500;
        }
        .bpurple { background: #EEEEFF; color: var(--accent); border: 1px solid #D4D6F5; }
        .bamber  { background: #FFF5E8; color: var(--amber);  border: 1px solid #F5E0C0; }
        .bgreen  { background: #E8F5EF; color: var(--green);  border: 1px solid #C0E0D0; }

        /* ── Summary ── */
        .summary {
            background: var(--surface);
            border: 1px solid var(--border);
            border-radius: 18px;
            padding: 28px;
            margin-top: 20px;
            box-shadow: var(--shadow);
            display: none;
        }
        .summary-title {
            font-family: 'DM Mono', monospace;
            font-size: 10px;
            letter-spacing: 2px;
            text-transform: uppercase;
            color: var(--muted);
            margin-bottom: 20px;
        }
        .summary-grid {
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 16px;
        }
        .sum-item {
            background: var(--bg);
            border: 1px solid var(--border);
            border-radius: 12px;
            padding: 16px;
        }
        .sum-item-label {
            font-size: 11px;
            color: var(--muted);
            font-family: 'DM Mono', monospace;
            margin-bottom: 6px;
        }
        .sum-item-val { font-size: 14px; font-weight: 600; color: var(--text); }
        .tag-agree    { color: var(--green); }
        .tag-disagree { color: var(--red);   }

        /* ── Footer ── */
        .footer {
            border-top: 1px solid var(--border);
            padding: 24px 40px;
            text-align: center;
            font-size: 12px;
            color: var(--muted);
            font-family: 'DM Mono', monospace;
        }
    </style>
</head>
<body>

<!-- Header -->
<header class="header">
    <div class="logo">
        <div class="logo-icon">🧠</div>
        <span class="logo-text">CompressionLab</span>
    </div>
    <div class="header-badge">Pradhi Bobade - Cummins College &middot; DL Project &middot; March 2026</div>
</header>

<!-- Hero -->
<div class="hero">
    <div class="hero-eyebrow">Live Inference Demo</div>
    <h1>Model <span>Compression</span><br>in Action</h1>
    <p>Upload a handwritten digit and watch three compressed models predict simultaneously — see how compression affects accuracy in real time.</p>
    <div class="stats-row">
        <div class="stat">
            <div class="stat-val">99.34%</div>
            <div class="stat-lbl">Teacher acc</div>
        </div>
        <div class="stat">
            <div class="stat-val">3.4&times;</div>
            <div class="stat-lbl">PTQ smaller</div>
        </div>
        <div class="stat">
            <div class="stat-val">63.6&times;</div>
            <div class="stat-lbl">Student smaller</div>
        </div>
        <div class="stat">
            <div class="stat-val">MNIST</div>
            <div class="stat-lbl">Dataset</div>
        </div>
    </div>
</div>

<!-- Main -->
<div class="container">

    <div class="tip">
        <span>💡</span>
        <span>Upload a <strong>single handwritten digit (0–9)</strong>. Works with white paper photos or black background images.</span>
    </div>

    <p class="section-label">Step 1 &mdash; Upload image</p>
    <div class="upload-card">
        <label class="drop-zone" for="fileInput">
            <div class="upload-icon">📂</div>
            <div class="drop-title">Click to upload a digit image</div>
            <div class="drop-sub">PNG &middot; JPG &middot; Single digit only</div>
            <input type="file" id="fileInput" accept="image/*">
        </label>

        <div id="preview-wrap">
            <img id="preview" src="" alt="preview"/>
            <div class="preview-label">Uploaded image</div>
        </div>

        <button class="run-btn" onclick="runModels()">&#9654; Run All 3 Models</button>
    </div>

    <div class="error" id="errorBox"></div>

    <div class="loading" id="loading">
        <div class="spinner"></div>
        <p>Running inference on all 3 models&hellip;</p>
    </div>

    <p class="section-label" id="results-label" style="display:none">Step 2 &mdash; Predictions</p>
    <div class="results" id="results">

        <div class="card">
            <div class="card-accent ca-purple"></div>
            <div class="card-label">Original Teacher &middot; 27.20 MB</div>
            <div class="pred-number pn-purple" id="pred0">&mdash;</div>
            <div class="conf-label">Confidence: <span id="confv0">&mdash;</span></div>
            <div class="conf-track"><div class="conf-fill cf-purple" id="bar0"></div></div>
            <div class="badges">
                <span class="badge bpurple">27.20 MB</span>
                <span class="badge bpurple">99.34% acc</span>
                <span class="badge bpurple">Baseline</span>
            </div>
        </div>

        <div class="card">
            <div class="card-accent ca-amber"></div>
            <div class="card-label">PTQ &middot; INT8 &middot; 7.91 MB</div>
            <div class="pred-number pn-amber" id="pred1">&mdash;</div>
            <div class="conf-label">Confidence: <span id="confv1">&mdash;</span></div>
            <div class="conf-track"><div class="conf-fill cf-amber" id="bar1"></div></div>
            <div class="badges">
                <span class="badge bamber">7.91 MB</span>
                <span class="badge bamber">3.4&times; smaller</span>
                <span class="badge bamber">INT8</span>
            </div>
        </div>

        <div class="card">
            <div class="card-accent ca-green"></div>
            <div class="card-label">Student &middot; KD &middot; 0.43 MB</div>
            <div class="pred-number pn-green" id="pred2">&mdash;</div>
            <div class="conf-label">Confidence: <span id="confv2">&mdash;</span></div>
            <div class="conf-track"><div class="conf-fill cf-green" id="bar2"></div></div>
            <div class="badges">
                <span class="badge bgreen">0.43 MB</span>
                <span class="badge bgreen">63.6&times; smaller</span>
                <span class="badge bgreen">KD</span>
            </div>
        </div>

    </div>

    <div class="summary" id="summary">
        <div class="summary-title">Prediction Summary</div>
        <div class="summary-grid">
            <div class="sum-item">
                <div class="sum-item-label">All models agree?</div>
                <div class="sum-item-val" id="sum-agree">&mdash;</div>
            </div>
            <div class="sum-item">
                <div class="sum-item-label">Highest confidence</div>
                <div class="sum-item-val" id="sum-best">&mdash;</div>
            </div>
            <div class="sum-item">
                <div class="sum-item-label">Student model predicted</div>
                <div class="sum-item-val" id="sum-student">&mdash;</div>
            </div>
        </div>
    </div>

</div>

<footer class="footer">
    Efficient Deep Learning &middot; Model Compression &amp; Quantization &middot;
    Pruning &middot; PTQ &middot; QAT &middot; Knowledge Distillation
</footer>

<script>
document.getElementById('fileInput').onchange = function(e) {
    const file = e.target.files[0];
    if (!file) return;
    const reader = new FileReader();
    reader.onload = function(ev) {
        document.getElementById('preview').src = ev.target.result;
        document.getElementById('preview-wrap').style.display = 'block';
    };
    reader.readAsDataURL(file);
};

async function runModels() {
    const file = document.getElementById('fileInput').files[0];
    if (!file) { showError('Please select an image first!'); return; }

    document.getElementById('errorBox').style.display       = 'none';
    document.getElementById('loading').style.display        = 'block';
    document.getElementById('results').style.display        = 'none';
    document.getElementById('summary').style.display        = 'none';
    document.getElementById('results-label').style.display  = 'none';

    const formData = new FormData();
    formData.append('image', file);

    try {
        const res  = await fetch('/predict', { method: 'POST', body: formData });
        const data = await res.json();
        if (data.error) { showError(data.error); return; }

        const keys  = ['teacher', 'ptq', 'student'];
        const names = ['Original Teacher', 'PTQ (INT8)', 'Student (KD)'];
        const preds = [];
        const confs = [];

        keys.forEach((m, i) => {
            const p = data[m].prediction;
            const c = data[m].confidence;
            preds.push(p);
            confs.push(c);
            document.getElementById('pred'+i).textContent  = p;
            document.getElementById('confv'+i).textContent = c.toFixed(1) + '%';
            setTimeout(() => {
                document.getElementById('bar'+i).style.width = Math.min(c, 100) + '%';
            }, 200 + i * 150);
        });

        const allAgree = preds.every(p => p === preds[0]);
        const bestIdx  = confs.indexOf(Math.max(...confs));

        document.getElementById('sum-agree').innerHTML = allAgree
            ? '<span class="tag-agree">&#10003; Yes &mdash; all predicted ' + preds[0] + '</span>'
            : '<span class="tag-disagree">&#10007; No &mdash; ' + preds.join(' / ') + '</span>';

        document.getElementById('sum-best').textContent =
            names[bestIdx] + ' &middot; ' + confs[bestIdx].toFixed(1) + '%';

        document.getElementById('sum-student').textContent =
            'Digit ' + preds[2] + ' &middot; ' + confs[2].toFixed(1) + '% confidence';

        document.getElementById('loading').style.display       = 'none';
        document.getElementById('results').style.display       = 'grid';
        document.getElementById('summary').style.display       = 'block';
        document.getElementById('results-label').style.display = 'block';

    } catch(e) {
        showError('Server error: ' + e.message);
    }
}

function showError(msg) {
    document.getElementById('loading').style.display = 'none';
    const box = document.getElementById('errorBox');
    box.textContent = '\u26A0 ' + msg;
    box.style.display = 'block';
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
    file = request.files['image']
    try:
        img    = Image.open(io.BytesIO(file.read()))
        tensor = preprocess(img)

        t_pred, t_conf = run_predict(teacher,   tensor)
        p_pred, p_conf = run_predict(ptq_model, tensor)
        s_pred, s_conf = run_predict(student,   tensor)

        return jsonify({
            'teacher': {'prediction': t_pred, 'confidence': t_conf},
            'ptq'    : {'prediction': p_pred, 'confidence': p_conf},
            'student': {'prediction': s_pred, 'confidence': s_conf},
        })
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    print("\n" + "="*50)
    print("  CompressionLab Demo running!")
    print("  Open browser: http://127.0.0.1:5000")
    print("="*50 + "\n")
    app.run(debug=False, port=5000)