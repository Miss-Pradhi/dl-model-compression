import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

def plot_full_report(results: dict, save_dir='results'):
    os.makedirs(save_dir, exist_ok=True)

    methods  = list(results.keys())
    accuracy = [results[m]['accuracy'] for m in methods]
    size_mb  = [results[m]['size_mb']  for m in methods]
    colors   = ['#7F77DD','#C04898','#9F59D4','#1D9E75','#EF9F27','#3B8BD4']

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.patch.set_facecolor('#F9F9F9')

    # ── Chart 1: Accuracy ──────────────────────────────────────
    ax1 = axes[0]
    bars1 = ax1.bar(methods, accuracy, color=colors, width=0.55,
                    edgecolor='white', linewidth=1.2)
    ax1.set_ylabel('Accuracy (%)', fontsize=11)
    ax1.set_title('Accuracy Comparison', fontsize=13, fontweight='bold', pad=12)
    ax1.set_ylim(96.0, 100.0)
    ax1.tick_params(axis='x', rotation=20, labelsize=9)
    ax1.spines[['top','right']].set_visible(False)
    ax1.set_facecolor('#FAFAFA')
    ax1.yaxis.set_minor_locator(ticker.MultipleLocator(0.2))
    for bar, val in zip(bars1, accuracy):
        ax1.text(bar.get_x() + bar.get_width()/2,
                 bar.get_height() + 0.05,
                 f'{val:.1f}%', ha='center', va='bottom',
                 fontsize=8, fontweight='bold')

    # ── Chart 2: Model size ─────────────────────────────────────
    ax2 = axes[1]
    bars2 = ax2.bar(methods, size_mb, color=colors, width=0.55,
                    edgecolor='white', linewidth=1.2)
    ax2.set_ylabel('Model Size (MB)', fontsize=11)
    ax2.set_title('Model Size Comparison', fontsize=13, fontweight='bold', pad=12)
    ax2.tick_params(axis='x', rotation=20, labelsize=9)
    ax2.spines[['top','right']].set_visible(False)
    ax2.set_facecolor('#FAFAFA')
    for bar, val in zip(bars2, size_mb):
        ax2.text(bar.get_x() + bar.get_width()/2,
                 bar.get_height() + 0.2,
                 f'{val:.2f}MB', ha='center', va='bottom',
                 fontsize=8, fontweight='bold')

    # ── Chart 3: Compression ratio ──────────────────────────────
    ax3 = axes[2]
    baseline = size_mb[0]
    ratios   = [baseline / s for s in size_mb]
    bars3 = ax3.bar(methods, ratios, color=colors, width=0.55,
                    edgecolor='white', linewidth=1.2)
    ax3.set_ylabel('Compression Ratio (x)', fontsize=11)
    ax3.set_title('Compression Ratio vs Original', fontsize=13,
                  fontweight='bold', pad=12)
    ax3.tick_params(axis='x', rotation=20, labelsize=9)
    ax3.spines[['top','right']].set_visible(False)
    ax3.set_facecolor('#FAFAFA')
    ax3.axhline(y=1.0, color='gray', linestyle='--', linewidth=0.8, alpha=0.6)
    for bar, val in zip(bars3, ratios):
        ax3.text(bar.get_x() + bar.get_width()/2,
                 bar.get_height() + 0.1,
                 f'{val:.1f}x', ha='center', va='bottom',
                 fontsize=8, fontweight='bold')

    plt.suptitle('Model Compression — Full Results Report',
                 fontsize=15, fontweight='bold', y=1.01)
    plt.tight_layout()
    out = os.path.join(save_dir, 'full_report.png')
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n  Chart saved → {out}")