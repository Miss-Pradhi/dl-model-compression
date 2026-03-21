import torch
import torch.optim as optim
import copy

from src.model      import TeacherCNN, StudentCNN
from src.train      import get_dataloaders, train_one_epoch, evaluate
from src.pruning    import (apply_unstructured_pruning, run_structured_pruning,
                            count_zero_weights)
from src.quantization import apply_ptq, run_qat, get_model_size_mb
from src.distillation import DistillationLoss, train_with_distillation
from src.visualize  import plot_full_report

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}\n")

train_loader, test_loader = get_dataloaders()
criterion = torch.nn.CrossEntropyLoss()

# ── 1. Train Teacher ───────────────────────────────────────────
print("=" * 50)
print("PHASE 1: Training teacher model...")
print("=" * 50)
teacher = TeacherCNN().to(device)
opt = optim.Adam(teacher.parameters(), lr=1e-3)
for epoch in range(5):
    loss = train_one_epoch(teacher, train_loader, opt, criterion, device)
    acc  = evaluate(teacher, test_loader, device)
    print(f"  Epoch {epoch+1}: loss={loss:.4f}  acc={acc:.4f}")
torch.save(teacher.state_dict(), 'results/teacher.pth')
teacher_acc  = evaluate(teacher, test_loader, device)
teacher_size = get_model_size_mb(teacher)

# ── 2. Unstructured Pruning ────────────────────────────────────
print("\n" + "=" * 50)
print("PHASE 2: Unstructured pruning (40%)...")
print("=" * 50)
unpruned_model = apply_unstructured_pruning(copy.deepcopy(teacher).cpu())
print(f"  Zero weights          : {count_zero_weights(unpruned_model):.1f}%")
unstructured_acc = evaluate(unpruned_model.to(device), test_loader, device)
print(f"  Accuracy after pruning: {unstructured_acc:.4f}")

# ── 3. Structured Pruning ──────────────────────────────────────
print("\n" + "=" * 50)
print("PHASE 3: Structured pruning (30%)...")
print("=" * 50)
structured_model, structured_acc = run_structured_pruning(
    teacher, test_loader, device, amount=0.3)

# ── 4. Post-Training Quantization ─────────────────────────────
print("\n" + "=" * 50)
print("PHASE 4: Post-Training Quantization (PTQ)...")
print("=" * 50)
ptq_model = apply_ptq(copy.deepcopy(teacher).cpu(), test_loader, device)
ptq_size  = get_model_size_mb(ptq_model)
ptq_acc = evaluate(ptq_model, test_loader, device)
print(f"  Original size : {teacher_size:.2f} MB")
print(f"  PTQ size      : {ptq_size:.2f} MB")
print(f"  PTQ accuracy  : {ptq_acc:.4f}")

# ── 5. Quantization-Aware Training ────────────────────────────
print("\n" + "=" * 50)
print("PHASE 5: Quantization-Aware Training (QAT)...")
print("=" * 50)
qat_model, qat_acc = run_qat(
    teacher, train_loader, test_loader, device, epochs=3)
qat_size = get_model_size_mb(qat_model)
print(f"  QAT size      : {qat_size:.2f} MB")
torch.save(qat_model.state_dict(), 'results/qat_model.pth')

# ── 6. Knowledge Distillation ──────────────────────────────────
print("\n" + "=" * 50)
print("PHASE 6: Knowledge Distillation (Teacher → Student)...")
print("=" * 50)
student   = StudentCNN().to(device)
dist_loss = DistillationLoss(temperature=4.0, alpha=0.7)
opt_s     = optim.Adam(student.parameters(), lr=1e-3)
student   = train_with_distillation(
    teacher, student, train_loader, opt_s, dist_loss, device, epochs=5)
student_acc  = evaluate(student, test_loader, device)
student_size = get_model_size_mb(student)
print(f"  Student accuracy: {student_acc:.4f}")
print(f"  Student size    : {student_size:.2f} MB")
torch.save(student.state_dict(), 'results/student_distilled.pth')

# ── 7. Full Comparison Report ──────────────────────────────────
print("\n" + "=" * 50)
print("PHASE 7: Generating full report...")
print("=" * 50)

results = {
    'Original'      : {'accuracy': teacher_acc    * 100, 'size_mb': teacher_size},
    'Unstructured'  : {'accuracy': unstructured_acc * 100, 'size_mb': teacher_size},
    'Structured'    : {'accuracy': structured_acc  * 100, 'size_mb': get_model_size_mb(structured_model)},
    'PTQ'           : {'accuracy': ptq_acc         * 100, 'size_mb': ptq_size},
    'QAT'           : {'accuracy': qat_acc         * 100, 'size_mb': qat_size},
    'Student (KD)'  : {'accuracy': student_acc     * 100, 'size_mb': student_size},
}

plot_full_report(results)

# ── 8. Print summary table ─────────────────────────────────────
print("\n" + "=" * 50)
print("FINAL SUMMARY")
print("=" * 50)
print(f"{'Method':<16} {'Accuracy':>10} {'Size (MB)':>12} {'Compression':>13}")
print("-" * 55)
baseline = results['Original']['size_mb']
for method, vals in results.items():
    ratio = baseline / vals['size_mb']
    print(f"{method:<16} {vals['accuracy']:>9.2f}%"
          f" {vals['size_mb']:>11.2f}  {ratio:>11.1f}x")

print("\nDone! Saved to results/")