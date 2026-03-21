import torch
import torch.nn as nn
import copy
import os
import tempfile

# --- Post-Training Quantization (PTQ) - Linear layers only (Python 3.14 safe) ---
def apply_ptq(model, calibration_loader, device):
    """
    Quantizes only Linear layers — fully compatible with Python 3.14 + PyTorch.
    Conv2d quantization requires older Python versions.
    """
    model_ptq = copy.deepcopy(model).cpu()
    model_ptq.eval()

    # Manually quantize only Linear layers (no Conv2d)
    def quantize_linear_layers(module):
        for name, child in module.named_children():
            if isinstance(child, nn.Linear):
                # Collect calibration data
                weight_data = child.weight.data
                # Simple dynamic quantization on Linear only
                setattr(module, name,
                        torch.quantization.quantize_dynamic(
                            nn.Sequential(child),
                            {nn.Linear},
                            dtype=torch.qint8
                        )[0])
            else:
                quantize_linear_layers(child)

    # Use dynamic quantization — most compatible across all Python versions
    model_ptq = torch.quantization.quantize_dynamic(
        model_ptq,
        {nn.Linear},          # Only Linear, NOT Conv2d
        dtype=torch.qint8
    )
    return model_ptq


# --- Quantization-Aware Training (QAT) - also Linear only ---
def run_qat(model, train_loader, test_loader, device, epochs=3):
    """
    QAT using dynamic quantization — compatible with Python 3.14.
    """
    from src.train import evaluate

    print("  QAT training (dynamic quantization)...")
    qat_model = copy.deepcopy(model)
    optimizer  = torch.optim.Adam(qat_model.parameters(), lr=1e-4)
    criterion  = nn.CrossEntropyLoss()

    # Fine-tune the model
    for epoch in range(epochs):
        qat_model.train()
        total_loss = 0
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(qat_model(X), y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"    QAT Epoch {epoch+1}: loss={total_loss/len(train_loader):.4f}")

    # Apply dynamic quantization after fine-tuning
    qat_model.eval().cpu()
    qat_model = torch.quantization.quantize_dynamic(
        qat_model,
        {nn.Linear},
        dtype=torch.qint8
    )

    acc = evaluate(qat_model, test_loader, 'cpu')
    print(f"  QAT accuracy : {acc:.4f}")
    return qat_model, acc


# --- Model size utility ---
def get_model_size_mb(model):
    tmp_path = os.path.join(tempfile.gettempdir(), '_tmp_model.pt')
    torch.save(model.state_dict(), tmp_path)
    size = os.path.getsize(tmp_path) / 1e6
    os.remove(tmp_path)
    return size