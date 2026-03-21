import torch
import torch.nn.utils.prune as prune

# --- Unstructured pruning (individual weights, any pattern) ---
def apply_unstructured_pruning(model, amount=0.4):
    """Zeros out 40% of weights globally — fast but sparse, needs sparse hardware."""
    params = [(m, 'weight') for m in model.modules()
              if isinstance(m, (torch.nn.Linear, torch.nn.Conv2d))]
    prune.global_unstructured(params,
                              pruning_method=prune.L1Unstructured,
                              amount=amount)
    return model

# --- Structured pruning (removes entire filters — hardware-friendly) ---
def apply_structured_pruning(model, amount=0.3):
    """Removes 30% of Conv2d output channels entirely."""
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d):
            prune.ln_structured(module, name='weight', amount=amount, n=2, dim=0)
    return model

# --- Make pruning permanent (remove masks) ---
def remove_pruning_masks(model):
    for name, module in model.named_modules():
        if isinstance(module, (torch.nn.Linear, torch.nn.Conv2d)):
            try:
                prune.remove(module, 'weight')
            except ValueError:
                pass
    return model

# --- FIXED: count zeros from weight_orig * weight_mask ---
def count_zero_weights(model):
    total = 0
    zeros = 0
    for name, module in model.named_modules():
        if isinstance(module, (torch.nn.Linear, torch.nn.Conv2d)):
            if hasattr(module, 'weight_mask'):
                zeros += (module.weight_mask == 0).sum().item()
                total += module.weight_mask.numel()
            elif hasattr(module, 'weight') and module.weight is not None:
                zeros += (module.weight == 0).sum().item()
                total += module.weight.numel()
    if total == 0:
        return 0.0
    return zeros / total * 100

# --- NEW: run structured pruning and evaluate ---
def run_structured_pruning(model, test_loader, device, amount=0.3):
    """
    Removes entire Conv2d filters — gives real speedup on standard hardware.
    """
    import copy
    from src.train import evaluate

    structured_model = copy.deepcopy(model).cpu()
    structured_model = apply_structured_pruning(structured_model, amount=amount)
    structured_model = remove_pruning_masks(structured_model)

    acc = evaluate(structured_model.to(device), test_loader, device)
    print(f"  Structured pruning ({int(amount*100)}%) accuracy: {acc:.4f}")
    return structured_model, acc