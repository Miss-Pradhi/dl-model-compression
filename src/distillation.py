import torch
import torch.nn as nn
import torch.nn.functional as F

class DistillationLoss(nn.Module):
    def __init__(self, temperature=4.0, alpha=0.7):
        """
        temperature: softens teacher probability distributions (higher = softer)
        alpha: weight for distillation loss vs hard-label cross-entropy
        """
        super().__init__()
        self.T = temperature
        self.alpha = alpha
        self.ce = nn.CrossEntropyLoss()

    def forward(self, student_logits, teacher_logits, labels):
        # Soft targets from teacher (the "dark knowledge")
        soft_teacher = F.softmax(teacher_logits / self.T, dim=1)
        soft_student = F.log_softmax(student_logits / self.T, dim=1)
        distill_loss = F.kl_div(soft_student, soft_teacher, reduction='batchmean') * (self.T ** 2)

        # Hard targets from ground truth
        hard_loss = self.ce(student_logits, labels)

        return self.alpha * distill_loss + (1 - self.alpha) * hard_loss

def train_with_distillation(teacher, student, loader, optimizer, criterion, device, epochs=5):
    teacher.eval()   # Teacher is frozen
    student.train()

    for epoch in range(epochs):
        total_loss = 0
        for X, y in loader:
            X, y = X.to(device), y.to(device)
            with torch.no_grad():
                teacher_logits = teacher(X)
            student_logits = student(X)
            loss = criterion(student_logits, teacher_logits, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}: distillation loss = {total_loss/len(loader):.4f}")
    return student