# Main code for knowledge distillation implementation
# Implementation of distillation loss function, which combines cross-entropy loss and KL divergence
import torch
import torch.nn.functional as F # Contains many neural network functions, including activation & loss functions.

def distillation_loss(student_logits, teacher_logits, labels, T=5, alpha=0.7):
    """
    student_logits: output from student
    teacher_logits: output from teacher
    labels: true one-hot labels
    T: temperature (controls softmax sharpness)
    alpha: weight between soft and hard targets
    """
    # Soft loss (KL divergence between teacher and student outputs)
    soft_loss = F.kl_div(F.log_softmax(student_logits/T, dim=1), # log is required for student for the KL divergence formula
                         F.softmax(teacher_logits/T, dim=1),
                         reduction='batchmean') * (T*T) # 'batchmean' is to average the loss over the batch, so that the loss-function is batch independent on not dependent on some random sample
    
    # Hard loss (cross-entropy loss between student outputs and true labels)
    hard_loss = F.cross_entropy(student_logits, labels)
    
    return alpha * soft_loss + (1 - alpha) * hard_loss
