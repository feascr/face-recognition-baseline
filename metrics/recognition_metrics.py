import torch


def compute_topk(logits: torch.Tensor, targets: torch.Tensor, topk=(1, 5)):
    # WARNING: works on torch tensors
    maxk = max(topk)
    batch_size = logits.shape[0]
    _, y_preds = logits.topk(k=maxk, dim=1)
    y_preds = y_preds.t()
    targets_reshaped = targets.view(1, -1).expand_as(y_preds)
    correct = (y_preds == targets_reshaped)
    list_topk_accs = []
    for k in topk:
        topk_ids = correct[:k]
        flattened_topk_ids = topk_ids.reshape(-1).float()
        topk_tp = flattened_topk_ids.float().sum(dim=0, keepdim=True)
        topk_acc = topk_tp / batch_size
        list_topk_accs.append(topk_acc.item())
    return list_topk_accs