import torch


def percentage(batch_size: int, max_index: int, current_index: int) -> float:
    """Calculate epoch progress percentage."""
    if batch_size == 0:
        return 0.0
    batched_max = max_index // batch_size
    if batched_max == 0:
        return 0.0
    return round(current_index / batched_max * 100, 2)


def nsp_accuracy(result: torch.Tensor, target: torch.Tensor) -> float:
    """Calculate Next Sentence Prediction accuracy."""
    if result.size(0) == 0:
        return 0.0
    s = (result.argmax(1) == target.argmax(1)).sum()
    return round(float(s / result.size(0)), 2)


def token_accuracy(
    result: torch.Tensor, target: torch.Tensor, inverse_token_mask: torch.Tensor
) -> float:
    """Calculate Masked Language Model accuracy over masked words only."""
    masked_result = result.argmax(-1).masked_select(~inverse_token_mask)
    masked_target = target.masked_select(~inverse_token_mask)
    if masked_target.numel() == 0:
        return 0.0
    matches = (masked_result == masked_target).sum()
    return round(float(matches / masked_target.numel()), 2)
