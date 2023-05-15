# "Rather dirty" implementation of 0-1 scaling and z-scoring for 1D tensors:


class ScaleValues(object):
    """
    Scales tensor values according to a given max (and optionally min) range.
    """

    def __init__(
        self, maxs: torch.Tensor, mins: torch.Tensor = None, clamp: bool = False
    ) -> None:
        self.max = torch.tensor(maxs).reshape((-1, 1, 1)).requires_grad_(False)
        if mins is None:
            self.min = torch.zeros_like(self.max)
            self.denom = self.max
        else:
            self.min = torch.tensor(mins).reshape((-1, 1, 1)).requires_grad_(False)
            self.denom = self.max - self.min
        self.clamp = clamp

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        if self.clamp:
            tensor = tensor.clamp(min=self.min, max=self.max)
        return (tensor - self.min) / self.denom


class ZScore(object):
    """
    Applies z-scoring ((x - mean) / std) with prescribed mean and std values.
    """

    def __init__(
        self, means: torch.Tensor, stds: torch.Tensor, eps: float = 1e-9
    ) -> None:
        self.means = torch.tensor(means).reshape((-1, 1, 1)).requires_grad_(False)
        self.stds = torch.tensor(stds + eps).reshape((-1, 1, 1)).requires_grad_(False)

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        return (tensor - self.means) / self.stds
