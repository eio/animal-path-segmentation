import torch


class ScaleValues:
    """
    Scales tensor values according to a given max (and optionally min) range.
    """

    def __init__(
        self,
        max_range: torch.Tensor,
        min_range: torch.Tensor = None,
        clamp: bool = False,
    ):
        """
        Constructor for ScaleValues class.

        Parameters:
        max_range (torch.Tensor): the maximum value of the range.
        min_range (torch.Tensor): the minimum value of the range, defaults to None.
        clamp (bool): whether to clamp the tensor values within the range, defaults to False.
        """
        # Convert max_range to a tensor and reshape it to have dimensions (-1, 1, 1).
        self.max_range = (
            torch.tensor(max_range).reshape((-1, 1, 1)).requires_grad_(False)
        )

        # If min_range is None, set it to a tensor of zeros with the same shape as max_range and set the denominator to max_range.
        if min_range is None:
            self.min_range = torch.zeros_like(self.max_range)
            self.denominator = self.max_range
        # Otherwise, convert min_range to a tensor and set the denominator to max_range - min_range.
        else:
            self.min_range = (
                torch.tensor(min_range).reshape((-1, 1, 1)).requires_grad_(False)
            )
            self.denominator = self.max_range - self.min_range

        # Set the clamp flag.
        self.clamp = clamp

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Scales the given tensor values to the specified range.

        Parameters:
        tensor (torch.Tensor): the tensor to be scaled.

        Returns:
        torch.Tensor: the scaled tensor values.
        """
        # If the clamp flag is set, clamp the tensor values within the range.
        if self.clamp:
            tensor = tensor.clamp(min=self.min_range, max=self.max_range)

        # Scale the tensor values to the range by subtracting the minimum value and dividing by the denominator.
        scaled_tensor = (tensor - self.min_range) / self.denominator

        # Return the scaled tensor values.
        return scaled_tensor

    def inverse_normalize(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Applies inverse normalization to the given tensor values.

        Parameters:
        tensor (torch.Tensor): the tensor with scaled values to be inverse normalized.

        Returns:
        torch.Tensor: the inverse normalized tensor values.
        """
        # Multiply the tensor values by the denominator and add the minimum value.
        inverse_normalized_tensor = (tensor * self.denominator) + self.min_range

        # Return the inverse normalized tensor values.
        return inverse_normalized_tensor
