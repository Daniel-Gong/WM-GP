import torch

class SpatialProximityAttention(torch.nn.Module):
    """
    Computes spatial attention weights based on 1D circular distance
    between queried locations and a cued target location.
    Both inputs are in degrees [-180, 180).

    Enhancement model: weights are centered around 1.0 (neutral baseline).
    The cued location receives a gain boost up to `attended_gain`, while
    distant locations remain at the 1.0 baseline.

        w(x) = 1.0 + (attended_gain - 1.0) * exp(-dist(x, x_cued)^2 / (2*sigma^2))

    When no cue is present, all weights are 1.0 (neutral, no scaling).
    """
    def __init__(self, spatial_std: float = 30.0, attended_gain: float = 2.0):
        super().__init__()
        self.spatial_std = spatial_std
        self.attended_gain = attended_gain

    def forward(self, locations: torch.Tensor, cued_location: float) -> torch.Tensor:
        """
        locations: Tensor of shape (N,) containing queried locations in [-180, 180)
        cued_location: Float representing the attended spatial location

        Returns weights in [1.0, attended_gain]:
          - cued location  -> attended_gain  (active enhancement)
          - distant locations -> 1.0          (neutral baseline, same as no-cue)
        """
        # Compute shortest circular distance
        dist = torch.abs(locations - cued_location)
        dist = torch.minimum(dist, 360.0 - dist)

        # Gaussian proximity: 1.0 at cued location, 0.0 far away
        gaussian = torch.exp(-0.5 * (dist / self.spatial_std) ** 2)

        # Enhancement: linearly scale from 1.0 (baseline) up to attended_gain
        weights = 1.0 + (self.attended_gain - 1.0) * gaussian
        return weights
