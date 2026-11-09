"""Positive block balancing for truncated CPU PCG; no new forward calls."""
import numpy as np
from .coupled_coordinates import Coordinates as BaseCoordinates, pcg, dot


class Coordinates(BaseCoordinates):
    def set_block_precondition(self, gradient):
        norms=np.sqrt(np.mean(gradient*gradient,axis=(-2,-1)))
        reference=max(float(norms.max()),1e-30)
        safe=np.maximum(norms,reference*1e-4)
        geometric=float(np.exp(np.log(safe).mean()))
        factors=np.clip(geometric/safe,1/30,30)
        self.block_factors=factors[:,None,None]
        return factors.tolist()

    def precondition(self, residual):
        # Block constants commute with the positive spatial operator; therefore
        # the composition stays symmetric positive definite for ordinary PCG.
        return super().precondition(residual)*self.block_factors

    def damping_metric(self, direction):
        """Positive block LM metric reciprocal to the gradient block scaling.

        This is a gradient-scale approximation, not a measured Fisher diagonal.
        Factors remain fixed throughout each PCG solve.
        """
        return direction/self.block_factors
