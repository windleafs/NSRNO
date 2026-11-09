"""All-receiver Woodbury inverse for scalar pressure solver coordinates."""
import time
import torch


class PressureReceiverNormal:
    @torch.no_grad()
    def __init__(self, operator, pressure_adjoint, response, penalty=None):
        start = time.time()
        g = operator.g
        self.g = g
        k, n = len(g.rx_cols), g.H*g.W
        device = g.mask.device
        self.real = torch.empty((k, n), device=device, dtype=torch.float32)
        self.imag = torch.empty_like(self.real)
        for j in range(0, k, 8):
            count = min(8, k-j)
            rhs = torch.zeros((count, g.H, g.W), device=device, dtype=torch.complex64)
            rhs[torch.arange(count, device=device), g.face, g.rx_cols[j:j+count]] = complex(response).conjugate()
            value = pressure_adjoint(rhs)*g.mask
            self.real[j:j+count].copy_(value.real.flatten(1))
            self.imag[j:j+count].copy_(value.imag.flatten(1))
        columns_seconds = time.time()-start
        gram = torch.complex(self.real@self.real.T + self.imag@self.imag.T,
                             self.real@self.imag.T - self.imag@self.real.T).to(torch.complex128)
        gram = (gram+gram.mH)*.5
        spectrum = torch.linalg.eigvalsh(gram)
        if not torch.isfinite(spectrum).all() or float(spectrum[-1]) <= 0:
            raise RuntimeError('Nonpositive or nonfinite pressure receiver Gram scale')
        # Chosen once from the necessary receiver Gram; no extra power probe.
        self.penalty = .01*float(spectrum[-1]) if penalty is None else float(penalty)
        if self.penalty <= 0:
            raise ValueError('Scalar-pressure penalty must remain strictly positive')
        shift = max(0., -float(spectrum[0])*1.01) + float(spectrum[-1])*5e-6
        self.cholesky = torch.linalg.cholesky(gram + torch.eye(k, device=device,
            dtype=torch.complex128)*(self.penalty+shift))
        self.metadata = {'receivers': k, 'stored_fields': 'scalar pressure',
            'storage_GiB': 8*k*n/2**30, 'columns_seconds': columns_seconds,
            'seconds': time.time()-start, 'Gram_min_eigenvalue': float(spectrum[0]),
            'Gram_max_eigenvalue': float(spectrum[-1]), 'preconditioner_roundoff_shift': shift,
            'field_penalty': self.penalty, 'new_penalty_rule': '.01*largest receiver Gram eigenvalue, frozen per frequency',
            'all_receivers_included': True, 'extra_power_probe': False}

    def project(self, value):
        qr, qi = value.real.flatten(1).contiguous(), value.imag.flatten(1).contiguous()
        return torch.complex(qr@self.real.T + qi@self.imag.T,
                             qi@self.real.T - qr@self.imag.T)

    @torch.no_grad()
    def __call__(self, value):
        projections = self.project(value).to(torch.complex128)
        coeff = torch.cholesky_solve(projections.T, self.cholesky).T.to(torch.complex64)
        combined = torch.complex(coeff.real@self.real - coeff.imag@self.imag,
                                 coeff.imag@self.real + coeff.real@self.imag).reshape_as(value)
        return (value-combined)/self.penalty
