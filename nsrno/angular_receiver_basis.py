"""Geometry-only angular QR; no RF values or material derivatives are used."""
import torch


def angular_receiver_qr(main, dipole, side, observable):
    """Return Q[A,Q,R,L,3] and relative column norms[A,Q,L,2].

    QR is formed on the actually observed receivers for each lag. Rank-deficient
    columns are zero. Geometry is orthogonalized in float64, then returned in
    the input dtype. The nonzero columns have unit l2 norm, so the projection
    itself does not amplify an arbitrary receiver-noise vector.
    """
    if observable.shape[0] != 1:
        raise ValueError('One current acquisition per update is required')
    w = main.double()[..., None] * observable[0].double()
    columns = [w, w * dipole.double()[..., None],
               w * side.double()[None, :, :, None]]
    reference = w.square().sum(2, keepdim=True).sqrt()
    threshold = 64 * torch.finfo(torch.float64).eps * reference
    q = []; relative_norms = []
    for index, raw in enumerate(columns):
        vector = raw
        for _ in range(2):
            for previous in q:
                vector = vector - (vector * previous).sum(2, keepdim=True) * previous
        norm = vector.square().sum(2, keepdim=True).sqrt()
        active = norm > threshold
        unit = vector / torch.where(active, norm, torch.ones_like(norm))
        unit = torch.where(active, unit, torch.zeros_like(unit))
        q.append(unit)
        if index:
            relative = norm / torch.where(reference > 0, reference, torch.ones_like(reference))
            relative_norms.append(torch.where(active, relative, torch.zeros_like(relative)).squeeze(2))
    return torch.stack(q, -1).to(main.dtype), torch.stack(relative_norms, -1).to(main.dtype)


def raw_receiver_feature_indices(sources, lags):
    return torch.arange(sources * lags * 6 * 2).reshape(sources, lags, 6, 2)[:, :, :3].reshape(-1)
