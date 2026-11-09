"""Independent TRAIN-only material marginals; no shared tissue memberships."""
from pathlib import Path
import argparse
import hashlib
import json
import shutil
import numpy as np
from scipy.special import logsumexp


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class FactorizedMaterialPrior:
    def __init__(self, path, tail=.1, temperature=0.):
        state = np.load(path, allow_pickle=False)
        self.weights = np.asarray(state['marginal_weights'], np.float64)
        self.means = np.asarray(state['marginal_means'], np.float64)
        self.variances = np.asarray(state['marginal_variances'], np.float64)
        if (self.weights.shape != self.means.shape or self.weights.shape != self.variances.shape
                or self.weights.ndim != 2 or self.weights.shape[0] != 3
                or not np.isfinite(self.weights).all() or not np.isfinite(self.means).all()
                or not np.isfinite(self.variances).all() or np.any(self.weights <= 0)
                or np.any(self.variances <= 0) or
                not np.allclose(self.weights.sum(1), 1., rtol=0, atol=1e-12)
                or not 0 < tail < 1 or not np.isfinite(temperature) or temperature < 0):
            raise ValueError('Positive independent marginal mixture required')
        self.tail, self.temperature = float(tail), float(temperature)
        self.lower = np.array([(1400-1540)/80,
            (1540e3*np.exp(-.25)/1680-1000)/100, -5.6/.7])
        self.upper = np.array([(1680-1540)/80,
            (1540e3*np.exp(.25)/1400-1000)/100, 1.4/.7])

    def marginal_value_gradient(self, values, field):
        """NLL and its derivative in one standardized physical feature."""
        values = np.asarray(values, np.float64)
        if (not np.isfinite(values).all() or np.any(values < self.lower[field]-1e-5)
                or np.any(values > self.upper[field]+1e-5)):
            raise ValueError('Input must be in the original feasible physical domain')
        variance = self.variances[field, :, None]+self.temperature**2
        difference = values[None]-self.means[field, :, None]
        log_core = (np.log((1-self.tail)*self.weights[field, :, None])
                    -.5*np.log(2*np.pi*variance)-.5*difference**2/variance)
        log_density = np.logaddexp(logsumexp(log_core, axis=0),
            np.log(self.tail/(self.upper[field]-self.lower[field])))
        # Each field has its own component probabilities. No three-field label.
        derivative = np.sum(np.exp(log_core-log_density[None])*difference/variance, axis=0)
        return -log_density, derivative

    def energy_and_scaled_log_gradient(self, material, known_edge, roi, weight=1., chunk=65536):
        """Physical energy and dE/d(log c/.05,log rho/.1,log tissue alpha/.7).

        Projection and material bounds remain the caller's separate physical map.
        This is an explicitly defined regularization energy, not a posterior
        density transformed between coordinates with an implicit Jacobian term.
        """
        material = np.asarray(material, np.float64)
        roi = np.asarray(roi, bool)
        if material.shape != (3, *roi.shape) or not roi.any() or weight < 0 or chunk < 1:
            raise ValueError('Positive physical fields and a nonempty independent ROI required')
        selected = material[:, roi].copy()
        selected[2] -= np.broadcast_to(known_edge, roi.shape)[roi]
        if not np.isfinite(selected).all() or np.any(selected <= 0):
            raise ValueError('Strictly positive c, rho, and tissue attenuation required')
        features = np.stack([(selected[0]-1540)/80, (selected[1]-1000)/100,
                             np.log(selected[2]/.5)/.7])
        pullback = np.stack([.05*selected[0]/80, .10*selected[1]/100,
                             np.ones(selected.shape[1])])
        count = selected.shape[1]
        loss, selected_gradient = 0., np.zeros_like(selected)
        for field in range(3):
            for begin in range(0, count, chunk):
                block = slice(begin, min(begin+chunk, count))
                values, derivative = self.marginal_value_gradient(features[field, block], field)
                loss += float(weight/count*values.sum())
                selected_gradient[field, block] = weight/count*derivative*pullback[field, block]
        gradient = np.zeros_like(material)
        gradient[:, roi] = selected_gradient
        return loss, gradient


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--training-prior', type=Path,
                        default=Path('out/v5/material_prior_v1/prior.npz'))
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    project = Path.cwd().resolve()
    source, out = args.training_prior.resolve(), args.out.resolve()
    if source != project/'out/v5/material_prior_v1/prior.npz' or not out.is_relative_to(project/'out/v5'):
        raise ValueError('Existing same-project TRAIN prior only')
    if out.exists():
        raise FileExistsError('Preserve existing preparation; do not reset')
    if shutil.disk_usage(project).free < 2*2**30+401*2**20:
        raise RuntimeError('Retain original data reserve')
    metadata = json.loads(source.with_name('metadata.json').read_text())
    if (metadata['status'] != 'fitted_not_reconstruction' or
        metadata['source_case'] != 'Neg_07_Left' or
        metadata['development_case_excluded'] != 'Neg_35_Left' or
        metadata['uses_development_materials'] or metadata['uses_rf']):
        raise ValueError('TRAIN-only provenance is not established')
    state = np.load(source, allow_pickle=False)
    w, mean, covariance = state['weight'], state['mean'], state['covariance']
    if mean.shape != (len(w), 3) or covariance.shape != (len(w), 3, 3):
        raise ValueError('Unexpected existing TRAIN mixture')
    out.mkdir()
    np.savez(out/'prior.npz', marginal_weights=np.tile(w, (3, 1)),
             marginal_means=mean.T, marginal_variances=np.diagonal(covariance, axis1=1, axis2=2).T)
    result = {'status': 'independent_TRAIN_marginals_constructed_not_reconstruction',
        'source_case': 'Neg_07_Left', 'development_material_arrays_read': False,
        'reserved_case_unread': 'Neg_47_Left', 'new_training': False,
        'new_acoustic_solves': 0, 'new_material_derivative_probes': 0,
        'source_prior_SHA256': sha(source), 'source_metadata_SHA256': sha(source.with_name('metadata.json')),
        'source_prior_feature_SHA256_as_recorded': metadata['feature_sha256'],
        'marginal_prior_SHA256': sha(out/'prior.npz'), 'code_SHA256': sha(__file__),
        'components_per_independent_field': len(w), 'shared_component_membership': False,
        'cross_field_log_prior_derivatives': 'identically zero before original c/Z/alpha feasibility projection',
        'patient_gradient_qualification_performed': False,
        'CPU_regularizer_gradient_qualification_still_required_before_use': True,
        'not_automatically_used_by_running_inverse': True}
    (out/'metadata.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
