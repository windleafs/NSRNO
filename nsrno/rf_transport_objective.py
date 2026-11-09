"""Fixed-observation RF transport objective and its CPU data-space adjoint."""
import numpy as np
if __package__:
    from .rf_vector_transport import lift,hilbert
    from .rf_weighted_transport_scaled import transport
else:
    from rf_vector_transport import lift,hilbert
    from rf_weighted_transport_scaled import transport


class RFTransportObjective:
    def __init__(self,observed,bins,window_weights,dt_us,lam,
                 tolerance=1e-7,maximum=12000,smoothing_fraction=.02):
        self.observed=np.array(observed,dtype=np.float64,copy=True)
        self.bins=np.array(bins,dtype=np.uint8,copy=True)
        if self.observed.ndim!=3 or self.bins.shape!=self.observed.shape or self.bins.max()>=5:
            raise ValueError('Require all shot-receiver-time samples and five fixed arrival bins')
        if not np.isfinite(self.observed).all():raise ValueError('Nonfinite observed RF')
        if min(dt_us,lam,smoothing_fraction)<=0:raise ValueError('Positive physical scales required')
        self.den=np.array([max(float(np.sum(self.observed[self.bins==k]**2)),1e-20) for k in range(5)])
        self.counts=np.bincount(self.bins.reshape(-1),minlength=5)
        if np.any(self.counts==0):raise ValueError('Each observed arrival window must exist')
        self.coefficients=.5*np.asarray(window_weights,np.float64)/self.den
        if self.coefficients.shape!=(5,) or min(self.coefficients)<=0:raise ValueError('Positive fixed weights required')
        self.amplitude=float(np.sqrt(self.den[0]/self.counts[0]))
        self.lam=float(lam);self.h=float(dt_us);self.tolerance=float(tolerance);self.maximum=int(maximum)
        # A common, observation-fixed smoothing level remains below the weak
        # window RMS. It is never fitted from the current predicted waveform.
        weakest_rms=float(np.sqrt(np.min(self.den/self.counts)))
        self.smoothing=float(smoothing_fraction*weakest_rms/self.amplitude)
        self.factor=self.amplitude**2*self.coefficients[0]/(self.lam*self.h)
        self.window_node_weights=self.coefficients/self.coefficients[0]
        self.warm={}

    def metadata(self):
        return {'fixed_amplitude_unit':self.amplitude,'fixed_window_denominators':self.den.tolist(),
            'fixed_window_counts':self.counts.tolist(),'fixed_original_L2_coefficients':self.coefficients.tolist(),
            'node_weights':self.window_node_weights.tolist(),'lambda_us_per_fixed_amplitude_unit':self.lam,
            'dt_us':self.h,'fixed_smoothing_amplitude_unit':self.smoothing,'fixed_external_factor':self.factor,
            'inner_relative_gap_target':self.tolerance,'inner_maximum':self.maximum,
            'waveform_normalized_by_predicted_mass':False,'all_original_RF_time_samples_used':True,
            'edge_cost':'sqrt(mean of adjacent node weights)',
            'objective':'weighted vector L1 transport with quadratic imbalance and analytic pressure lift'}

    def evaluate(self,prediction,shot,gradient=True,tolerance=None,maximum=None):
        p=np.asarray(prediction,np.float64)
        if p.shape!=self.observed[shot].shape or not np.isfinite(p).all():raise ValueError('Complete finite shot RF required')
        lower=upper=0.;derivative=np.empty_like(p) if gradient else None
        reports=[];all_converged=True;maximum_gap=0.;inner_seconds=0.;maximum_iterations=0
        scaled_adjoint_error_squared=0.
        for receiver in range(len(p)):
            normalized=p[receiver]/self.amplitude
            observed=self.observed[shot,receiver]/self.amplitude
            b=self.window_node_weights[self.bins[shot,receiver]]
            g=lift(normalized,True,self.smoothing)-lift(observed,True,self.smoothing)
            _,v,state,report=transport(g,self.h,self.lam,node_weight=b,warm=self.warm.get((shot,receiver)),
                tolerance=self.tolerance if tolerance is None else tolerance,
                maximum=self.maximum if maximum is None else maximum)
            if state is None:self.warm.pop((shot,receiver),None)
            else:self.warm[shot,receiver]=state
            lower+=self.factor*report['dual'];upper+=self.factor*report['primal']
            if gradient:
                hp=hilbert(normalized);envelope=np.sqrt(normalized*normalized+hp*hp+self.smoothing**2)
                adj=v[:,0]+v[:,2]*normalized/envelope-hilbert(v[:,1]+v[:,2]*hp/envelope)
                derivative[receiver]=self.factor/self.amplitude*adj
            gap=max(report['primal']-report['dual'],0.)
            # ||J_lift||<=2 for the real periodic Hilbert transform. This is
            # conservative and describes RF-adjoint error, not material error.
            bound=2*self.factor/self.amplitude*np.sqrt(2*self.lam*float(b.max())*self.h*gap)
            scaled_adjoint_error_squared+=bound*bound
            all_converged &= report['converged'];maximum_gap=max(maximum_gap,report['relative_gap'])
            inner_seconds+=report['seconds'];maximum_iterations=max(maximum_iterations,report['iterations'])
            reports.append({'receiver':receiver,**report})
        raw_difference=p-self.observed[shot]
        raw_loss=.5*float(np.sum(self.coefficients[self.bins[shot]]*raw_difference*raw_difference))
        relative=(upper-lower)/max(upper,1e-300)
        result={'loss':.5*(upper+lower),'lower':lower,'upper':upper,'relative_gap':relative,
            'raw_RF_loss':raw_loss,'all_inner_converged':bool(all_converged),
            'maximum_channel_relative_gap':maximum_gap,'maximum_inner_iterations':maximum_iterations,
            'CPU_inner_seconds':inner_seconds,'RF_adjoint_L2_error_upper_bound':float(np.sqrt(scaled_adjoint_error_squared)),
            'all_original_RF_samples_used':True,'gradient_is_exact_only_at_zero_dual_gap':True,
            'receivers':reports}
        return result,derivative
