"""Frozen pure neural channel operator and low-memory full-observation VJP.

No acquisition-reference wavefield, wave equation or true material map is read.
"""
from pathlib import Path
import json
import torch
from torch import nn
from nsrno.uno_rf import ChannelUNO
from nsrno.uno_data import rf_objective,rf_statistics
from nsrno.rf_representation import decode

class UNOSurrogate(nn.Module):
    def __init__(self,checkpoint='out/v5/uno_round1_triparam/best.pt',
                 data_index='out/v5/uno_triparam_v1/index.json',device='cuda'):
        super().__init__(); self.device=device
        saved=torch.load(checkpoint,map_location='cpu',weights_only=True)
        self.model=ChannelUNO().to(device)
        config=saved.get('config',saved.get('settings',{}).get('network',{}))
        if config.get('internal_representation')!=self.model.config['internal_representation']:
            raise ValueError('Checkpoint source-clock representation does not match this model')
        self.model.load_state_dict(saved['model'])
        self.model.requires_grad_(False); self.model.eval()
        self.scales=saved['scales'] if 'scales' in saved else json.load(open(data_index))['scales']
        self.checkpoint_step=saved['step']

    def scale(self,source_mhz): return self.scales[str(float(source_mhz))]['iq_rms']

    def forward(self,material,source_mhz=7.5,angles=None,rf=True):
        """[3,961,1121] c/rho/current alpha -> [angles,receivers,time]."""
        if material.shape!=(3,961,1121): raise ValueError('Native3-parameter maps required')
        angles=list(range(11)) if angles is None else list(angles)
        feature=self.model.encode(material[None]); source=torch.tensor([source_mhz],device=material.device)
        predictions=[]
        for a in angles:
            angle=torch.tensor([-8+1.6*a],device=material.device)
            value=self.model.observe(feature,angle,source)*self.scale(source_mhz)
            predictions.append(decode(value)[0] if rf else value[0])
        return torch.stack(predictions)

    def objective_and_gradient(self,material,observations):
        """Full objective and native3-map gradient, one update for every source/angle.

        observations: list of dicts with source_mhz and IQ [11,2,192,640].
        The shared-encoder VJP is checked against ordinary autograd in profiling.
        """
        leaf_material=material.detach().requires_grad_(True)
        feature=self.model.encode(leaf_material[None]); leaf=feature.detach().requires_grad_(True)
        total=0.; count=11*len(observations)
        for row in observations:
            source=torch.tensor([row['source_mhz']],device=material.device)
            scale=self.scale(row['source_mhz']); target=row['iq'].to(material.device)
            for a in range(11):
                angle=torch.tensor([-8+1.6*a],device=material.device)
                prediction=self.model.observe(leaf,angle,source)
                loss,_=rf_objective(prediction,target[a:a+1]/scale)
                (loss/count).backward(); total+=float(loss.detach())/count
        feature.backward(leaf.grad)
        return total,leaf_material.grad.detach()

    @torch.no_grad()
    def objective(self,material,observations):
        import numpy as np
        feature=self.model.encode(material[None]); total=0.; count=11*len(observations); stats=np.zeros((5,2))
        for row in observations:
            source=torch.tensor([row['source_mhz']],device=material.device)
            scale=self.scale(row['source_mhz']); target=row['iq'].to(material.device)
            for a in range(11):
                angle=torch.tensor([-8+1.6*a],device=material.device)
                prediction=self.model.observe(feature,angle,source)
                loss,_=rf_objective(prediction,target[a:a+1]/scale); total+=float(loss)/count
                stats+=rf_statistics(prediction*scale,target[a:a+1])
        return {'data_loss':total,'relative_rms':np.sqrt(stats[:,0]/np.maximum(stats[:,1],1e-30)).tolist()}
