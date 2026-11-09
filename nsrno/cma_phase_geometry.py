"""CPU common-mid-angle phase imaging and sparse geometric slowness rays."""
import time
import numpy as np
from scipy import fft,ndimage,sparse


def analytic_channels(coefficients,mask):
    spectrum=np.zeros((*coefficients.shape[:-1],3601),np.complex128)
    spectrum[...,np.flatnonzero(mask)]=2*coefficients
    return fft.ifft(spectrum,axis=-1,norm='ortho',workers=1)


def sample_channels(rf,position):
    i=np.floor(position).astype(np.int64)
    good=(i>=1)&(i<rf.shape[-1]-2)
    i=np.clip(i,1,rf.shape[-1]-3)
    t=position-np.floor(position)
    t2,t3=t*t,t*t*t
    weights=[-.5*t+t2-.5*t3,1-2.5*t2+1.5*t3,
             .5*t+2*t2-1.5*t3,-.5*t2+.5*t3]
    row=np.arange(rf.shape[0])[None,:]
    value=sum(w*rf[row,i+offset] for w,offset in zip(weights,[-1,0,1,2]))
    return value*good


def beamform(rf,angles,x,z,source_x,source_depth,receivers,c0,f0,deadline,progress):
    X,Z=np.meshgrid(x,z)
    xq,zq=X.ravel(),Z.ravel()
    out=np.empty((len(angles),len(z),len(x)),np.complex64)
    for j,theta in enumerate(np.deg2rad(angles)):
        target=np.zeros(len(xq),np.complex64)
        tref=4/f0-float(np.min(source_x*np.sin(theta)/c0))
        for start in range(0,len(xq),256):
            if time.time()>deadline:
                raise TimeoutError('CPU phase initialization budget exhausted')
            xx,zz=xq[start:start+256],zq[start:start+256]
            lateral=xx[:,None]-receivers[None,:]
            theta_rx=np.arctan2(lateral,zz[:,None])
            ratio=theta_rx/np.deg2rad(30.)
            weight=(.5+.5*np.cos(np.pi*np.clip(ratio,-1,1)))*(abs(ratio)<1)
            incident=tref+(xx*np.sin(theta)+(zz-source_depth)*np.cos(theta))/c0
            clock=(incident[:,None]+np.sqrt(lateral*lateral+zz[:,None]**2)/c0)/2.5e-8
            values=sample_channels(rf[j],clock)
            footprint=xx-(zz-source_depth)*np.tan(theta)
            valid=(footprint>=source_x.min())&(footprint<=source_x.max())
            target[start:start+256]=(values*weight).sum(-1)/np.maximum(weight.sum(-1),1e-30)*valid
        out[j]=target.reshape(len(z),len(x))
        progress(j+1)
    return out


def common_midangle_images(compound,gamma,dx,dz,f0,c0,halfwidth_deg=1.6):
    nz,nx=compound.shape[-2:]
    pz,px=128,64
    H,W=fft.next_fast_len(nz+2*pz),fft.next_fast_len(nx+2*px)
    padded=np.zeros((len(compound),H,W),np.complex64)
    padded[:,pz:pz+nz,px:px+nx]=compound
    spectrum=fft.fft2(padded,axes=(-2,-1),workers=1)
    kz=fft.fftfreq(H,dz)[:,None]
    kx=fft.fftfreq(W,dx)[None,:]
    normal=kz*np.cos(gamma)+kx*np.sin(gamma)
    tangent=kx*np.cos(gamma)-kz*np.sin(gamma)
    angle=np.arctan2(tangent,normal)/np.deg2rad(halfwidth_deg)
    radial=(normal/(2*f0/c0)-1)/.25
    mask=(np.cos(np.pi*np.clip(angle,-1,1)/2)**2*
          np.cos(np.pi*np.clip(radial,-1,1)/2)**2*
          (abs(angle)<1)*(abs(radial)<1)*(normal>0)).astype(np.float32)
    filtered=fft.ifft2(spectrum*mask,axes=(-2,-1),workers=1)
    return filtered[:,pz:pz+nz,px:px+nx].copy()


def tracked_phase(a,b,x,z,xc,zc):
    sigma=(.0007/(z[1]-z[0]),.0007/(x[1]-x[0]))
    cov=a.astype(np.complex128)*b.conj()
    smooth=lambda v:ndimage.gaussian_filter(v,sigma,mode='nearest')
    cr,ci=smooth(cov.real),smooth(cov.imag)
    ea,eb=smooth(abs(a.astype(np.complex128))**2),smooth(abs(b.astype(np.complex128))**2)
    X,Z=np.meshgrid(xc,zc)
    coordinates=np.stack([(Z-z[0])/(z[1]-z[0]),(X-x[0])/(x[1]-x[0])])
    at=lambda v:ndimage.map_coordinates(v,coordinates,order=1,mode='nearest',prefilter=False)
    covariance=at(cr)+1j*at(ci)
    aa,bb=at(ea),at(eb)
    epsilon=max(float(np.percentile(ea+eb,99))*1e-10,1e-30)
    product=aa*bb
    coherence=np.clip(abs(covariance)**2/(product+epsilon**2),0,1)
    weight=coherence**2*product/(product+epsilon**2)
    return np.angle(covariance),weight,coherence


def geometric_coverage(x,z,raw_angles,gamma,tx_range,rx_range,source_depth):
    X,Z=np.meshgrid(x,z)
    valid=np.ones(X.shape,bool)
    for theta in np.deg2rad(raw_angles):
        foot=X-(Z-source_depth)*np.tan(theta)
        valid&=(foot>=tx_range[0])&(foot<=tx_range[1])
    central=np.deg2rad(raw_angles[len(raw_angles)//2])
    psi=2*gamma-central
    for rx_angle in [psi-np.deg2rad(3.2),psi+np.deg2rad(3.2)]:
        foot=X-Z*np.tan(rx_angle)
        valid&=(foot>=rx_range[0])&(foot<=rx_range[1])
    return valid


def ray_matrix(theta,x,zedges):
    z=(zedges[:-1]+zedges[1:])/2
    X,Z=np.meshgrid(x,z)
    xx,zz=X.ravel(),Z.ravel()
    nq=len(xx);nx=len(x)
    rows,cols,values=[],[],[]
    for j,(lo,hi) in enumerate(zip(zedges[:-1],zedges[1:])):
        length=np.clip(zz-lo,0,hi-lo)
        use=length>0
        mid=lo+length[use]/2
        lateral=xx[use]-(zz[use]-mid)*np.tan(theta)
        coordinate=(lateral-x[0])/(x[1]-x[0])
        left=np.floor(coordinate).astype(np.int64)
        fraction=coordinate-left
        query=np.flatnonzero(use)
        for index,weight in [(left,1-fraction),(left+1,fraction)]:
            rows.append(query);cols.append(j*nx+np.clip(index,0,nx-1))
            values.append(length[use]/np.cos(theta)*weight)
    return sparse.coo_matrix((np.concatenate(values),
        (np.concatenate(rows),np.concatenate(cols))),shape=(nq,nq)).tocsr()


def inverse_modulus(frequency,fit,t):
    D=np.exp(2j*np.pi*frequency*2.5e-9)
    e=sum(fit['eta_per_unit_alpha']);A=(1+.5*e*t)**2
    k=sum(v/(1+tau*(D-1)/2.5e-9)
          for v,tau in zip(fit['eta_per_unit_alpha'],fit['taus_s']))
    h=k*t/A
    return (1-(1-1/D)*h)/(A*(1-h))


def effective_speed(c,fit,f0=3e6,alpha=.5):
    return c/np.real(np.sqrt(inverse_modulus(f0,fit,alpha*c/1540)))


def nominal_physical_speed(effective,fit,f0=3e6,alpha=.5):
    value=np.array(effective,np.float64)
    for _ in range(5):
        value=effective*np.real(np.sqrt(inverse_modulus(f0,fit,alpha*value/1540)))
    return value

