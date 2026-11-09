// Directional derivative of the existing discrete acoustic solver.
// The original solver and transpose stay unchanged and are compiled here.
#include "native_time.cu"

__global__ void tangent_setup(const float* m,const float* dm,const float* k,float* dk,Config c){
 int i=blockIdx.x*blockDim.x+threadIdx.x,n=c.nz*c.nx;if(i>=n)return;
 int z=i/c.nx,x=i%c.nx,ix=z*c.nx+min(x+1,c.nx-1),iz=min(z+1,c.nz-1)*c.nx+x;
 float speed=m[i],alpha=m[2*n+i],ds=dm[i],da=dm[2*n+i];
 float de1=(alpha*ds+speed*da)/1540.f*c.e1,de2=(alpha*ds+speed*da)/1540.f*c.e2;
 float A=k[6*n+i],dA=sqrtf(A)*(de1+de2);
 dk[i]=dm[n+i];
 dk[n+i]=-.5f*k[n+i]*k[n+i]*(dm[n+i]+dm[n+ix]);
 dk[2*n+i]=-.5f*k[2*n+i]*k[2*n+i]*(dm[n+i]+dm[n+iz]);
 dk[3*n+i]=2.f*speed*A*ds+speed*speed*dA;
 dk[4*n+i]=c.dt/(c.tau1*A)*de1-k[4*n+i]*dA/A;
 dk[5*n+i]=c.dt/(c.tau2*A)*de2-k[5*n+i]*dA/A;
 dk[6*n+i]=dA;
}

__global__ void tangent_velocity(const float* old,const float* d_old,float* d_out,float* d_vp,
 const float* k,const float* dk,Config c){
 int i=blockIdx.x*blockDim.x+threadIdx.x,n=c.nz*c.nx;if(i>=n)return;int z=i/c.nx,x=i%c.nx;
 float vx=d_old[n+i]-c.dt*(dk[n+i]*plus(old,z,x,0,c)+k[n+i]*plus(d_old,z,x,0,c));
 float vz=d_old[2*n+i]-c.dt*(dk[2*n+i]*plus(old,z,x,1,c)+k[2*n+i]*plus(d_old,z,x,1,c));
 d_vp[i]=vx;d_vp[n+i]=vz;
 d_out[n+i]=vx+(k[7*n+i]-1.f)*d_old[n+i];
 d_out[2*n+i]=vz+(k[8*n+i]-1.f)*d_old[2*n+i];
}

__global__ void tangent_pressure(const float* old,const float* d_old,float* d_out,
 const float* vp,const float* d_vp,const float* k,const float* dk,Config c){
 int i=blockIdx.x*blockDim.x+threadIdx.x,n=c.nz*c.nx;if(i>=n)return;int z=i/c.nx,x=i%c.nx;
 float dx=minus(vp,z,x,0,c),dz=minus(vp+n,z,x,1,c);
 float qx=old[3*n+i]-c.dt*k[i]*dx,qz=old[4*n+i]-c.dt*k[i]*dz;
 float dqx=d_old[3*n+i]-c.dt*(dk[i]*dx+k[i]*minus(d_vp,z,x,0,c));
 float dqz=d_old[4*n+i]-c.dt*(dk[i]*dz+k[i]*minus(d_vp+n,z,x,1,c));
 d_out[3*n+i]=dqx;d_out[4*n+i]=dqz;
 d_out[i]=dk[3*n+i]*(qx+qz)+k[3*n+i]*(dqx+dqz)-d_old[5*n+i]-d_old[6*n+i];
 float f=old[i]+old[5*n+i]+old[6*n+i],df=d_old[i]+d_old[5*n+i]+d_old[6*n+i];
 d_out[5*n+i]=(1.f-c.dt/c.tau1)*d_old[5*n+i]+dk[4*n+i]*f+k[4*n+i]*df;
 d_out[6*n+i]=(1.f-c.dt/c.tau2)*d_old[6*n+i]+dk[5*n+i]*f+k[5*n+i]*df;
}

__global__ void tangent_density_pml(const float* d_old,const float* out,float* d_out,
 const float* k,const float* dk,Config c){
 int i=blockIdx.x*blockDim.x+threadIdx.x,n=c.nz*c.nx;if(i>=n)return;int z=i/c.nx,x=i%c.nx;
 if(x<c.pml||x>=c.nx-c.pml)
  d_out[3*n+i]=k[9*n+i]*d_old[3*n+i]-c.dt*(dk[i]*minus(out+n,z,x,0,c)+k[i]*minus(d_out+n,z,x,0,c));
 if(z<c.pml||z>=c.nz-c.pml)
  d_out[4*n+i]=k[10*n+i]*d_old[4*n+i]-c.dt*(dk[i]*minus(out+2*n,z,x,1,c)+k[i]*minus(d_out+2*n,z,x,1,c));
}

extern "C" int native_tangent(const Config* pc,const float* m,const float* dm,
 const float* src,const int* si,const float* sw,const int* ri,const float* rw,
 float* raw,float* d_raw,float* checkpoints,float* state,float* other,float* vp,
 float* d_state,float* d_other,float* d_vp,float* k,float* dk,void* stream_ptr){
 Config c=*pc;cudaStream_t stream=(cudaStream_t)stream_ptr;int blocks=(c.nz*c.nx+255)/256;
 size_t bytes=(size_t)7*c.nz*c.nx*sizeof(float);
 cudaMemsetAsync(state,0,bytes,stream);cudaMemsetAsync(other,0,bytes,stream);
 cudaMemsetAsync(d_state,0,bytes,stream);cudaMemsetAsync(d_other,0,bytes,stream);
 setup<<<blocks,256,0,stream>>>(m,k,c);
 tangent_setup<<<blocks,256,0,stream>>>(m,dm,k,dk,c);
 for(int t=0;t<c.nt;t++){
  if(checkpoints && t%c.chunk==0)
   cudaMemcpyAsync(checkpoints+(size_t)(t/c.chunk)*7*c.nz*c.nx,state,bytes,cudaMemcpyDeviceToHost,stream);
  velocity<<<blocks,256,0,stream>>>(state,other,vp,k,c);
  tangent_velocity<<<blocks,256,0,stream>>>(state,d_state,d_other,d_vp,k,dk,c);
  pressure_memory<<<blocks,256,0,stream>>>(state,other,vp,k,c);
  tangent_pressure<<<blocks,256,0,stream>>>(state,d_state,d_other,vp,d_vp,k,dk,c);
  density_pml<<<blocks,256,0,stream>>>(state,other,k,c);
  tangent_density_pml<<<blocks,256,0,stream>>>(d_state,other,d_other,k,dk,c);
  source_add<<<(c.ns+255)/256,256,0,stream>>>(other,src,si,sw,t,c);
  record<<<(c.nr+255)/256,256,0,stream>>>(other,raw,ri,rw,t,c);
  record<<<(c.nr+255)/256,256,0,stream>>>(d_other,d_raw,ri,rw,t,c);
  std::swap(state,other);std::swap(d_state,d_other);
 }
 cudaError_t err=cudaGetLastError();if(err!=cudaSuccess)return (int)err;
 return (int)cudaStreamSynchronize(stream);
}
