// Exact-order split-density/two-relaxation acoustic steps and their transpose.
// Plain CUDA ABI: tensor storage and current stream are owned by Python/PyTorch.
#include <cuda_runtime.h>
#include <stdint.h>
#include <algorithm>

struct Config { int nz,nx,nt,ns,nr,pml,chunk; float dx,dt,tau1,tau2,e1,e2; };
__device__ __constant__ float FD[4]={1.1962890625f,-0.07975260416666667f,0.0095703125f,-0.0006975446428571429f};
__device__ inline float read0(const float* p,int z,int x,Config c){return z<0||x<0||z>=c.nz||x>=c.nx?0.f:p[z*c.nx+x];}
__device__ inline float plus(const float* p,int z,int x,int axis,Config c){
 float v=0; for(int j=0;j<4;j++) v+=FD[j]*(axis?read0(p,z+j+1,x,c)-read0(p,z-j,x,c):read0(p,z,x+j+1,c)-read0(p,z,x-j,c)); return v/c.dx;
}
__device__ inline float minus(const float* p,int z,int x,int axis,Config c){
 float v=0; for(int j=0;j<4;j++) v+=FD[j]*(axis?read0(p,z+j,x,c)-read0(p,z-j-1,x,c):read0(p,z,x+j,c)-read0(p,z,x-j-1,c)); return v/c.dx;
}
__device__ inline float damp(int pos,int n,int pml,float shift,Config c){
 float a=0; if(pos<pml) a=1.f-(pos+shift)/pml;
 if(pos>=n-pml) a=1.f-(n-1-pos-shift)/pml;
 float q=a*a; return 1.f-c.dt*(2.f*1540.f/c.dx)*q*q;
}
__global__ void setup(const float* m,float* k,Config c){
 int i=blockIdx.x*blockDim.x+threadIdx.x, n=c.nz*c.nx; if(i>=n)return;
 int z=i/c.nx,x=i%c.nx; float speed=m[i],rho=m[n+i],a=m[2*n+i];
 float eta1=a*speed/1540.f*c.e1,eta2=a*speed/1540.f*c.e2;
 float root=1.f+.5f*eta1+.5f*eta2,A=root*root;
 k[i]=rho; k[n+i]=2.f/(rho+m[n+z*c.nx+min(x+1,c.nx-1)]);
 k[2*n+i]=2.f/(rho+m[n+min(z+1,c.nz-1)*c.nx+x]);
 k[3*n+i]=speed*speed*A; k[4*n+i]=c.dt*eta1/(c.tau1*A); k[5*n+i]=c.dt*eta2/(c.tau2*A); k[6*n+i]=A;
 k[7*n+i]=damp(x,c.nx,c.pml,.5f,c); k[8*n+i]=damp(z,c.nz,c.pml,.5f,c);
 k[9*n+i]=damp(x,c.nx,c.pml,0.f,c); k[10*n+i]=damp(z,c.nz,c.pml,0.f,c);
}
// State order p,vx,vz,qx,qz,S1,S2. p is formed before PML/source overwrites.
__global__ void velocity(const float* old,float* out,float* vp,const float* k,Config c){
 int i=blockIdx.x*blockDim.x+threadIdx.x,n=c.nz*c.nx;if(i>=n)return; int z=i/c.nx,x=i%c.nx;
 float vx=old[n+i]-c.dt*k[n+i]*plus(old,z,x,0,c),vz=old[2*n+i]-c.dt*k[2*n+i]*plus(old,z,x,1,c);
 vp[i]=vx;vp[n+i]=vz;out[n+i]=vx+(k[7*n+i]-1.f)*old[n+i];out[2*n+i]=vz+(k[8*n+i]-1.f)*old[2*n+i];
}
__global__ void pressure_memory(const float* old,float* out,const float* vp,const float* k,Config c){
 int i=blockIdx.x*blockDim.x+threadIdx.x,n=c.nz*c.nx;if(i>=n)return;int z=i/c.nx,x=i%c.nx;
 float qx=old[3*n+i]-c.dt*k[i]*minus(vp,z,x,0,c),qz=old[4*n+i]-c.dt*k[i]*minus(vp+n,z,x,1,c);
 out[3*n+i]=qx;out[4*n+i]=qz;out[i]=k[3*n+i]*(qx+qz)-old[5*n+i]-old[6*n+i];
 float f=old[i]+old[5*n+i]+old[6*n+i];
 out[5*n+i]=(1.f-c.dt/c.tau1)*old[5*n+i]+k[4*n+i]*f;
 out[6*n+i]=(1.f-c.dt/c.tau2)*old[6*n+i]+k[5*n+i]*f;
}
__global__ void density_pml(const float* old,float* out,const float* k,Config c){
 int i=blockIdx.x*blockDim.x+threadIdx.x,n=c.nz*c.nx;if(i>=n)return;int z=i/c.nx,x=i%c.nx;
 if(x<c.pml||x>=c.nx-c.pml)out[3*n+i]=k[9*n+i]*old[3*n+i]-c.dt*k[i]*minus(out+n,z,x,0,c);
 if(z<c.pml||z>=c.nz-c.pml)out[4*n+i]=k[10*n+i]*old[4*n+i]-c.dt*k[i]*minus(out+2*n,z,x,1,c);
}
__global__ void source_add(float* out,const float* src,const int* index,const float* weight,int t,Config c){
 int s=blockIdx.x*blockDim.x+threadIdx.x,n=c.nz*c.nx;if(s>=c.ns)return;
 float v=src[(size_t)t*c.ns+s]*c.dt/(1540.f*c.dx);
 for(int j=0;j<4;j++){int i=index[s*4+j];float w=v*weight[s*4+j];if(w!=0){atomicAdd(out+3*n+i,w);atomicAdd(out+4*n+i,w);}}
}
__global__ void record(const float* out,float* raw,const int* index,const float* weight,int t,Config c){
 int r=blockIdx.x*blockDim.x+threadIdx.x;if(r>=c.nr)return;float v=0;
 for(int j=0;j<4;j++)v+=out[index[4*r+j]]*weight[4*r+j];raw[(size_t)t*c.nr+r]=v;
}
__global__ void record_transpose(float* lp,const float* residual,const int* index,const float* weight,int t,Config c){
 int r=blockIdx.x*blockDim.x+threadIdx.x;if(r>=c.nr)return;float v=residual[(size_t)t*c.nr+r];
 for(int j=0;j<4;j++){float w=v*weight[r*4+j];if(w!=0)atomicAdd(lp+index[r*4+j],w);}
}
__global__ void weighted_pml_q(const float* adj,const float* k,float* work,Config c){
 int i=blockIdx.x*blockDim.x+threadIdx.x,n=c.nz*c.nx;if(i>=n)return;int z=i/c.nx,x=i%c.nx;
 work[i]=k[i]*adj[3*n+i]*(x<c.pml||x>=c.nx-c.pml);work[n+i]=k[i]*adj[4*n+i]*(z<c.pml||z>=c.nz-c.pml);
}
// work: W0(2), adjVd(2), WqStar(2), b*adjVStar(2), adjB(2).
__global__ void reverse_local(const float* m,const float* old,const float* vd,const float* vp,
 const float* adj,float* prev,const float* k,float* work,double* grad,Config c){
 int i=blockIdx.x*blockDim.x+threadIdx.x,n=c.nz*c.nx;if(i>=n)return;int z=i/c.nx,x=i%c.nx;
 bool mx=x<c.pml||x>=c.nx-c.pml,mz=z<c.pml||z>=c.nz-c.pml;
 float lp=adj[i],rho=k[i],A=k[6*n+i],speed=m[i],alpha=m[2*n+i];
 float qx=adj[3*n+i]*(1-mx)+k[3*n+i]*lp,qz=adj[4*n+i]*(1-mz)+k[3*n+i]*lp;
 float vx=adj[n+i]+c.dt*plus(work,z,x,0,c),vz=adj[2*n+i]+c.dt*plus(work+n,z,x,1,c);
 work[2*n+i]=vx;work[3*n+i]=vz;work[4*n+i]=rho*qx;work[5*n+i]=rho*qz;
 prev[n+i]=(k[7*n+i]-1.f)*vx;prev[2*n+i]=(k[8*n+i]-1.f)*vz;
 prev[3*n+i]=qx+mx*k[9*n+i]*adj[3*n+i];prev[4*n+i]=qz+mz*k[10*n+i]*adj[4*n+i];
 float memory=k[4*n+i]*adj[5*n+i]+k[5*n+i]*adj[6*n+i];
 prev[i]=memory;prev[5*n+i]=(1.f-c.dt/c.tau1)*adj[5*n+i]+memory-lp;
 prev[6*n+i]=(1.f-c.dt/c.tau2)*adj[6*n+i]+memory-lp;
 float dx=minus(vp,z,x,0,c),dz=minus(vp+n,z,x,1,c);
 float sumq=old[3*n+i]+old[4*n+i]-c.dt*rho*(dx+dz);
 float f=old[i]+old[5*n+i]+old[6*n+i];
 float gA=speed*speed*sumq*lp-f/A*memory,root=sqrtf(A);
 float ge1=c.dt/(c.tau1*A)*f*adj[5*n+i]+root*gA;
 float ge2=c.dt/(c.tau2*A)*f*adj[6*n+i]+root*gA;
 float combined=c.e1*ge1+c.e2*ge2;
 grad[i]+=(double)(2.f*speed*A*sumq*lp+alpha/1540.f*combined);
 grad[2*n+i]+=(double)(speed/1540.f*combined);
 float gr=-c.dt*(qx*dx+qz*dz+mx*adj[3*n+i]*minus(vd+n,z,x,0,c)+mz*adj[4*n+i]*minus(vd+2*n,z,x,1,c));
 grad[n+i]+=(double)gr;
}
__global__ void reverse_velocity(const float* old,float* prev,const float* k,float* work,Config c){
 int i=blockIdx.x*blockDim.x+threadIdx.x,n=c.nz*c.nx;if(i>=n)return;int z=i/c.nx,x=i%c.nx;
 float vx=work[2*n+i]+c.dt*plus(work+4*n,z,x,0,c),vz=work[3*n+i]+c.dt*plus(work+5*n,z,x,1,c);
 prev[n+i]+=vx;prev[2*n+i]+=vz;work[6*n+i]=k[n+i]*vx;work[7*n+i]=k[2*n+i]*vz;
 work[8*n+i]=-c.dt*vx*plus(old,z,x,0,c);work[9*n+i]=-c.dt*vz*plus(old,z,x,1,c);
}
__global__ void reverse_pressure_density(float* prev,const float* k,const float* work,double* grad,Config c){
 int i=blockIdx.x*blockDim.x+threadIdx.x,n=c.nz*c.nx;if(i>=n)return;int z=i/c.nx,x=i%c.nx;
 prev[i]+=c.dt*(minus(work+6*n,z,x,0,c)+minus(work+7*n,z,x,1,c));
 float value=-.5f*k[n+i]*k[n+i]*work[8*n+i]*(x==c.nx-1?2.f:1.f);
 value+=-.5f*k[2*n+i]*k[2*n+i]*work[9*n+i]*(z==c.nz-1?2.f:1.f);
 if(x>0)value+=-.5f*k[n+i-1]*k[n+i-1]*work[8*n+i-1];
 if(z>0)value+=-.5f*k[2*n+i-c.nx]*k[2*n+i-c.nx]*work[9*n+i-c.nx];
 grad[n+i]+=(double)value;
}

inline void step(float* old,float* next,float* vp,const float* k,const float* src,const int* si,const float* sw,int t,Config c,cudaStream_t stream){
 int blocks=(c.nz*c.nx+255)/256;
 velocity<<<blocks,256,0,stream>>>(old,next,vp,k,c);
 pressure_memory<<<blocks,256,0,stream>>>(old,next,vp,k,c);
 density_pml<<<blocks,256,0,stream>>>(old,next,k,c);
 source_add<<<(c.ns+255)/256,256,0,stream>>>(next,src,si,sw,t,c);
}
extern "C" int native_forward(const Config* pc,const float* m,const float* src,const int* si,const float* sw,
 const int* ri,const float* rw,float* raw,float* checkpoints,float* state,float* other,float* vp,float* k,void* stream_ptr){
 Config c=*pc;cudaStream_t stream=(cudaStream_t)stream_ptr;size_t bytes=(size_t)7*c.nz*c.nx*sizeof(float);
 cudaMemsetAsync(state,0,bytes,stream);cudaMemsetAsync(other,0,bytes,stream);
 setup<<<(c.nz*c.nx+255)/256,256,0,stream>>>(m,k,c);
 for(int t=0;t<c.nt;t++){
  if(checkpoints && t%c.chunk==0)cudaMemcpyAsync(checkpoints+(size_t)(t/c.chunk)*7*c.nz*c.nx,state,bytes,cudaMemcpyDeviceToHost,stream);
  step(state,other,vp,k,src,si,sw,t,c,stream);
  record<<<(c.nr+255)/256,256,0,stream>>>(other,raw,ri,rw,t,c);std::swap(state,other);
 }
 cudaError_t err=cudaGetLastError();if(err!=cudaSuccess)return (int)err;return (int)cudaStreamSynchronize(stream);
}
extern "C" int native_backward(const Config* pc,const float* m,const float* src,const int* si,const float* sw,
 const int* ri,const float* rw,const float* residual,const float* checkpoints,float* state,float* other,float* vp,
 float* k,float* tape,float* adj,float* prev,float* work,double* grad,void* stream_ptr){
 Config c=*pc;cudaStream_t stream=(cudaStream_t)stream_ptr;int n=c.nz*c.nx,blocks=(n+255)/256;
 size_t bytes=(size_t)7*n*sizeof(float);cudaMemsetAsync(adj,0,bytes,stream);cudaMemsetAsync(prev,0,bytes,stream);
 cudaMemsetAsync(grad,0,(size_t)3*n*sizeof(double),stream);setup<<<blocks,256,0,stream>>>(m,k,c);
 for(int block=(c.nt-1)/c.chunk;block>=0;block--){
  int begin=block*c.chunk,end=std::min(c.nt,begin+c.chunk),length=end-begin;
  cudaMemcpyAsync(state,checkpoints+(size_t)block*7*n,bytes,cudaMemcpyHostToDevice,stream);
  for(int j=0;j<length;j++){
   cudaMemcpyAsync(tape+(size_t)j*7*n,state,bytes,cudaMemcpyDeviceToDevice,stream);
   step(state,other,vp,k,src,si,sw,begin+j,c,stream);std::swap(state,other);
  }
  for(int j=length-1;j>=0;j--){
   const float* old=tape+(size_t)j*7*n;
   velocity<<<blocks,256,0,stream>>>(old,other,vp,k,c);
   record_transpose<<<(c.nr+255)/256,256,0,stream>>>(adj,residual,ri,rw,begin+j,c);
   weighted_pml_q<<<blocks,256,0,stream>>>(adj,k,work,c);
   reverse_local<<<blocks,256,0,stream>>>(m,old,other,vp,adj,prev,k,work,grad,c);
   reverse_velocity<<<blocks,256,0,stream>>>(old,prev,k,work,c);
   reverse_pressure_density<<<blocks,256,0,stream>>>(prev,k,work,grad,c);
   std::swap(adj,prev);
  }
 }
 cudaError_t err=cudaGetLastError();if(err!=cudaSuccess)return (int)err;return (int)cudaStreamSynchronize(stream);
}
extern "C" const char* native_error(int code){return cudaGetErrorString((cudaError_t)code);}
