// Four residual channels; the original propagator and its state transpose stay unchanged.
#include "native_source_extension.cu"

__global__ void multi_pack(const float* adj,const float* work,const float* node,
 const float* face_x,const float* face_z,float* force,double* norm,Config c){
 int i=blockIdx.x*blockDim.x+threadIdx.x,n=c.nz*c.nx;double value=0;
 if(i<n){
  float b1=c.dt*c.e1/c.tau1,b2=c.dt*c.e2/c.tau2,bn=sqrtf(b1*b1+b2*b2);
  float p=node[i]*adj[i],x=face_x[i]*work[6*n+i]/1540.f,z=face_z[i]*work[7*n+i]/1540.f;
  float m=node[i]*(b1*adj[5*n+i]+b2*adj[6*n+i])/bn;
  if(force){force[i]=p;force[n+i]=x;force[2*n+i]=z;force[3*n+i]=m;}
  value=(double)p*p+(double)x*x+(double)z*z+(double)m*m;
 }
 if(norm){
  __shared__ double sums[256];sums[threadIdx.x]=value;__syncthreads();
  for(int d=128;d;d>>=1){if(threadIdx.x<d)sums[threadIdx.x]+=sums[threadIdx.x+d];__syncthreads();}
  if(threadIdx.x==0)atomicAdd(norm,sums[0]);
 }
}

__global__ void multi_add_velocity(float* out,float* vp,const float* k,const float* force,
 const float* face_x,const float* face_z,float scale,Config c){
 int i=blockIdx.x*blockDim.x+threadIdx.x,n=c.nz*c.nx;if(i>=n)return;
 float x=scale*face_x[i]*k[n+i]*force[n+i]/1540.f;
 float z=scale*face_z[i]*k[2*n+i]*force[2*n+i]/1540.f;
 vp[i]+=x;vp[n+i]+=z;out[n+i]+=x;out[2*n+i]+=z;
}
__global__ void multi_add_pressure_memory(float* out,const float* force,const float* node,float scale,Config c){
 int i=blockIdx.x*blockDim.x+threadIdx.x,n=c.nz*c.nx;if(i>=n)return;
 float b1=c.dt*c.e1/c.tau1,b2=c.dt*c.e2/c.tau2,bn=sqrtf(b1*b1+b2*b2);
 float a=scale*node[i];out[i]+=a*force[i];
 out[5*n+i]+=a*b1/bn*force[3*n+i];out[6*n+i]+=a*b2/bn*force[3*n+i];
}
inline void multi_step(float* old,float* next,float* vp,const float* k,const float* src,
 const int* si,const float* sw,int t,const float* force,const float* node,const float* fx,const float* fz,
 float scale,Config c,cudaStream_t stream){
 int blocks=(c.nz*c.nx+255)/256;
 velocity<<<blocks,256,0,stream>>>(old,next,vp,k,c);
 if(force&&scale!=0.f)multi_add_velocity<<<blocks,256,0,stream>>>(next,vp,k,force,fx,fz,scale,c);
 pressure_memory<<<blocks,256,0,stream>>>(old,next,vp,k,c);
 if(force&&scale!=0.f)multi_add_pressure_memory<<<blocks,256,0,stream>>>(next,force,node,scale,c);
 density_pml<<<blocks,256,0,stream>>>(old,next,k,c);
 source_add<<<(c.ns+255)/256,256,0,stream>>>(next,src,si,sw,t,c);
}

// Centered conditional quadratic: [aK,gK,aR,gR,aFx,gFx,aFz,gFz].
// Known residual forcing supplies the linear terms without cancellation of large normal equations.
__global__ void multi_statistics(const float* old,const float* next,const float* vp,const float* k,
 const float* force,const float* node,const float* fx,const float* fz,float scale,double* stats,Config c){
 int i=blockIdx.x*blockDim.x+threadIdx.x,n=c.nz*c.nx;if(i>=n)return;
 double wp=force?(double)scale*node[i]*force[i]:0.;
 double wx=force?(double)scale*fx[i]*force[n+i]:0.;
 double wz=force?(double)scale*fz[i]*force[2*n+i]:0.;
 double wm=force?(double)scale*node[i]*force[3*n+i]:0.;
 if(node[i]!=0.f){
  double ds=((double)next[3*n+i]+next[4*n+i])/(double)k[i];
  double f=(double)old[i]+old[5*n+i]+old[6*n+i];
  double b1=(double)c.dt*c.e1/c.tau1,b2=(double)c.dt*c.e2/c.tau2,bn=sqrt(b1*b1+b2*b2);
  stats[i]+=ds*ds;stats[n+i]-=ds*wp;
  stats[2*n+i]+=(b1*b1+b2*b2)*f*f;stats[3*n+i]-=bn*f*wm;
 }
 if(fx[i]!=0.f){double u=770.*((double)vp[i]-old[n+i]);stats[4*n+i]+=u*u;stats[5*n+i]+=u*wx;}
 if(fz[i]!=0.f){double u=770.*((double)vp[n+i]-old[2*n+i]);stats[6*n+i]+=u*u;stats[7*n+i]+=u*wz;}
}

inline void multi_block(Config c,int begin,int end,const float* raw_dual,const int* ri,const float* rw,
 const float* node,const float* fx,const float* fz,const float* cp,float* adj,float* prev,
 const float* k,float* work,float* force,cudaStream_t stream){
 int n=c.nz*c.nx,blocks=(n+255)/256;size_t bytes=(size_t)7*n*sizeof(float);
 cudaMemcpyAsync(adj,cp,bytes,cudaMemcpyHostToDevice,stream);
 for(int t=end-1;t>=begin;t--){
  record_transpose<<<(c.nr+255)/256,256,0,stream>>>(adj,raw_dual,ri,rw,t,c);
  extension_reverse(adj,prev,k,work,c,stream);
  multi_pack<<<blocks,256,0,stream>>>(adj,work,node,fx,fz,force+(size_t)(t-begin)*4*n,nullptr,c);
  std::swap(adj,prev);
 }
}

extern "C" int multi_adjoint(const Config* pc,const float* m,const float* raw_dual,const int* ri,const float* rw,
 const float* node,const float* fx,const float* fz,float* cp,float* adj,float* prev,float* k,float* work,
 double* norm,float* full_force,void* sp){
 Config c=*pc;cudaStream_t stream=(cudaStream_t)sp;int n=c.nz*c.nx,blocks=(n+255)/256;
 size_t bytes=(size_t)7*n*sizeof(float);cudaMemsetAsync(adj,0,bytes,stream);cudaMemsetAsync(prev,0,bytes,stream);
 cudaMemsetAsync(norm,0,sizeof(double),stream);setup<<<blocks,256,0,stream>>>(m,k,c);
 for(int b=(c.nt-1)/c.chunk;b>=0;b--){
  int begin=b*c.chunk,end=std::min(c.nt,begin+c.chunk);
  if(cp)cudaMemcpyAsync(cp+(size_t)b*7*n,adj,bytes,cudaMemcpyDeviceToHost,stream);
  for(int t=end-1;t>=begin;t--){
   record_transpose<<<(c.nr+255)/256,256,0,stream>>>(adj,raw_dual,ri,rw,t,c);
   extension_reverse(adj,prev,k,work,c,stream);
   multi_pack<<<blocks,256,0,stream>>>(adj,work,node,fx,fz,full_force?full_force+(size_t)t*4*n:nullptr,norm,c);
   std::swap(adj,prev);
  }
 }
 cudaError_t e=cudaGetLastError();return e!=cudaSuccess?(int)e:(int)cudaStreamSynchronize(stream);
}

extern "C" int multi_forward(const Config* pc,const float* m,const float* src,const int* si,const float* sw,
 const int* ri,const float* rw,const float* raw_dual,const float* node,const float* fx,const float* fz,
 const float* cp_dual,float scale,float* raw,float* state,float* other,float* vp,float* k,
 float* adj,float* prev,float* work,float* force,double* stats,void* sp){
 Config c=*pc;cudaStream_t stream=(cudaStream_t)sp;int n=c.nz*c.nx,blocks=(n+255)/256;
 size_t bytes=(size_t)7*n*sizeof(float);cudaMemsetAsync(state,0,bytes,stream);cudaMemsetAsync(other,0,bytes,stream);
 if(stats)cudaMemsetAsync(stats,0,(size_t)8*n*sizeof(double),stream);
 setup<<<blocks,256,0,stream>>>(m,k,c);
 for(int b=0;b<=(c.nt-1)/c.chunk;b++){
  int begin=b*c.chunk,end=std::min(c.nt,begin+c.chunk);
  multi_block(c,begin,end,raw_dual,ri,rw,node,fx,fz,cp_dual+(size_t)b*7*n,adj,prev,k,work,force,stream);
  for(int t=begin;t<end;t++){
   const float* w=force+(size_t)(t-begin)*4*n;
   multi_step(state,other,vp,k,src,si,sw,t,w,node,fx,fz,scale,c,stream);
   if(stats)multi_statistics<<<blocks,256,0,stream>>>(state,other,vp,k,w,node,fx,fz,scale,stats,c);
   record<<<(c.nr+255)/256,256,0,stream>>>(other,raw,ri,rw,t,c);std::swap(state,other);
  }
 }
 cudaError_t e=cudaGetLastError();return e!=cudaSuccess?(int)e:(int)cudaStreamSynchronize(stream);
}

// Explicit forcing is used only by the one small, synthetic operator qualification.
extern "C" int multi_explicit(const Config* pc,const float* m,const float* src,const int* si,const float* sw,
 const int* ri,const float* rw,const float* node,const float* fx,const float* fz,const float* force,
 float* raw,float* state,float* other,float* vp,float* k,double* stats,void* sp){
 Config c=*pc;cudaStream_t stream=(cudaStream_t)sp;int n=c.nz*c.nx,blocks=(n+255)/256;
 size_t bytes=(size_t)7*n*sizeof(float);cudaMemsetAsync(state,0,bytes,stream);cudaMemsetAsync(other,0,bytes,stream);
 if(stats)cudaMemsetAsync(stats,0,(size_t)8*n*sizeof(double),stream);
 setup<<<blocks,256,0,stream>>>(m,k,c);
 for(int t=0;t<c.nt;t++){
  const float* w=force?force+(size_t)t*4*n:nullptr;
  multi_step(state,other,vp,k,src,si,sw,t,w,node,fx,fz,w?1.f:0.f,c,stream);
  if(stats)multi_statistics<<<blocks,256,0,stream>>>(state,other,vp,k,w,node,fx,fz,w?1.f:0.f,stats,c);
  record<<<(c.nr+255)/256,256,0,stream>>>(other,raw,ri,rw,t,c);std::swap(state,other);
 }
 cudaError_t e=cudaGetLastError();return e!=cudaSuccess?(int)e:(int)cudaStreamSynchronize(stream);
}
