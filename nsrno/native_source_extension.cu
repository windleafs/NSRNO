// Scalar physical volume-source extension. The original propagator is immutable.
#include "native_time.cu"

// Same state transpose as reverse_local/velocity/pressure_density, with no
// material derivative or forward-state dependency. State order is unchanged.
__global__ void extension_reverse_local(const float* adj,float* prev,const float* k,float* work,Config c){
 int i=blockIdx.x*blockDim.x+threadIdx.x,n=c.nz*c.nx;if(i>=n)return;int z=i/c.nx,x=i%c.nx;
 bool mx=x<c.pml||x>=c.nx-c.pml,mz=z<c.pml||z>=c.nz-c.pml;
 float lp=adj[i],rho=k[i];
 float qx=adj[3*n+i]*(1-mx)+k[3*n+i]*lp,qz=adj[4*n+i]*(1-mz)+k[3*n+i]*lp;
 float vx=adj[n+i]+c.dt*plus(work,z,x,0,c),vz=adj[2*n+i]+c.dt*plus(work+n,z,x,1,c);
 work[2*n+i]=vx;work[3*n+i]=vz;work[4*n+i]=rho*qx;work[5*n+i]=rho*qz;
 prev[n+i]=(k[7*n+i]-1.f)*vx;prev[2*n+i]=(k[8*n+i]-1.f)*vz;
 prev[3*n+i]=qx+mx*k[9*n+i]*adj[3*n+i];prev[4*n+i]=qz+mz*k[10*n+i]*adj[4*n+i];
 float memory=k[4*n+i]*adj[5*n+i]+k[5*n+i]*adj[6*n+i];
 prev[i]=memory;prev[5*n+i]=(1.f-c.dt/c.tau1)*adj[5*n+i]+memory-lp;
 prev[6*n+i]=(1.f-c.dt/c.tau2)*adj[6*n+i]+memory-lp;
}
__global__ void extension_reverse_velocity(float* prev,const float* k,float* work,Config c){
 int i=blockIdx.x*blockDim.x+threadIdx.x,n=c.nz*c.nx;if(i>=n)return;int z=i/c.nx,x=i%c.nx;
 float vx=work[2*n+i]+c.dt*plus(work+4*n,z,x,0,c),vz=work[3*n+i]+c.dt*plus(work+5*n,z,x,1,c);
 prev[n+i]+=vx;prev[2*n+i]+=vz;work[6*n+i]=k[n+i]*vx;work[7*n+i]=k[2*n+i]*vz;
}
__global__ void extension_reverse_pressure(float* prev,const float* work,Config c){
 int i=blockIdx.x*blockDim.x+threadIdx.x,n=c.nz*c.nx;if(i>=n)return;int z=i/c.nx,x=i%c.nx;
 prev[i]+=c.dt*(minus(work+6*n,z,x,0,c)+minus(work+7*n,z,x,1,c));
}
inline void extension_reverse(float* adj,float* prev,const float* k,float* work,Config c,cudaStream_t stream){
 int blocks=(c.nz*c.nx+255)/256;
 weighted_pml_q<<<blocks,256,0,stream>>>(adj,k,work,c);
 extension_reverse_local<<<blocks,256,0,stream>>>(adj,prev,k,work,c);
 extension_reverse_velocity<<<blocks,256,0,stream>>>(prev,k,work,c);
 extension_reverse_pressure<<<blocks,256,0,stream>>>(prev,work,c);
}
__global__ void extension_pack(const float* adj,const float* weight,float* force,double* norm,Config c){
 int i=blockIdx.x*blockDim.x+threadIdx.x,n=c.nz*c.nx;double value=0;
 if(i<n){
  float b=c.dt/(1540.f*c.dx)*weight[i],a=adj[3*n+i]+adj[4*n+i];
  if(force)force[i]=b*b*a;
  double q=(double)b*(double)a;value=q*q;
 }
 if(norm){
  __shared__ double sums[256];sums[threadIdx.x]=value;__syncthreads();
  for(int d=128;d;d>>=1){if(threadIdx.x<d)sums[threadIdx.x]+=sums[threadIdx.x+d];__syncthreads();}
  if(threadIdx.x==0)atomicAdd(norm,sums[0]);
 }
}
__global__ void extension_inject(float* state,const float* force,float scale,Config c){
 int i=blockIdx.x*blockDim.x+threadIdx.x,n=c.nz*c.nx;if(i>=n)return;
 float q=scale*force[i];state[3*n+i]+=q;state[4*n+i]+=q;
}
inline void extension_block(const Config& c,int begin,int end,const float* raw_dual,const int* ri,const float* rw,
 const float* weight,const float* cp,float* adj,float* prev,const float* k,float* work,float* force,cudaStream_t stream){
 int n=c.nz*c.nx,blocks=(n+255)/256;size_t bytes=(size_t)7*n*sizeof(float);
 cudaMemcpyAsync(adj,cp,bytes,cudaMemcpyHostToDevice,stream);
 for(int t=end-1;t>=begin;t--){
  record_transpose<<<(c.nr+255)/256,256,0,stream>>>(adj,raw_dual,ri,rw,t,c);
  extension_pack<<<blocks,256,0,stream>>>(adj,weight,force+(size_t)(t-begin)*n,nullptr,c);
  extension_reverse(adj,prev,k,work,c,stream);std::swap(adj,prev);
 }
}
extern "C" int extension_adjoint(const Config* pc,const float* m,const float* raw_dual,
 const int* ri,const float* rw,const float* weight,float* cp,float* adj,float* prev,float* k,float* work,double* norm,void* sp){
 Config c=*pc;cudaStream_t stream=(cudaStream_t)sp;int n=c.nz*c.nx,blocks=(n+255)/256;
 size_t bytes=(size_t)7*n*sizeof(float);cudaMemsetAsync(adj,0,bytes,stream);cudaMemsetAsync(prev,0,bytes,stream);
 cudaMemsetAsync(norm,0,sizeof(double),stream);setup<<<blocks,256,0,stream>>>(m,k,c);
 for(int b=(c.nt-1)/c.chunk;b>=0;b--){
  int begin=b*c.chunk,end=std::min(c.nt,begin+c.chunk);
  if(cp)cudaMemcpyAsync(cp+(size_t)b*7*n,adj,bytes,cudaMemcpyDeviceToHost,stream);
  for(int t=end-1;t>=begin;t--){
   record_transpose<<<(c.nr+255)/256,256,0,stream>>>(adj,raw_dual,ri,rw,t,c);
   extension_pack<<<blocks,256,0,stream>>>(adj,weight,nullptr,norm,c);
   extension_reverse(adj,prev,k,work,c,stream);std::swap(adj,prev);
  }
 }
 cudaError_t err=cudaGetLastError();return err!=cudaSuccess?(int)err:(int)cudaStreamSynchronize(stream);
}
extern "C" int extension_forward(const Config* pc,const float* m,const float* src,const int* si,const float* sw,
 const int* ri,const float* rw,const float* raw_dual,const float* weight,const float* cp_dual,float scale,
 float* raw,float* cp_forward,float* state,float* other,float* vp,float* k,float* adj,float* prev,float* work,float* force,void* sp){
 Config c=*pc;cudaStream_t stream=(cudaStream_t)sp;int n=c.nz*c.nx,blocks=(n+255)/256;
 size_t bytes=(size_t)7*n*sizeof(float);cudaMemsetAsync(state,0,bytes,stream);cudaMemsetAsync(other,0,bytes,stream);
 setup<<<blocks,256,0,stream>>>(m,k,c);
 for(int b=0;b<=(c.nt-1)/c.chunk;b++){
  int begin=b*c.chunk,end=std::min(c.nt,begin+c.chunk);
  extension_block(c,begin,end,raw_dual,ri,rw,weight,cp_dual+(size_t)b*7*n,adj,prev,k,work,force,stream);
  if(cp_forward)cudaMemcpyAsync(cp_forward+(size_t)b*7*n,state,bytes,cudaMemcpyDeviceToHost,stream);
  for(int t=begin;t<end;t++){
   step(state,other,vp,k,src,si,sw,t,c,stream);
   extension_inject<<<blocks,256,0,stream>>>(other,force+(size_t)(t-begin)*n,scale,c);
   record<<<(c.nr+255)/256,256,0,stream>>>(other,raw,ri,rw,t,c);std::swap(state,other);
  }
 }
 cudaError_t err=cudaGetLastError();return err!=cudaSuccess?(int)err:(int)cudaStreamSynchronize(stream);
}
extern "C" int extension_backward(const Config* pc,const float* m,const float* src,const int* si,const float* sw,
 const int* ri,const float* rw,const float* raw_dual,const float* weight,const float* cp_dual,const float* cp_forward,float scale,
 float* state,float* other,float* vp,float* k,float* adj,float* prev,float* work,float* force,float* tape,double* grad,void* sp){
 Config c=*pc;cudaStream_t stream=(cudaStream_t)sp;int n=c.nz*c.nx,blocks=(n+255)/256;
 size_t bytes=(size_t)7*n*sizeof(float);cudaMemsetAsync(grad,0,(size_t)3*n*sizeof(double),stream);
 setup<<<blocks,256,0,stream>>>(m,k,c);
 for(int b=(c.nt-1)/c.chunk;b>=0;b--){
  int begin=b*c.chunk,end=std::min(c.nt,begin+c.chunk);
  extension_block(c,begin,end,raw_dual,ri,rw,weight,cp_dual+(size_t)b*7*n,adj,prev,k,work,force,stream);
  cudaMemcpyAsync(state,cp_forward+(size_t)b*7*n,bytes,cudaMemcpyHostToDevice,stream);
  for(int t=begin;t<end;t++){
   cudaMemcpyAsync(tape+(size_t)(t-begin)*7*n,state,bytes,cudaMemcpyDeviceToDevice,stream);
   step(state,other,vp,k,src,si,sw,t,c,stream);
   extension_inject<<<blocks,256,0,stream>>>(other,force+(size_t)(t-begin)*n,scale,c);std::swap(state,other);
  }
  // Reuse the original dual checkpoint; no different RHS or new adjoint solve.
  cudaMemcpyAsync(adj,cp_dual+(size_t)b*7*n,bytes,cudaMemcpyHostToDevice,stream);
  for(int t=end-1;t>=begin;t--){
   const float* old=tape+(size_t)(t-begin)*7*n;
   velocity<<<blocks,256,0,stream>>>(old,other,vp,k,c);
   record_transpose<<<(c.nr+255)/256,256,0,stream>>>(adj,raw_dual,ri,rw,t,c);
   weighted_pml_q<<<blocks,256,0,stream>>>(adj,k,work,c);
   reverse_local<<<blocks,256,0,stream>>>(m,old,other,vp,adj,prev,k,work,grad,c);
   reverse_velocity<<<blocks,256,0,stream>>>(old,prev,k,work,c);
   reverse_pressure_density<<<blocks,256,0,stream>>>(prev,k,work,grad,c);std::swap(adj,prev);
  }
 }
 cudaError_t err=cudaGetLastError();return err!=cudaSuccess?(int)err:(int)cudaStreamSynchronize(stream);
}
