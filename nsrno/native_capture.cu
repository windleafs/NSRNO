// Observe the existing native step; do not change its material or update.
#include "native_time.cu"

__global__ void accumulate_pressure(const float* state,const double2* phase,
 double2* fields,int capture_index,int captures,int frequencies,Config c){
 int i=blockIdx.x*blockDim.x+threadIdx.x,f=blockIdx.y,n=c.nz*c.nx;
 if(i>=n||f>=frequencies)return;
 double2 w=phase[(size_t)f*captures+capture_index];
 double2 value=fields[(size_t)f*n+i];double p=(double)state[i];
 value.x+=p*w.x;value.y+=p*w.y;fields[(size_t)f*n+i]=value;
}

extern "C" int native_capture(const Config* pc,const float* m,const float* src,
 const int* si,const float* sw,const int* ri,const float* rw,float* raw,
 const double2* phase,double2* fields,int frequencies,int stride,
 float* state,float* other,float* vp,float* coefficients,void* stream_ptr){
 Config c=*pc;cudaStream_t stream=(cudaStream_t)stream_ptr;
 size_t state_bytes=(size_t)7*c.nz*c.nx*sizeof(float);
 cudaMemsetAsync(state,0,state_bytes,stream);cudaMemsetAsync(other,0,state_bytes,stream);
 cudaMemsetAsync(fields,0,(size_t)frequencies*c.nz*c.nx*sizeof(double2),stream);
 setup<<<(c.nz*c.nx+255)/256,256,0,stream>>>(m,coefficients,c);
 int captures=(c.nt-1)/stride+1;
 for(int t=0;t<c.nt;t++){
  step(state,other,vp,coefficients,src,si,sw,t,c,stream);
  record<<<(c.nr+255)/256,256,0,stream>>>(other,raw,ri,rw,t,c);
  if(t%stride==0){
   dim3 grid((c.nz*c.nx+255)/256,frequencies);
   accumulate_pressure<<<grid,256,0,stream>>>(other,phase,fields,t/stride,captures,frequencies,c);
  }
  std::swap(state,other);
 }
 cudaError_t err=cudaGetLastError();if(err!=cudaSuccess)return (int)err;
 return (int)cudaStreamSynchronize(stream);
}
