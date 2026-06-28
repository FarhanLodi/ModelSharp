// Standalone parity + benchmark test for ms_attention_f32.
#include "../ms_kernels.h"
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <random>
#include <chrono>
#include <algorithm>
#ifdef _OPENMP
#include <omp.h>
#endif

static void ref_attention(const float* q, const float* k, const float* v, float* out,
                          int BH, int Sq, int Sk, int D, float scale, int causal) {
    int offset = causal ? (Sk - Sq) : 0;
    std::vector<float> s(Sk);
    for (int bh = 0; bh < BH; ++bh) {
        for (int i = 0; i < Sq; ++i) {
            const float* qrow = q + ((size_t)bh*Sq + i)*D;
            int kmax = causal ? std::min(Sk, i + offset + 1) : Sk;
            float* orow = out + ((size_t)bh*Sq + i)*D;
            if (kmax <= 0) { for (int d=0;d<D;++d) orow[d]=0.f; continue; }
            float m = -INFINITY;
            for (int j=0;j<kmax;++j){
                float dot=0.f;
                for (int d=0;d<D;++d) dot += qrow[d]*k[((size_t)bh*Sk+j)*D+d];
                s[j]=dot*scale; if (s[j]>m) m=s[j];
            }
            float l=0.f;
            for (int j=0;j<kmax;++j){ s[j]=std::exp(s[j]-m); l+=s[j]; }
            for (int d=0;d<D;++d){
                float acc=0.f;
                for (int j=0;j<kmax;++j) acc += s[j]*v[((size_t)bh*Sk+j)*D+d];
                orow[d]=acc/l;
            }
        }
    }
}

// Combined abs/rel error: pass if |a-b| <= atol + rtol*|b| anywhere; report the
// worst rel error among entries that exceed the abs floor. Softmax outputs live
// in [-1,1] and can be near zero, so a tiny absolute slack avoids false fails
// from -ffast-math reassociation on near-cancelling sums.
static float relerr(const std::vector<float>&a,const std::vector<float>&b){
    const float atol=1e-3f;
    float maxr=0.f;
    for (size_t i=0;i<a.size();++i){
        float d=std::fabs(a[i]-b[i]);
        if (d<=atol) continue;            // within absolute slack -> ok
        float den=std::fabs(b[i]); if(den<1e-3f) den=1e-3f;
        maxr=std::max(maxr,d/den);
    }
    return maxr;
}

static std::mt19937 rng(123);
static void fill(std::vector<float>&v){
    std::uniform_real_distribution<float> d(-1.f,1.f);
    for(auto&x:v) x=d(rng);
}

int main(){
    int fails=0;
    int BHs[]={1,8}; int Ss[]={1,16,128}; int Ds[]={16,64};
    for(int bh:BHs) for(int sq:Ss) for(int sk:Ss) for(int dd:Ds) for(int causal=0;causal<2;++causal){
        std::vector<float> q((size_t)bh*sq*dd), k((size_t)bh*sk*dd), v((size_t)bh*sk*dd);
        std::vector<float> o((size_t)bh*sq*dd), r((size_t)bh*sq*dd);
        fill(q); fill(k); fill(v);
        float scale=1.f/std::sqrt((float)dd);
        ms_attention_f32(q.data(),k.data(),v.data(),o.data(),bh,sq,sk,dd,scale,causal);
        ref_attention(q.data(),k.data(),v.data(),r.data(),bh,sq,sk,dd,scale,causal);
        float e=relerr(o,r);
        bool ok=e<1e-3f;
        if(!ok) ++fails;
        printf("[%s] BH=%d Sq=%-3d Sk=%-3d D=%-2d causal=%d relerr=%.2e\n",
               ok?"PASS":"FAIL",bh,sq,sk,dd,causal,e);
    }

    // Benchmark: BH=32 Sq=Sk=512 D=64 causal
    {
        int bh=32,s=512,dd=64; float scale=1.f/std::sqrt((float)dd);
        std::vector<float> q((size_t)bh*s*dd),k((size_t)bh*s*dd),v((size_t)bh*s*dd),o((size_t)bh*s*dd);
        fill(q);fill(k);fill(v);
        // FLOPs (causal ~ half): qk: 2*D per (i,j visible); av: 2*D per (i,j visible).
        double visible=(double)bh*((double)s*(s+1)/2.0);
        double flops=visible*(2.0*dd + 2.0*dd);
        auto bench=[&](int nt){
#ifdef _OPENMP
            omp_set_num_threads(nt);
#endif
            ms_attention_f32(q.data(),k.data(),v.data(),o.data(),bh,s,s,dd,scale,1); // warm
            int it=10; auto t0=std::chrono::high_resolution_clock::now();
            for(int i=0;i<it;++i) ms_attention_f32(q.data(),k.data(),v.data(),o.data(),bh,s,s,dd,scale,1);
            auto t1=std::chrono::high_resolution_clock::now();
            double sec=std::chrono::duration<double>(t1-t0).count()/it;
            return flops/sec/1e9;
        };
        int maxt=1;
#ifdef _OPENMP
        maxt=omp_get_max_threads();
#endif
        double st=bench(1);
        double mt=bench(maxt);
        printf("\nBenchmark attention BH=32 Sq=Sk=512 D=64 causal:\n");
        printf("  1T     : %.1f GFLOP/s\n", st);
        printf("  %2dT    : %.1f GFLOP/s\n", maxt, mt);
    }

    if(fails){ printf("\n%d FAILED\n",fails); return 1; }
    printf("\nALL PASS\n");
    return 0;
}
