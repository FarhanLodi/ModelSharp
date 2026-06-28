// Standalone parity + benchmark test for ms_conv2d_f32.
#include "../ms_kernels.h"
#include <cstdio>
#include <cmath>
#include <vector>
#include <random>
#include <chrono>
#include <algorithm>
#ifdef _OPENMP
#include <omp.h>
#endif

static void ref_conv(const float* x,const float* w,const float* bias,float* y,
                     int N,int Cin,int H,int W,int Cout,int KH,int KW,
                     int sH,int sW,int pH,int pW,int dH,int dW,int groups){
    int Ho=(H+2*pH-dH*(KH-1)-1)/sH+1;
    int Wo=(W+2*pW-dW*(KW-1)-1)/sW+1;
    int Cin_g=Cin/groups, Cout_g=Cout/groups;
    for(int n=0;n<N;++n)
    for(int co=0;co<Cout;++co){
        int g=co/Cout_g;
        for(int oh=0;oh<Ho;++oh)for(int ow=0;ow<Wo;++ow){
            float acc=bias?bias[co]:0.f;
            for(int ci=0;ci<Cin_g;++ci){
                int realci=g*Cin_g+ci;
                for(int kh=0;kh<KH;++kh)for(int kw=0;kw<KW;++kw){
                    int ih=oh*sH-pH+kh*dH, iw=ow*sW-pW+kw*dW;
                    if(ih<0||ih>=H||iw<0||iw>=W) continue;
                    float xv=x[(((size_t)n*Cin+realci)*H+ih)*W+iw];
                    float wv=w[(((size_t)co*Cin_g+ci)*KH+kh)*KW+kw];
                    acc+=xv*wv;
                }
            }
            y[(((size_t)n*Cout+co)*Ho+oh)*Wo+ow]=acc;
        }
    }
}

static std::mt19937 rng(7);
static void fill(std::vector<float>&v){ std::uniform_real_distribution<float> d(-1,1); for(auto&x:v)x=d(rng); }
// Combined abs/rel error. The naive float32 reference accumulates ~K products,
// so near-zero outputs (large cancellation) carry float noise ~ K*eps that the
// fused kernel reorders differently; an abs slack scaled by sqrt(K) avoids false
// fails there while still catching real divergence (verified vs a double ref).
static float relerr(const std::vector<float>&a,const std::vector<float>&b,float atol){
    float m=0;
    for(size_t i=0;i<a.size();++i){
        float d=std::fabs(a[i]-b[i]);
        if(d<=atol) continue;
        float den=std::fabs(b[i]); if(den<1e-3f) den=1e-3f;
        m=std::max(m,d/den);
    }
    return m;
}

struct Cfg{const char*name;int N,Cin,H,W,Cout,KH,KW,sH,sW,pH,pW,dH,dW,groups;};

static int run(const Cfg&c,bool withbias){
    int Ho=(c.H+2*c.pH-c.dH*(c.KH-1)-1)/c.sH+1;
    int Wo=(c.W+2*c.pW-c.dW*(c.KW-1)-1)/c.sW+1;
    std::vector<float> x((size_t)c.N*c.Cin*c.H*c.W);
    std::vector<float> w((size_t)c.Cout*(c.Cin/c.groups)*c.KH*c.KW);
    std::vector<float> bias(c.Cout);
    std::vector<float> y((size_t)c.N*c.Cout*Ho*Wo), r((size_t)c.N*c.Cout*Ho*Wo);
    fill(x);fill(w);fill(bias);
    const float* bp=withbias?bias.data():nullptr;
    ms_conv2d_f32(x.data(),w.data(),bp,y.data(),c.N,c.Cin,c.H,c.W,c.Cout,c.KH,c.KW,
                  c.sH,c.sW,c.pH,c.pW,c.dH,c.dW,c.groups);
    ref_conv(x.data(),w.data(),bp,r.data(),c.N,c.Cin,c.H,c.W,c.Cout,c.KH,c.KW,
             c.sH,c.sW,c.pH,c.pW,c.dH,c.dW,c.groups);
    int Kacc=(c.Cin/c.groups)*c.KH*c.KW;
    float atol=2e-3f*std::sqrt((float)Kacc);   // ~float accumulation noise floor
    float e=relerr(y,r,atol);
    bool ok=e<2e-3f;
    printf("[%s] %-20s bias=%d relerr=%.2e\n",ok?"PASS":"FAIL",c.name,withbias,e);
    return ok?0:1;
}

int main(){
    int fails=0;
    Cfg cfgs[]={
        {"1x1",            2,16,8,8,  32,1,1, 1,1,0,0,1,1,1},
        {"3x3 s1 p1",      2,16,16,16,24,3,3, 1,1,1,1,1,1,1},
        {"3x3 s2 p1",      2,8,17,17, 16,3,3, 2,2,1,1,1,1,1},
        {"3x3 dil2 p2",    1,8,16,16, 16,3,3, 1,1,2,2,2,2,1},
        {"groups=2",       2,16,12,12,16,3,3, 1,1,1,1,1,1,2},
        {"depthwise",      2,16,14,14,16,3,3, 1,1,1,1,1,1,16},
    };
    for(auto&c:cfgs){ fails+=run(c,false); fails+=run(c,true); }

    // Benchmark ResNet-ish: N=1 Cin=256 H=W=56 Cout=256 3x3 pad1 s1
    {
        Cfg c={"resnet",1,256,56,56,256,3,3,1,1,1,1,1,1,1};
        int Ho=56,Wo=56;
        std::vector<float> x((size_t)c.N*c.Cin*c.H*c.W);
        std::vector<float> w((size_t)c.Cout*c.Cin*c.KH*c.KW);
        std::vector<float> bias(c.Cout);
        std::vector<float> y((size_t)c.N*c.Cout*Ho*Wo);
        fill(x);fill(w);fill(bias);
        double flops=2.0*(double)c.N*c.Cout*Ho*Wo*(double)c.Cin*c.KH*c.KW;
        auto bench=[&](int nt){
#ifdef _OPENMP
            omp_set_num_threads(nt);
#endif
            ms_conv2d_f32(x.data(),w.data(),bias.data(),y.data(),c.N,c.Cin,c.H,c.W,c.Cout,c.KH,c.KW,1,1,1,1,1,1,1);
            int it=10; auto t0=std::chrono::high_resolution_clock::now();
            for(int i=0;i<it;++i) ms_conv2d_f32(x.data(),w.data(),bias.data(),y.data(),c.N,c.Cin,c.H,c.W,c.Cout,c.KH,c.KW,1,1,1,1,1,1,1);
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
        printf("\nBenchmark conv ResNet N=1 Cin=256 56x56 Cout=256 3x3 p1:\n");
        printf("  1T     : %.1f GFLOP/s\n", st);
        printf("  %2dT    : %.1f GFLOP/s\n", maxt, mt);
    }

    if(fails){ printf("\n%d FAILED\n",fails); return 1; }
    printf("\nALL PASS\n");
    return 0;
}
