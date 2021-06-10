#include <stdio.h>
#include <stdlib.h>
#include <algorithm>
#include <time.h>
#include <map>
#include <vector>
using namespace std;

// Rabin-Karp rolling hash

long long Power(long long a,long long b,long long p)
{
    long long ans=1;
    for(;b;a=a*a%p,b>>=1)if(b&1)ans=ans*a%p;
    return ans;
}

namespace Primitive_root
{
    int pri[100],tot;
    bool check_prime(int g)
    {
        for(int i=2;i*i<=g;++i)if(g%i==0)return false;
        return true;
    }
    int calc(int p, int low)
    {
        // find a primitive root (>low) of given prime p
        srand(time(0));
        int i,j=p-1,g;tot=0;
        for(i=2;i*i<=j;++i)
            if(j%i==0)
                for(pri[++tot]=i;j%i==0;j/=i);
        if(j!=1)pri[++tot]=j;
        for(;;)
        {
            g = rand()%(p-1)+1;
            if(check_prime(g)&&(g>low))
            {
                for(i=1;i<=tot;++i)
                    if(Power(g,(p-1)/pri[i],p)==1)
                        break;
                if(i>tot) return g;
            }
        }
    }
}

# define W 42 // width
# define H 42 // height
# define C 4  // channels
#define P1 1000000007 // prime modulus of rolling hash
#define P2 1000000009 // prime modulus of rolling hash
const int b1 = Primitive_root::calc(P1, 100000); // base of rolling hash
const int b2 = Primitive_root::calc(P2, 100000); // base of rolling hash

int m = 5;
int state_tot = 0;

extern "C"
void init(int init_m)
{
    m = init_m;
}

long long calc_hash(unsigned char obs[W][H][C], int p, int b)
{
    long long h = 0;
    for(int i=0;i<W;++i)
        for(int j=0;j<H;++j)
            for(int k=0;k<C;++k)
                h = (h*b+obs[i][j][k])%p;
    return h;
}

extern "C"
long long get_hash(unsigned char obs[W][H][C])
{
    long long h1 = calc_hash(obs, P1, b1);
    long long h2 = calc_hash(obs, P2, b2);
    long long h = h1*(P2+5)+h2;
    return h;
}

# define L 233
map<long long, int> Hash[L];
map<long long, float> Hash_avg[L];
map<pair<long long, int>, float> Hash_val[L];

extern "C"
int get_state_tot()
{
    return state_tot;
}

extern "C"
int get_return(long long h)
{
    if(Hash_avg[h%L].find(h)==Hash_avg[h%L].end()) return 0;
    return Hash_avg[h%L][h];
}

extern "C"
void add_return(long long h, float value)
{
    if(Hash[h%L].find(h)==Hash[h%L].end())
    {
        ++state_tot;
        Hash[h%L][h] = 0;
    }
    int now = ++Hash[h%L][h];
    if(now <= m) Hash_avg[h%L][h] = (Hash_avg[h%L][h]*(now-1)+value)/now;
    else Hash_avg[h%L][h] += (value - Hash_val[h%L][make_pair(h,now%m)]) / m;
    Hash_val[h%L][make_pair(h,now%m)] = value;
}
