 #define MAXTOP 11
#define MAXBOT 26
#define PI 3.14159265358979323846
#define M 1

double stirling(double x){
    if(x ==0){
        return 0.0;
    }
    else{
        return x * log(x) - x + .5*log(2*PI*x);
    }
}

double newcomb(double N, double L) {
    int a = N;
    int b = L;
    double n = N;
    double l = L;

    if (a == 0) {
        return 0.0;
        }
    if (b == 0 || a == b){
        return (stirling(ceil(n/2)) + stirling(floor(n/2)) - stirling(n));
        }
    else{
        return (stirling(ceil(n/2)) + stirling(floor(n/2)) - stirling(l)- stirling(n-l));
        }
    }

double calcP(double NA, double NB, double NC, double ND, double la,double lA,double lb, double lB, double lgamma, double lc, double ldelta, double ld,double halpha, double hA, double hbeta, double hB, double hgamma,double hdelta, double hc, double hd, double kcoop,double kcomp, double kdu, double kud, double kx){
    double e = (halpha * (la - ((MAXTOP-NA)/2)) + hbeta * (lb - ((MAXTOP-NB)/2)) + hgamma*(lgamma-((MAXBOT-NC))/2) + hdelta*(ldelta-(MAXBOT-ND)/2) + hA * (lA-NA/2) + hB*(lB-NB/2) + hc*(lc-NC/2)+hd*(ld-ND/2) + kcoop*((la-lA)*NA - (MAXTOP*NA/2) + (lb-lB)*NB- (MAXTOP*NB/2)) + kcomp*((lA-la)*NB -(MAXTOP*NB/2) + (lB-lb)*NA - (MAXTOP*NA/2)) + kdu*((la-lA)*NC - (NC*MAXTOP/2) +(lb-lB)*ND -(ND*MAXTOP/2)) + kud*(NA*(lc-lgamma)-(NA*MAXBOT/2)+NB*(ld-ldelta)-(NB*MAXBOT/2))+kx*((lB-lb)*NC-(NC*MAXTOP/2)+(lA-la)*ND-(ND*MAXTOP/2)));
    double sum = (newcomb(MAXTOP-1-NA,la) + newcomb(MAXTOP-1-NB,lb)+ newcomb(NA, lA) + newcomb(NB, lB) + newcomb(MAXBOT-1-NC,lgamma) + newcomb(NC,lc) + newcomb(MAXBOT-1-ND,ldelta) + newcomb(ND,ld) + e);
    return sum;
}

__kernel void calc_next_state(__global double* Result, __global double* Params, __global double* current_state)
{

     int i = get_global_id(0);   //ARRAYindexes:[nai,nbj,nck,ndl, nam, nbn, nco, ndp] [i,j,k,l ,m,n,o,p]
     const double EXP_UPPER_LIMIT = 650;  // Approx ln(DBL_MAX)
     const double EXP_LOWER_LIMIT = -650; // Approx ln(DBL_MIN)

     int nam = i/(MAXTOP*MAXBOT*MAXBOT)%MAXTOP;
     int nbn = i/(MAXBOT*MAXBOT)%MAXTOP;
     int nco = i/MAXBOT%MAXBOT;
     int ndp = i%MAXBOT;

     int nai = current_state[0];
     int nbj = current_state[1];
     int nck = current_state[2];
     int ndl = current_state[3];

     double halpha = Params[0];
     double ha = Params[1];
     double hbeta = Params[2];
     double hb = Params[3];
     double hgamma = Params[4];
     double hc = Params[5];
     double hdelta = Params[6];
     double hd = Params[7];
     double kcoop = Params[8];
     double kcomp = Params[9];
     double kdu = Params[10];
     double kup = Params[11];
     double kx = Params[12];


     double pvalue=0;   //a=lA, b=lB, c=lC,d=lD, e=la,f=lb,g=lc,h=ld
     double loop4 = 0;
     for(int a=0; a<=nai;a++)
     {
        for(int b=0; b<=nbj;b++)
        {
            for (int c=0; c<=M;c++)
            {
                for(int d=0; d<=M;d++)
                {
                    int e = nam - nai + a;  //lalpha
                    int f = nbn - nbj + b;  //lbeta
                    int g = nco - nck + c;  //lgamma
                    int h = ndp - ndl + d; // ldelta
                    if(0 <= e && e <= (MAXTOP-1-nai) && 0 <= f && f <= (MAXTOP-1-nbj) && 0 <= g && g <= (M) && 0 <= h && h <= (M) )
                    //if(0 <= e && e <= (M) && 0 <= f && f <= (M) && 0 <= g && g <= (M) && 0 <= h && h <= (M) )

                    {

                        pvalue = calcP(nai,nbj,nck,ndl, e,a,f,b,g,c,h,d, halpha,ha,hbeta,hb,hgamma,hdelta,hc,hd,kcoop,kcomp,kdu,kup,kx);
                        pvalue = fmax(fmin(pvalue, EXP_UPPER_LIMIT), EXP_LOWER_LIMIT);
                        loop4 +=  exp(pvalue);

                    }
                }

            }


        }

    }
    Result[i] = loop4;
}


