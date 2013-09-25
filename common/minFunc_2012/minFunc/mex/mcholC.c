#include <math.h>
#include "mex.h"

double mymax(double x, double y)
{
    if (x > y)
        return x;
    else
        return y;
}

double absolute(double x)
{
    if (x >= -x)
        return x;
    else
        return -x;
}

void permuteInt(int *x, int p, int q)
{
    int temp;
    temp = x[p];
    x[p] = x[q];
    x[q] = temp;
}

void permute(double *x, int p, int q)
{
    double temp;
    temp = x[p];
    x[p] = x[q];
    x[q] = temp;
}

void permuteRows(double *x, int p, int q,int n)
{
    int i;
    double temp;
    for(i = 0; i < n; i++)
    {
        temp = x[p+i*n];
        x[p+i*n] = x[q+i*n];
        x[q+i*n] = temp;
    }
}

void permuteCols(double *x, int p, int q,int n)
{
    int i;
    double temp;
    for(i = 0; i < n; i++)
    {
        temp = x[i+p*n];
        x[i+p*n] = x[i+q*n];
        x[i+q*n] = temp;
    }
}

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    int n,sizL[2],sizD[2],i,j,q,s,
    *P;
    
    double mu,gamma,xi,delta,beta,maxVal,theta,
    *c,    *H, *L, *D, *A;
    
    /* Input */
    H = mxGetPr(prhs[0]);
    if (nrhs == 1)
    {
        mu = 1e-12;
    }
    else
    {
        mu = mxGetScalar(prhs[1]);
    }
    
    /* Compute Sizes */
    n = mxGetDimensions(prhs[0])[0];
    
    /* Form Output */
    sizL[0] = n;
    sizL[1] = n;
    plhs[0] = mxCreateNumericArray(2,sizL,mxDOUBLE_CLASS,mxREAL);
    L = mxGetPr(plhs[0]);
    sizD[0] = n;
    sizD[1] = 1;
    plhs[1] = mxCreateNumericArray(2,sizD,mxDOUBLE_CLASS,mxREAL);
    D = mxGetPr(plhs[1]);
    plhs[2] = mxCreateNumericArray(2,sizD,mxINT32_CLASS,mxREAL);
    P = (int*)mxGetData(plhs[2]);
    
    /* Initialize */
    c = mxCalloc(n*n,sizeof(double));
    A = mxCalloc(n*n,sizeof(double));
    
    for (i = 0; i < n; i++)
    {
        P[i] = i;
        for (j = 0;j < n; j++)
        {
            A[i+n*j] = H[i+n*j];
        }
    }
    
    gamma = 0;
    for (i = 0; i < n; i++)
    {
        L[i+n*i] = 1;
        c[i+n*i] = A[i+n*i];
    }
    
    /* Compute modification parameters */
    gamma = -1;
    xi = -1;
    for (i = 0; i < n; i++)
    {
        gamma = mymax(gamma,absolute(A[i+n*i]));
        for (j = 0;j < n; j++)
        {
            /*printf("A(%d,%d) = %f, %f\n",i,j,A[i+n*j],absolute(A[i+n*j]));*/
            if (i != j)
                xi = mymax(xi,absolute(A[i+n*j]));
        }
    }
    delta = mu*mymax(gamma+xi,1);
    
    if (n > 1)
    {
        beta = sqrt(mymax(gamma,mymax(mu,xi/sqrt(n*n-1))));
    }
    else
    {
        beta = sqrt(mymax(gamma,mu));
    }
    
    for (j = 0; j < n; j++)
    {
        
    /* Find q that results in Best Permutation with j */
        maxVal = -1;
        q = 0;
        for(i = j; i < n; i++)
        {
            if (absolute(c[i+n*i]) > maxVal)
            {
                maxVal = mymax(maxVal,absolute(c[i+n*i]));
                q = i;
            }
        }
        
        /* Permute D,c,L,A,P */
        permute(D,j,q);
        permuteInt(P,j,q);
        permuteRows(c,j,q,n);
        permuteCols(c,j,q,n);
        permuteRows(L,j,q,n);
        permuteCols(L,j,q,n);
        permuteRows(A,j,q,n);
        permuteCols(A,j,q,n);
        
        for(s = 0; s <= j-1; s++)
            L[j+n*s] = c[j+n*s]/D[s];
        
        for(i = j+1; i < n; i++)
        {
            c[i+j*n] = A[i+j*n];
            for(s = 0; s <= j-1; s++)
            {
                c[i+j*n] -= L[j+n*s]*c[i+n*s];
            }
        }
        
        theta = 0;
        if (j < n-1)
        {
            for(i = j+1;i < n; i++)
                theta = mymax(theta,absolute(c[i+n*j]));
        }
        
        D[j] = mymax(absolute(c[j+n*j]),mymax(delta,theta*theta/(beta*beta)));
        
        if (j < n-1)
        {
            for(i = j+1; i < n; i++)
            {
                c[i+n*i] = c[i+n*i] - c[i+n*j]*c[i+n*j]/D[j];
            }
        }
        
    }
    
    for(i = 0; i < n; i++)
        P[i]++;
    
    mxFree(c);
    mxFree(A);
}