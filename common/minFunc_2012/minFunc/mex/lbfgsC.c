#include <math.h>
#include "mex.h"

/* See lbfgs.m for details! */
/* This function may not exit gracefully on bad input! */


void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    /* Variable Declarations */
    
    double *s, *y, *g, *H, *d, *ro, *alpha, *beta, *q, *r;
    int nVars,nSteps,lhs_dims[2];
    double temp;
    int i,j;
    
    /* Get Input Pointers */
	
    g = mxGetPr(prhs[0]);
    s = mxGetPr(prhs[1]);
    y = mxGetPr(prhs[2]);
    H = mxGetPr(prhs[3]);
    
    /* Compute number of variables (p), rank of update (d) */
    
    nVars = mxGetDimensions(prhs[1])[0];
    nSteps = mxGetDimensions(prhs[1])[1];
    
	/* Allocated Memory for Function Variables */
    ro = mxCalloc(nSteps,sizeof(double));
	alpha = mxCalloc(nSteps,sizeof(double));
	beta = mxCalloc(nSteps,sizeof(double));
	q = mxCalloc(nVars*(nSteps+1),sizeof(double));
	r = mxCalloc(nVars*(nSteps+1),sizeof(double));
	
    /* Set-up Output Vector */
    
    lhs_dims[0] = nVars;
    lhs_dims[1] = 1;
    
    plhs[0] = mxCreateNumericArray(2,lhs_dims,mxDOUBLE_CLASS,mxREAL);
    d = mxGetPr(plhs[0]);
    
    /* ro = 1/(y(:,i)'*s(:,i)) */
    for(i=0;i<nSteps;i++)
    {
        temp = 0;
        for(j=0;j<nVars;j++)
        {
			temp += y[j+nVars*i]*s[j+nVars*i];
        }
        ro[i] = 1/temp;
    }
	
	/* q(:,k+1) = g */
	for(i=0;i<nVars;i++)
	{
		q[i+nVars*nSteps] = g[i];
	}

	for(i=nSteps-1;i>=0;i--)
	{
		/* alpha(i) = ro(i)*s(:,i)'*q(:,i+1) */
		alpha[i] = 0;
		for(j=0;j<nVars;j++)
		{
			alpha[i] += s[j+nVars*i]*q[j+nVars*(i+1)]; 
		}
		alpha[i] *= ro[i];

		/* q(:,i) = q(:,i+1)-alpha(i)*y(:,i) */
		for(j=0;j<nVars;j++)
		{
			q[j+nVars*i]=q[j+nVars*(i+1)]-alpha[i]*y[j+nVars*i];
		}
	}

	/*  r(:,1) = q(:,1) */
	for(i=0;i<nVars;i++)
	{
		r[i] = H[0]*q[i];
	}

	for(i=0;i<nSteps;i++)
	{
		/* beta(i) = ro(i)*y(:,i)'*r(:,i) */
		beta[i] = 0;
		for(j=0;j<nVars;j++)
		{
			beta[i] += y[j+nVars*i]*r[j+nVars*i];
		}
		beta[i] *= ro[i];

		/* r(:,i+1) = r(:,i) + s(:,i)*(alpha(i)-beta(i)) */
		for(j=0;j<nVars;j++)
		{
			r[j+nVars*(i+1)]=r[j+nVars*i]+s[j+nVars*i]*(alpha[i]-beta[i]);
		}
	}

	/* d = r(:,k+1) */
	for(i=0;i<nVars;i++)
	{
		d[i]=r[i+nVars*nSteps];
	}

	/* Free Memory */
	
	mxFree(ro);
	mxFree(alpha);
	mxFree(beta);
	mxFree(q);
	mxFree(r);
	
}
