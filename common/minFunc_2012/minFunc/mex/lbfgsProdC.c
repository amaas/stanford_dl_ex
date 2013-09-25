#include <math.h>
#include "mex.h"

/* See lbfgsProd.m for details */
/* This function will not exit gracefully on bad input! */

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
	/* Variable Declarations */
	
	double *S, *Y, *YS, *g, Hdiag, *d, *alpha, *beta;
	int i,j,nVars,nCor,maxCor,lbfgs_start,lbfgs_end;
	
	/* Get Input Pointers */
	
	g = mxGetPr(prhs[0]);
	S = mxGetPr(prhs[1]);
	Y = mxGetPr(prhs[2]);
	YS= mxGetPr(prhs[3]);
	lbfgs_start = (int)mxGetScalar(prhs[4]);
	lbfgs_end = (int)mxGetScalar(prhs[5]);
	Hdiag = mxGetScalar(prhs[6]);
	
	if (!mxIsClass(prhs[4],"int32")||!mxIsClass(prhs[5],"int32"))
		mexErrMsgTxt("lbfgs_start and lbfgs_end must be int32");
	
	/* Compute number of variables, maximum number of corrections */
	
	nVars = mxGetDimensions(prhs[1])[0];
	maxCor = mxGetDimensions(prhs[1])[1];
	
	/* Compute number of corrections available */
	if (lbfgs_start == 1)
		nCor = lbfgs_end-lbfgs_start+1;
	else
		nCor = maxCor;
	
	/* Allocate Memory for Local Variables */
	alpha = mxCalloc(nCor,sizeof(double));
	beta = mxCalloc(nCor,sizeof(double));
	
	/* Set-up Output Vector */
	plhs[0] = mxCreateDoubleMatrix(nVars,1,mxREAL);
	d = mxGetPr(plhs[0]);
	
	for(j=0;j<nVars;j++)
		d[j] = -g[j];
	
	for(i = lbfgs_end-1;i >= 0;i--) {
		alpha[i] = 0;
		for(j=0;j<nVars;j++)
			alpha[i] += S[j + nVars*i]*d[j];
		alpha[i] /= YS[i];
		for(j=0;j<nVars;j++)
			d[j] -= alpha[i]*Y[j + nVars*i];
	}
	if(lbfgs_start != 1) {
		for(i = maxCor-1;i >= lbfgs_start-1;i--) {
			alpha[i] = 0;
			for(j=0;j<nVars;j++)
				alpha[i] += S[j + nVars*i]*d[j];
			alpha[i] /= YS[i];
			for(j=0;j<nVars;j++)
				d[j] -= alpha[i]*Y[j + nVars*i];
		}
	}
	
	for(j=0;j<nVars;j++)
		d[j] *= Hdiag;
	
	if(lbfgs_start != 1) {
		for(i = lbfgs_start-1; i < maxCor; i++) {
			beta[i] = 0;
			for(j=0;j<nVars;j++)
				beta[i] += Y[j + nVars*i]*d[j];
			beta[i] /= YS[i];
			for(j=0;j<nVars;j++)
				d[j] += S[j+nVars*i]*(alpha[i]-beta[i]);
		}
	}
	for(i = 0; i < lbfgs_end; i++) {
		beta[i] = 0;
		for(j=0;j<nVars;j++)
			beta[i] += Y[j + nVars*i]*d[j];
		beta[i] /= YS[i];
		for(j=0;j<nVars;j++)
			d[j] += S[j+nVars*i]*(alpha[i]-beta[i]);
	}
	
	mxFree(alpha);
	mxFree(beta);
	
}
