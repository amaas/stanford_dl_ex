#include <math.h>
#include "mex.h"

/* See lbfgsAdd.m for details */
/* This function will not exit gracefully on bad input! */

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
	/* Variable Declarations */
	
	double *s,*y,*S, *Y, ys;
	int i,j,nVars,lbfgs_end;
	
	/* Get Input Pointers */
	
	y = mxGetPr(prhs[0]);
	s = mxGetPr(prhs[1]);
	Y = mxGetPr(prhs[2]);
	S = mxGetPr(prhs[3]);
	ys= mxGetScalar(prhs[4]);
	lbfgs_end = (int)mxGetScalar(prhs[5]);
	
	if (!mxIsClass(prhs[5],"int32"))
		mexErrMsgTxt("lbfgs_end must be int32");
	
	/* Compute number of variables, maximum number of corrections */
	
	nVars = mxGetDimensions(prhs[2])[0];
	
	for(j=0;j<nVars;j++) {
		S[j+nVars*(lbfgs_end-1)] = s[j];
		Y[j+nVars*(lbfgs_end-1)] = y[j];
	}
}
