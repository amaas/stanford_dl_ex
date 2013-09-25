function [S,Y,YS,lbfgs_start,lbfgs_end,Hdiag,skipped] = lbfgsAdd(y,s,S,Y,YS,lbfgs_start,lbfgs_end,Hdiag,useMex)
ys = y'*s;
skipped = 0;
corrections = size(S,2);
if ys > 1e-10
	if lbfgs_end < corrections
		lbfgs_end = lbfgs_end+1;
		if lbfgs_start ~= 1
			if lbfgs_start == corrections
				lbfgs_start = 1;
			else
				lbfgs_start = lbfgs_start+1;
			end
		end
	else
		lbfgs_start = min(2,corrections);
		lbfgs_end = 1;
	end
	
	if useMex
		lbfgsAddC(y,s,Y,S,ys,int32(lbfgs_end));
	else
		S(:,lbfgs_end) = s;
		Y(:,lbfgs_end) = y;
	end
	YS(lbfgs_end) = ys;
	
	% Update scale of initial Hessian approximation
	Hdiag = ys/(y'*y);
else
	skipped = 1;
end