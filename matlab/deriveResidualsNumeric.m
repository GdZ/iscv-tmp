function [ Jac, residual, weights ] = deriveResidualsNumeric( IRef, DRef, I, xi, K, norm_param, use_hubernorm )
    % calculate numeric derivative. SLOW

    eps = 1e-6;
    Jac = zeros(size(I,1) * size(I,2),6);
    [residual, weights ] = calcResiduals(IRef,DRef,I,xi,K, norm_param, use_hubernorm);

    for j=1:6
        epsVec = zeros(6,1);
        epsVec(j) = eps;

        % multiply epsilon from left onto the current estimate.
        xiPerm =  se3Log(se3Exp(epsVec) * se3Exp(xi));
        Jac(:,j) = (calcResiduals(IRef,DRef,I,xiPerm,K, norm_param, use_hubernorm) - residual) / eps;
    end
end

