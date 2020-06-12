function [ Jac, residual, weights ] = deriveResidualsAnalytic( IRef, DRef, I, xi, K, norm_param, use_hubernorm )

% get shorthands (R, t)
T = se3Exp(xi);
R = T(1:3, 1:3);
t = T(1:3,4);
KInv = K^-1;
RKInv = R * K^-1;


% ========= warp pixels into other image, save intermediate results ===============
% these contain the x,y image coordinates of the respective
% reference-pixel, transformed & projected into the new image.
xImg = zeros(size(IRef))-10;
yImg = zeros(size(IRef))-10;

% these contain the 3d position of the transformed point
xp = NaN(size(IRef));
yp = NaN(size(IRef));
zp = NaN(size(IRef));
wxp = NaN(size(IRef));
wyp = NaN(size(IRef));
wzp = NaN(size(IRef));
for x=1:size(IRef,2)
    for y=1:size(IRef,1)
        
        % 3D point in reference image. note that the pixel-coordinates of the
        % 2D point (1,1) are actually (0,0).
        p = DRef(y,x) * KInv * [x-1;y-1;1];
        
        % transform to image (unproject, rotate & translate)
        pTrans = R * p + t;
        
        % if point is valid (depth > 0), project and save result.
        if(pTrans(3) > 0 && DRef(y,x) > 0)
            % projected point (for interpolation of intensity and gradients)
            pTransProj = K * pTrans;
            xImg(y,x) = pTransProj(1) / pTransProj(3);
            yImg(y,x) = pTransProj(2) / pTransProj(3);
            
            % unwarped/warped 3d point, for calculation of Jacobian.
            xp(y,x) = p(1);
            yp(y,x) = p(2);
            zp(y,x) = p(3);
            
            wxp(y,x) = pTrans(1);
            wyp(y,x) = pTrans(2);
            wzp(y,x) = pTrans(3);
        end
    end
end


% ========= calculate actual derivative. ===============
% 1.: calculate image derivatives, and interpolate at warped positions.
dxI = NaN(size(I));
dyI = NaN(size(I));
dyI(2:(end-1),:) = 0.5*(I(3:(end),:) - I(1:(end-2),:));
dxI(:,2:(end-1)) = 0.5*(I(:,3:(end)) - I(:,1:(end-2)));
dxInterp = K(1,1) * reshape(interp2(dxI, xImg+1, yImg+1),size(I,1) * size(I,2),1);
dyInterp = K(2,2) * reshape(interp2(dyI, xImg+1, yImg+1),size(I,1) * size(I,2),1);

% 2.: get warped 3d points (x', y', z').
xp = reshape(xp,size(I,1) * size(I,2),1);
yp = reshape(yp,size(I,1) * size(I,2),1);
zp = reshape(zp,size(I,1) * size(I,2),1);
wxp = reshape(wxp,size(I,1) * size(I,2),1);
wyp = reshape(wyp,size(I,1) * size(I,2),1);
wzp = reshape(wzp,size(I,1) * size(I,2),1);


% 3. Jacobian calculation
Jac = zeros(size(I,1) * size(I,2),6);
residual = reshape(IRef - interp2(I, xImg+1, yImg+1),size(I,1) * size(I,2),1);


for i=1:size(I,1) * size(I,2)
     
    if isnan( xp(i) ) | wzp(i) == 0 | isnan(dxInterp(i)) | isnan(dyInterp(i))
        residual(i) = 0;
        continue
    end
    
    dT_dw1 = [ 0, 0, 0, 0; 0, 0, -1, 0; 0, 1, 0, 0; 0, 0, 0, 0 ];
    dT_dw2 = [ 0, 0, 1, 0; 0, 0, 0, 0; -1, 0, 0, 0; 0, 0, 0, 0 ];
    dT_dw3 = [ 0, -1, 0, 0; 1, 0, 0, 0; 0, 0, 0, 0; 0, 0, 0, 0 ];
    dT_dv1 = [ 0, 0, 0, 1; 0, 0, 0, 0; 0, 0, 0, 0; 0, 0, 0, 0 ];
    dT_dv2 = [ 0, 0, 0, 0; 0, 0, 0, 1; 0, 0, 0, 0; 0, 0, 0, 0 ];
    dT_dv3 = [ 0, 0, 0, 0; 0, 0, 0, 0; 0, 0, 0, 1; 0, 0, 0, 0 ];
    
    p_i = [ xp(i); yp(i); zp(i); 1 ];
    T_p_i = T * p_i;
    
    Tpw1 = dT_dw1 * T_p_i;
    Tpw2 = dT_dw2 * T_p_i;
    Tpw3 = dT_dw3 * T_p_i;
    Tpv1 = dT_dv1 * T_p_i;
    Tpv2 = dT_dv2 * T_p_i;
    Tpv3 = dT_dv3 * T_p_i;
         
%     pw1 = dT_dw1 * p_i;
%     pw2 = dT_dw2 * p_i;
%     pw3 = dT_dw3 * p_i;
%     pv1 = dT_dv1 * p_i;
%     pv2 = dT_dv2 * p_i;
%     pv3 = dT_dv3 * p_i;

    %pw1 = [ 0; -zp(i); yp(i); 0 ];
    %pw2 = [ zp(i); 0; -xp(i); 0 ];
    %pw3 = [ -yp(i); xp(i); 0; 0 ];
    %pv1 = [ 1; 0; 0; 0 ];
    %pv2 = [ 0; 1; 0; 0 ];
    %pv3 = [ 0; 0; 1; 0 ];    
    
    %Tpw1 = T * pw1;
    %Tpw2 = T * pw2;
    %Tpw3 = T * pw3;
    %Tpv1 = T * pv1;
    %Tpv2 = T * pv2;
    %Tpv3 = T * pv3;
    
    dpi = [ 1./wzp(i), 0, -wxp(i)/wzp(i)/wzp(i); 0, 1./wzp(i), -wyp(i)/wzp(i)/wzp(i) ];
    
    dI = [ dxInterp(i), dyInterp(i) ];
    
    dIdpi = dI * dpi;
    
    Jac( i, 1 ) = -dIdpi * Tpv1(1:3,:);
    Jac( i, 2 ) = -dIdpi * Tpv2(1:3,:);
    Jac( i, 3 ) = -dIdpi * Tpv3(1:3,:);
    Jac( i, 4 ) = -dIdpi * Tpw1(1:3,:);
    Jac( i, 5 ) = -dIdpi * Tpw2(1:3,:);
    Jac( i, 6 ) = -dIdpi * Tpw3(1:3,:);
    
    
end

weights = 0*residual + 2;
if use_hubernorm
    % Huber norm
    idx = find( abs(residual) > norm_param );
    weights( idx ) = norm_param / abs(residual( idx ));
else
    % Geman-McClure norm
    weights = 2. ./ ( 1. + residual.^2 / norm_param^2 ).^2;
end

% ========= plot residual image =========
%residual( isnan(residual), 1 ) = 0
figure(1);
imagesc(reshape(residual,size(I)));
colormap gray;
set(gca, 'CLim', [-1,1]);

figure(2);
imagesc(reshape(weights,size(I)));
colormap gray;
set(gca, 'CLim', [-1,1]);

end
