% Inverse transform of trilinear FEM elements
% Only applicable to orthogonal grids
% Created by Rohit Deshmukh

function [N] = FEM_shape_calculator_3D_ortho_gfemlr(x,y,z,xpt,ypt,zpt)

sumxpt = sum(xpt)/8;
sumypt = sum(ypt)/8;
sumzpt = sum(zpt)/8;

dxpt = (xpt(1)+xpt(2)+xpt(3)+xpt(4)-xpt(5)-xpt(6)-xpt(7)-xpt(8))/4;
dypt = (-ypt(1)+ypt(2)+ypt(3)-ypt(4)-ypt(5)+ypt(6)+ypt(7)-ypt(8))/4;
dzpt = (zpt(1)+zpt(2)-zpt(3)-zpt(4)+zpt(5)+zpt(6)-zpt(7)-zpt(8))/4;

zeta_i = [ 1 1  1  1 -1 -1 -1 -1];
eta_i  = [-1 1  1 -1 -1  1  1 -1];
phi_i  = [ 1 1 -1 -1  1  1 -1 -1];

% Inverse transform for parallelogram elements, bilinear shape functions
zeta = 2*(x-sumxpt)/(dxpt);
eta  = 2*(y-sumypt)/(dypt);
phi  = 2*(z-sumzpt)/(dzpt);


N = zeros(8,1);
% FEM shape function values
for i = 1:8
    N(i) = (1/8)*(1+zeta_i(i)*zeta)*(1+eta_i(i)*eta)*(1+phi_i(i)*phi);
end
