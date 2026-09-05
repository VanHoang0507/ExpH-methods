%% ========================================================================
% Brusselator 3D: extrapolated reference solution
%
% The fourth-order ExpRB42 method is run with N and 2N time steps.
% Richardson extrapolation is then applied:
%
%       Yref = Y_2N + (Y_2N-Y_N)/(2^4-1)
%            = (16*Y_2N-Y_N)/15.
% ========================================================================

clear;
clc;
close all;

%% ========================================================================
% 0. Paths
% ========================================================================
script_dir  = fileparts(mfilename('fullpath'));
project_dir = fileparts(script_dir);

addpath('../phipmsimuliom')
addpath('../KronPACK/src')

dataFolder = fullfile(script_dir,'Data');

if ~exist(dataFolder,'dir')
    mkdir(dataFolder);
end

%% ========================================================================
% 1. Problem parameters
% ========================================================================
d = 3;

n = 64*ones(1,d);
a = zeros(1,d);
b = ones(1,d);

T = 1;

deltau = 0.01;
deltav = 0.02;

alphau = 0.1;
alphav = 0.1;

a1u = 1;
a2u = 2;

tol_phipm_simul_iom = 1e-10;

% Richardson levels: N and 2N
nsteps = 280;
nsteps_coarse = nsteps;
nsteps_fine   = 2*nsteps;

fprintf('Grid: %d x %d x %d\n',n(1),n(2),n(3));
fprintf('Number of unknowns: %d\n',2*prod(n));
fprintf('Coarse time steps: %d\n',nsteps_coarse);
fprintf('Fine time steps:   %d\n',nsteps_fine);

%% ========================================================================
% 2. Spatial discretization
% ========================================================================
x = cell(1,d);
D1 = cell(1,d);
D2 = cell(1,d);

A_sp = cell(2,1);
A_sp{1} = cell(1,d);
A_sp{2} = cell(1,d);

for mu = 1:d
    x{mu} = linspace(a(mu),b(mu),n(mu));
    h = (b(mu)-a(mu))/(n(mu)-1);

    % Second-order centered approximation of the second derivative
    D2{mu} = spdiags( ...
        ones(n(mu),1)*([1,-2,1]/h^2), ...
        -1:1,n(mu),n(mu));

    % Homogeneous Neumann boundary conditions
    D2{mu}(1,1:2) = [-2,2]/h^2;
    D2{mu}(end,end-1:end) = [2,-2]/h^2;

    % Centered approximation of the first derivative
    D1{mu} = spdiags( ...
        ones(n(mu),1)*([-1,0,1]/(2*h)), ...
        -1:1,n(mu),n(mu));

    % Zero boundary rows for the advection operator
    D1{mu}(1,:) = 0;
    D1{mu}(end,:) = 0;

    A_sp{1}{mu} = -alphau*D1{mu} + deltau*D2{mu};
    A_sp{2}{mu} = -alphav*D1{mu} + deltav*D2{mu};
end

%% ========================================================================
% 3. Three-dimensional Kronecker-sum operators
% ========================================================================
K = cell(2,1);

K{1} = kronsum(A_sp{1});
K{2} = kronsum(A_sp{2});

pn = prod(n);
sz = n;

%% ========================================================================
% 4. Initial condition
% ========================================================================
X = cell(1,d);
[X{:}] = ndgrid(x{:});

U0 = cell(2,1);

U0{1} = 1 ...
    + cos(2*pi*X{1}) ...
    .*cos(2*pi*X{2}) ...
    .*cos(2*pi*X{3});

U0{2} = 3*ones(n);

Y0 = [U0{1}(:); U0{2}(:)];

%% ========================================================================
% 5. Reaction functions and their Jacobian entries
% ========================================================================
g1 = @(t,u,v) -(a1u+1)*u + a2u + (u.*u).*v;
g2 = @(t,u,v) a1u*u - (u.*u).*v;

dg11 = @(t,u,v) -(a1u+1) + 2*(u.*v);
dg12 = @(t,u,v) u.*u;
dg21 = @(t,u,v) a1u - 2*(u.*v);
dg22 = @(t,u,v) -u.*u;

%% ========================================================================
% 6. Vectorized right-hand side and full Jacobian
% ========================================================================
Kfun = @(Y) [ ...
    K{1}*Y(1:pn); ...
    K{2}*Y(pn+1:2*pn)];

Nfun = @(t,Y) [ ...
    g1(t,Y(1:pn),Y(pn+1:2*pn)); ...
    g2(t,Y(1:pn),Y(pn+1:2*pn))];

Ffun = @(t,Y) Kfun(Y) + Nfun(t,Y);

Jfun = @(t,Y) [ ...
    K{1} + spdiags( ...
        dg11(t,Y(1:pn),Y(pn+1:2*pn)),0,pn,pn), ...
    spdiags( ...
        dg12(t,Y(1:pn),Y(pn+1:2*pn)),0,pn,pn); ...
    spdiags( ...
        dg21(t,Y(1:pn),Y(pn+1:2*pn)),0,pn,pn), ...
    K{2} + spdiags( ...
        dg22(t,Y(1:pn),Y(pn+1:2*pn)),0,pn,pn)];

%% ========================================================================
% 7. Coarse ExpRB42 solution
% ========================================================================
fprintf('\nComputing ExpRB42 with %d time steps...\n',nsteps_coarse);

tic;
[~,Ycoarse] = ExpRB42_local( ...
    Ffun,Jfun,0,T,Y0,nsteps_coarse,tol_phipm_simul_iom);
coarse_time = toc;

fprintf('Coarse CPU time: %.2f seconds\n',coarse_time);

%% ========================================================================
% 8. Fine ExpRB42 solution
% ========================================================================
fprintf('\nComputing ExpRB42 with %d time steps...\n',nsteps_fine);

tic;
[~,Yfine] = ExpRB42_local( ...
    Ffun,Jfun,0,T,Y0,nsteps_fine,tol_phipm_simul_iom);
fine_time = toc;

fprintf('Fine CPU time: %.2f seconds\n',fine_time);

%% ========================================================================
% 9. Fourth-order Richardson extrapolation
% ========================================================================
p = 4;

Yfinal = Yfine + (Yfine-Ycoarse)/(2^p-1);

% Equivalent expression:
% Yfinal = (16*Yfine-Ycoarse)/15;

coarse_fine_difference = norm(Yfine-Ycoarse,inf);
extrapolation_correction = norm(Yfinal-Yfine,inf);
wctime = coarse_time + fine_time;

fprintf('\nCoarse/fine difference:   %.3e\n', ...
    coarse_fine_difference);
fprintf('Extrapolation correction: %.3e\n', ...
    extrapolation_correction);
fprintf('Total CPU time:            %.2f seconds\n',wctime);

%% ========================================================================
% 10. Reshape and save the reference solution
% ========================================================================
Uref = cell(2,1);

Uref{1} = reshape(Yfinal(1:pn),sz);
Uref{2} = reshape(Yfinal(pn+1:2*pn),sz);

referenceFile = fullfile(dataFolder,'brusselator_3D_Uref.mat');

save(referenceFile, ...
    'Uref', ...
    'Yfinal', ...
    'Ycoarse', ...
    'Yfine', ...
    'nsteps_coarse', ...
    'nsteps_fine', ...
    'coarse_fine_difference', ...
    'extrapolation_correction', ...
    'coarse_time', ...
    'fine_time', ...
    'wctime', ...
    'T', ...
    'n', ...
    'tol_phipm_simul_iom', ...
    '-v7.3');

fprintf('\nReference solution saved to:\n%s\n',referenceFile);

rmpath('../phipmsimuliom')
rmpath('../KronPACK/src')

%% ========================================================================
% Local ExpRB42 implementation
% ========================================================================
function [tgrid,yFinal] = ExpRB42_local( ...
    Ffun,Jfun,t0,tEnd,y0,nsteps,tol)

    h = (tEnd-t0)/nsteps;
    tgrid = linspace(t0,tEnd,nsteps+1);

    y = y0(:);
    zeroVec = zeros(size(y));

    for k = 1:nsteps
        tn = tgrid(k);

        Fn = Ffun(tn,y);
        Jn = Jfun(tn,y);

        % Internal stage
        alpha = (3/4)*h;

        incStage = alpha*phipm_simul_iom( ...
            1,alpha*Jn,[zeroVec,Fn],tol,1,2);

        yStage = y + incStage;

        % Nonlinear Rosenbrock defect
        FStage = Ffun(tn+alpha,yStage);
        Dn2 = FStage-Fn-Jn*(yStage-y);

        % Final update
        incFinal = h*phipm_simul_iom( ...
            1,h*Jn,[zeroVec,Fn,zeroVec,(32/9)*Dn2],tol,1,2);

        y = y + incFinal;
    end

    yFinal = y;
end
