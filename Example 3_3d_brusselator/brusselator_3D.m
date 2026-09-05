clear all
close all

addpath('phipmsimuliom')
addpath('KronPACK/src')

d = 3;

n = 64*ones(1,d);
a = 0*ones(1,d);
b = 1*ones(1,d);
T = 1;

deltau = 0.01;
deltav = 0.02;
alphau = 0.1;
alphav = 0.1;
a1u = 1;
a2u = 2;

method = 'exprb42'; % exprb42 exph42
compute_err = true; % if true, measure error against precomputed reference solution
                    % else plot u component at time T

tol_phipm_simul_iom = 1e-10;
nsteps = 15; % needed if not using matlab ODE suite
fprintf('Number of timesteps: %.3e\n',nsteps)

tau = T/nsteps;
N_exph42 = [10, 16, 28, 48, 80, 140];
N_exprb42 = [8, 15, 27, 48, 80, 140];

for mu = 1:d
  x{mu} = linspace(a(mu),b(mu),n(mu));
  h(mu) = (b(mu)-a(mu))/(n(mu)-1);
  D2{mu} = spdiags(ones(n(mu),1)*([1,-2,1]/(h(mu)^2)),-1:1,n(mu),n(mu));
  D2{mu}(1,1:2) = [-2,2]/(h(mu)^2);
  D2{mu}(n(mu),(n(mu)-1):n(mu)) = [2,-2]/(h(mu)^2);
  D1{mu} = spdiags(ones(n(mu),1)*([-1,0,1]/(2*h(mu))),-1:1,n(mu),n(mu));
  D1{mu}(1,1:2) = [0,0];
  D1{mu}(n(mu),(n(mu)-1):n(mu)) = [0,0];
  A_sp{1}{mu} = -alphau*D1{mu}+deltau*D2{mu};
  A_sp{2}{mu} = -alphav*D1{mu}+deltav*D2{mu};
  A{1}{mu} = full(A_sp{1}{mu});
  A{2}{mu} = full(A_sp{2}{mu});
end
[X{1:d}] = ndgrid(x{1:d});

g{1} = @(t,u,v) -(a1u+1)*u+a2u+(u.*u).*v;
g{2} = @(t,u,v) a1u*u-(u.*u).*v;

dgdu{1}{1} = @(t,u,v) -(a1u+1)+2*(u.*v); %dg1du
dgdu{1}{2} = @(t,u,v) u.*u; %dg1dv
dgdu{2}{1} = @(t,u,v) a1u-2*(u.*v); %dg2du
dgdu{2}{2} = @(t,u,v) -u.*u; %dg2dv

F{1} = @(t,u,v) kronsumv(u,A{1}) + g{1}(t,u,v);
F{2} = @(t,u,v) kronsumv(v,A{2}) + g{2}(t,u,v);

U0{1} = 1 + sin(2*pi*X{1}).*sin(2*pi*X{2}).*sin(2*pi*X{3});
U0{2} = 3*ones(n);
% ================================================================
% Vectorized form for ExpRB42
%
% Y = [u(:); v(:)]
% ================================================================

pn = prod(n);      % number of spatial grid points
sz = size(U0{1});  % original 3D shape

% Sparse Kronecker-sum matrices
K{1} = kronsum(A_sp{1});
K{2} = kronsum(A_sp{2});

% Initial vector
Y0 = [U0{1}(:); U0{2}(:)];

% Linear part
Kfun = @(Y) [K{1}*Y(1:pn); K{2}*Y(pn+1:2*pn)];

% Nonlinear part N(Y)
Nfun = @(t,Y) [ g{1}(t,Y(1:pn),Y(pn+1:2*pn)); ...
                g{2}(t,Y(1:pn),Y(pn+1:2*pn)) ];

% Full RHS F(Y) = K*Y + N(Y)
Ffun = @(t,Y) Kfun(Y) + Nfun(t,Y);

% Full Rosenbrock Jacobian J(Y) = K + N'(Y)
Jfun = @(t,Y) [ ...
    K{1} + spdiags(dgdu{1}{1}(t,Y(1:pn),Y(pn+1:2*pn)),0,pn,pn), ...
           spdiags(dgdu{1}{2}(t,Y(1:pn),Y(pn+1:2*pn)),0,pn,pn); ...
           spdiags(dgdu{2}{1}(t,Y(1:pn),Y(pn+1:2*pn)),0,pn,pn), ...
    K{2} + spdiags(dgdu{2}{2}(t,Y(1:pn),Y(pn+1:2*pn)),0,pn,pn) ];

JN = @(t,Y) [spdiags(dgdu{1}{1}(t,Y(1:pn),Y(pn+1:2*pn)),0,pn,pn), ...
             spdiags(dgdu{1}{2}(t,Y(1:pn),Y(pn+1:2*pn)),0,pn,pn); ...
             spdiags(dgdu{2}{1}(t,Y(1:pn),Y(pn+1:2*pn)),0,pn,pn), ...
             spdiags(dgdu{2}{2}(t,Y(1:pn),Y(pn+1:2*pn)),0,pn,pn) ];


fprintf('Method: %s\n',method)
switch method
  case 'exph42'
    tic
    [~, Yfinal] = ExpH3(Nfun, Kfun, JN, 0, T, Y0, nsteps, tol_phipm_simul_iom, 0.5);
    wctime = toc;
    U{1} = reshape(Yfinal(1:pn),sz);
    U{2} = reshape(Yfinal(pn+1:2*pn),sz);
  case 'exprb42'
    tic
    [~, Yfinal] = exprb42_phipm_simul_iom(Ffun,Jfun,0,T,Y0,nsteps,tol_phipm_simul_iom);
    wctime = toc;
    U{1} = reshape(Yfinal(1:pn),sz);
    U{2} = reshape(Yfinal(pn+1:2*pn),sz);
  otherwise
    error('Method not known.')
end

% save(fullfile('brusselator_3D_Uref1.mat'), ...
%      'Uref','T','n','-v7.3');
if compute_err
  load('brusselator_3D_Uref1.mat')

  % Componentwise relative maximum-norm errors
  normrefu = norm(Uref{1}(:), inf);
  normrefv = norm(Uref{2}(:), inf);

  abs_erru = norm(U{1}(:) - Uref{1}(:), inf)/normrefu;
  abs_errv = norm(U{2}(:) - Uref{2}(:), inf)/normrefv;

  erru = abs_erru ;
  errv = abs_errv ;

  % Overall error: worst relative component error
  err = max(erru, errv);

  fprintf('Overall error:    %.3e\n', err)
else
%% ========================================================================
% Plot one slice of u and v in one figure
% ========================================================================

nn = n(1);   % middle/final slice index, depending on your grid definition

figure;
tiledlayout(1, 2, 'TileSpacing', 'compact', 'Padding', 'compact');

%% Plot u
nexttile;
surf(X{1}(:,:,nn), X{2}(:,:,nn), U{1}(:,:,nn), ...
    'EdgeColor', 'none');

axis equal tight;
view(2);

xlabel('x_1');
ylabel('x_2');
title('u');

colorbar;

%% Plot v
nexttile;
surf(X{1}(:,:,nn), X{2}(:,:,nn), U{2}(:,:,nn), ...
    'EdgeColor', 'none');

axis equal tight;
view(2);

xlabel('x_1');
ylabel('x_2');
title('v');

colorbar;

%% Figure size for export
set(gcf, 'Units', 'inches', 'Position', [1 1 9 4]);
set(gcf, 'PaperUnits', 'inches');
set(gcf, 'PaperPosition', [0 0 9 4]);
set(gcf, 'PaperSize', [9 4]);

drawnow;

print('Brusselator3D_uv_slice', '-depsc');
end

fprintf('Wall-clock time: %.2f s\n',wctime)

rmpath('phipmsimuliom')
rmpath('KronPACK/src')

function [t, uFinal, U] = ExpH3(Nfun, L, JN, t0, tEnd, u0, nSteps, tol, c2)
%EXPH4s2 Fourth-order 2 stage exponential Hermite method.
%
%   Solves
%
%       u'(t) = L*u(t) + N(u(t)),     u(t0) = u0.
%
%   Define the full vector field
%
%       F(u) = L*u + N(u).
%
%   The ExpH4s2 scheme (c2 = 0.5) is 
%
%       U_{n2} = u_n + c2*h*phi_1(c2*h*L) F(u_n)
%              + c2^2*h^2*phi_2(c2*h*L) N'(u_n)F(u_n),
%
%       H_{n2} = N'(U_{n2})F(U_{n2}) - N'(u_n)F(u_n),
%
%       u_{n+1} = u_n + h*phi_1(h*L) F(u_n)
%              + h^2*phi_2(h*L) N'(u_n)F(u_n)
%              + (1/c2)*h^2*phi_3(h*L) H_{n2}.
%
%   Inputs:
%       Nfun    - nonlinear function handle, Nfun(u)
%       L       - linear matrix/operator
%       JN      - Jacobian of N, JN(u)
%       t0      - initial time
%       tEnd    - final time
%       u0      - initial vector
%       nSteps  - number of time steps
%       tol     - tolerance for phipm_simul_iom
%       c2      - internal stage coefficient, default c2 = 1/2
%
%   Outputs:
%       t       - time grid
%       uFinal  - numerical solution at tEnd
%       U       - optional full solution history

    if nargin < 8 || isempty(tol)
        tol = 1e-10;
    end

    if nargin < 9 || isempty(c2)
        c2 = 1/2;
    end

    h = (tEnd - t0) / nSteps;
    t = linspace(t0, tEnd, nSteps + 1);

    u = u0(:);
    n = length(u);
    zeroVec = zeros(n, 1);

    saveSolution = (nargout >= 3);

    if saveSolution
        U = zeros(n, nSteps + 1);
        U(:, 1) = u;
    end

    for k = 1:nSteps

        % Full vector field:
        %
        % F_n = L*u_n + N(u_n)
        Nu = Nfun(t(k),u);
        Fn = L(u) + Nu;

        % Nonlinear Hermite derivative:
        %
        % G_n = N'(u_n)F_n
        Gn = JN(t(k),u) * Fn;

        % ---------------------------------------------------------------
        % Stage value:
        %
        % U_{n2} = u_n + c2*h*phi_1(c2*h*L)F_n
        %        + c2^2*h^2*phi_2(c2*h*L)G_n
        %
        % phipm_simul_iom(t,A,[v0,v1,v2]) computes
        %
        %        t*phi_1(tA)v1 + t^2*phi_2(tA)v2
        %
        % because v0 = 0 here.
        % ---------------------------------------------------------------
        stageIncrement = phipm_simul_iom( ...
            c2*h, L, [zeroVec, Fn, Gn], tol, 1, 2);

        uStage = u + stageIncrement;

        % Full vector field at the stage:
        %
        % F(U_{n2}) = L*U_{n2} + N(U_{n2})
        NStage = Nfun(t(k)+c2*h,uStage);
        FStage = L(uStage) + NStage;

        % Difference term:
        %
        % H_{n2} = N'(U_{n2})F(U_{n2}) - N'(u_n)F(u_n)
        Hn2 = JN(t(k)+c2*h,uStage)*FStage - Gn;

        % ---------------------------------------------------------------
        % Final update:
        %
        % u_{n+1} = u_n
        %          + h*phi_1(hL)F_n
        %          + h^2*phi_2(hL)G_n
        %          + (1/c2)*h^2*phi_3(hL)H_{n2}.
        %
        % Important:
        % phipm_simul_iom(h,L,[0,Fn,Gn,V3]) gives
        %
        %     h*phi_1(hL)Fn
        %   + h^2*phi_2(hL)Gn
        %   + h^3*phi_3(hL)V3.
        %
        % To obtain h^2*phi_3(hL)Hn2/c2, we set
        %
        %     V3 = Hn2/(c2*h).
        % ---------------------------------------------------------------
        finalIncrement = phipm_simul_iom( ...
            h, L, [zeroVec, Fn, Gn, Hn2/(c2*h)], tol, 1, 2);

        u = u + finalIncrement;

        if saveSolution
            U(:, k + 1) = u;
        end
    end

    uFinal = u;
end

function [tgrid, yFinal, Yhist] = exprb42_phipm_simul_iom(Ffun,Jfun,t0,tEnd,y0,nsteps,tol)
% exprb42_phipm_simul_iom Fourth-order two-stage exponential Rosenbrock method.
%
% Generic vector form:
%
%     y' = F(t,y).
%
% At each step:
%
%     J_n = F_y(t_n,y_n).
%
% The method is:
%
%     Y_{n2} = y_n + (3/4)h phi_1((3/4)hJ_n) F_n,
%
%     D_{n2} = F(t_n,Y_{n2}) - F(t_n,y_n) - J_n(Y_{n2}-y_n),
%
%     y_{n+1} = y_n + h phi_1(hJ_n)F_n
%                   + h phi_3(hJ_n)(32/9)D_{n2}.
%
% Inputs:
%     Ffun   - full RHS, Ffun(t,y)
%     Jfun   - full Jacobian, Jfun(t,y)
%     t0     - initial time
%     tEnd   - final time
%     y0     - initial vector
%     nsteps - number of time steps
%     tol    - tolerance for phipm_simul_iom
%
% Outputs:
%     tgrid  - time grid
%     yFinal - solution at tEnd
%     Yhist  - optional full trajectory

    if nargin < 8 || isempty(tol)
        tol = 1e-10;
    end

    h = (tEnd - t0)/nsteps;
    tgrid = linspace(t0,tEnd,nsteps+1);

    y = y0(:);
    m = length(y);

    zeroVec = zeros(m,1);

    saveHist = (nargout >= 3);

    if saveHist
        Yhist = zeros(m,nsteps+1);
        Yhist(:,1) = y;
    end

    for k = 1:nsteps

        tn = tgrid(k);

        % ================================================================
        % Step data
        % ================================================================
        Fn = Ffun(tn,y);
        Jn = Jfun(tn,y);

        % ================================================================
        % Internal stage
        %
        % Y_{n2} = y_n + alpha phi_1(alpha J_n) F_n
        %
        % alpha = 3h/4.
        %
        % phipm_simul_iom(1,alpha*Jn,[0,Fn]) gives
        %
        %     phi_1(alpha J_n)F_n.
        % ================================================================
        alpha = (3/4)*h;

        incStage = alpha * phipm_simul_iom( ...
            1, alpha*Jn, [zeroVec, Fn], tol, 1, 2);

        yStage = y + incStage;

        % ================================================================
        % Rosenbrock nonlinear defect
        %
        % D_{n2} = F(Y_{n2}) - F(y_n) - J_n(Y_{n2}-y_n)
        %
        % This is equivalent to
        %
        % D_{n2} = g_n(Y_{n2}) - g_n(y_n)
        %
        % but avoids explicitly forming N'(y_n).
        % ================================================================
        FStage = Ffun(tn+alpha,yStage);

        Dn2 = FStage - Fn - Jn*(yStage - y);

        % ================================================================
        % Final update
        %
        % y_{n+1} = y_n
        %          + h phi_1(hJ_n)F_n
        %          + h phi_3(hJ_n)(32/9)D_{n2}
        %
        % phipm_simul_iom(1,hJn,[0,Fn,0,(32/9)Dn2]) gives
        %
        %     phi_1(hJ_n)F_n + phi_3(hJ_n)(32/9)D_{n2}.
        % ================================================================
        incFinal = h * phipm_simul_iom( ...
            1, h*Jn, [zeroVec, Fn, zeroVec, (32/9)*Dn2], tol, 1, 2);

        y = y + incFinal;

        if saveHist
            Yhist(:,k+1) = y;
        end
    end

    yFinal = y;
end
