% Consider Advection Diffusion Reaction Model:
%
%   u_t = epsilon(u_xx + u_yy) - alpha(u_x + u_y)
%         + gamma u(u - 1/2)(1 - u)
%
% on [0,1]^2 with homogeneous Neumann boundary condition.

close all;
clear;
clc;

addpath('../integrators','../phipmsimuliom');


%% ========================================================================
% 1. Parameters

epsilon = 0.01;
alpha   = -10;
gamma   = 100;

tspan = [0, 0.08];
t0    = tspan(1);
tEnd  = tspan(2);

n = 100;

tol = 1e-10;

%% ========================================================================
% 2. Build ADR 2D problem
%
% Adr_2d_function returns old notation:
%
%     F(U) = A*U + g(U)
%
% For ExpH, we use
%
%     U' = L*U + Nfun(U).
% ========================================================================

[F, L, Nfun, JN, Jn, U0, hx] = Adr_2d_function(epsilon, alpha, gamma, n);

%% ========================================================================
% 3. Reference solution
% ========================================================================

script_dir = fileparts(mfilename('fullpath'));

load(fullfile(script_dir, 'Data', 'Adr_2d_sol_100.mat'));

%% ========================================================================
% 4. Time-step refinement
% ========================================================================

k = 40;
nStepsList = [k, 2*k, 4*k, 8*k];

hPlot = 1./nStepsList;

ExpH4s3_err = zeros(size(nStepsList));
ExpH4s2_err = zeros(size(nStepsList));

ExpRB43_err = zeros(size(nStepsList));
ExpRB42_err = zeros(size(nStepsList));
ExpRK45_err = zeros(size(nStepsList));

CPU_T_ExpH4s3 = zeros(size(nStepsList));
CPU_T_ExpH4s2 = zeros(size(nStepsList));

CPU_T_ExpRB43 = zeros(size(nStepsList));
CPU_T_ExpRB42 = zeros(size(nStepsList));
CPU_T_ExpRK45 = zeros(size(nStepsList));

%% ========================================================================
% 5. Run methods
% ========================================================================

for i = 1:length(nStepsList)

    nSteps = nStepsList(i);

    fprintf('Running nSteps = %d\n', nSteps);

    % --------------------------------------------------------------------
    % ExpH4s3
    % --------------------------------------------------------------------
    tic;
    [t, ExpH4s3_sol] = ExpH4s3( ...
        Nfun, L, JN, t0, tEnd, U0, nSteps, tol, [0.35, 0.8]);
    CPU_T_ExpH4s3(i) = toc;

    ExpH4s3_err(i) = norm(exac_sol - ExpH4s3_sol, inf);

    % --------------------------------------------------------------------
    tic;
    [t, ExpH4s2_sol] = ExpH3( ...
        Nfun, L, JN, t0, tEnd, U0, nSteps, tol, 1/2);
    CPU_T_ExpH4s2(i) = toc;

    ExpH4s2_err(i) = norm(exac_sol - ExpH4s2_sol, inf);

    % --------------------------------------------------------------------
    tic;
    [t, ExpRB43_sol] = expRB43( ...
        Nfun, L, JN, t0, tEnd, U0, nSteps, tol);
    CPU_T_ExpRB43(i) = toc;

    ExpRB43_err(i) = norm(exac_sol - ExpRB43_sol, inf);

    % --------------------------------------------------------------------
    tic;
    [t, ExpRB42_sol] = expRB42( ...
        Nfun, L, JN, t0, tEnd, U0, nSteps, tol);
    CPU_T_ExpRB42(i) = toc;

    ExpRB42_err(i) = norm(exac_sol - ExpRB42_sol, inf);
    
    % --------------------------------------------------------------------
    tic;
    [t, ExpRK45_sol] = ExpRK45( ...
        Nfun, L, t0, tEnd, U0, nSteps, tol);
    CPU_T_ExpRK45(i) = toc;

    ExpRK45_err(i) = norm(exac_sol - ExpRK45_sol, inf);

end

%% ========================================================================
% 6. Observed orders
% ========================================================================

ExpH4s3_order = zeros(1, length(nStepsList)-1);
ExpH4s2_order = zeros(1, length(nStepsList)-1);
ExpRB43_order = zeros(1, length(nStepsList)-1);
ExpRB42_order = zeros(1, length(nStepsList)-1);
ExpRK45_order = zeros(1, length(nStepsList)-1);

for i = 1:length(nStepsList)-1

    ExpH4s3_order(i) = log(ExpH4s3_err(i)/ExpH4s3_err(i+1))/log(2);

    ExpH4s2_order(i) = log(ExpH4s2_err(i)/ExpH4s2_err(i+1))/log(2);

    ExpRB43_order(i) = log(ExpRB43_err(i)/ExpRB43_err(i+1))/log(2);

    ExpRB42_order(i) = log(ExpRB42_err(i)/ExpRB42_err(i+1))/log(2);

    ExpRK45_order(i) = log(ExpRK45_err(i)/ExpRK45_err(i+1))/log(2);

end

disp('Observed orders:')
ExpH4s3_order
ExpH4s2_order
ExpRB43_order
ExpRB42_order
ExpRK45_order

%% ========================================================================
% 7. Order plot
% ========================================================================

set(0, 'DefaultTextFontSize', 15);
set(0, 'DefaultAxesFontSize', 15);

figure;

loglog(hPlot, ExpH4s3_err, 'o-', ...
    'LineWidth', 2, 'MarkerSize', 10);
hold on;

loglog(hPlot, ExpH4s2_err, '>-', ...
    'LineWidth', 2, 'MarkerSize', 10);

loglog(hPlot, ExpRB43_err, 'v-', ...
    'LineWidth', 2, 'MarkerSize', 10);

loglog(hPlot, ExpRB42_err, 'd-', ...
    'LineWidth', 2, 'MarkerSize', 10);

loglog(hPlot, ExpRK45_err, 's-', ...
    'LineWidth', 2, 'MarkerSize', 10);

% Reference fourth-order slope
loglog(hPlot, hPlot.^4*10000, '--', ...
    'LineWidth', 2);

grid on;

k  = gca;   
xt = [0, flip(hPlot)]; 
yt = [ 1e-9,1e-8, 1e-7, 1e-6, 1e-5, 1e-4 1e-3]; 
axis([0 max(hPlot) ExpH4s3_err(end) ExpRK45_err(1)]) %


set(k,'YTick',yt) 
set(k,'XTick',xt )
xticklabels({'0','640','320','160','80'})
set(k,'ticklength',2*get(gca,'ticklength'))
set(k, 'OuterPosition', [0 0 1 1])
legend('ExpH4s3','ExpH4s2','exprb43','exprb42','expRK4s5','Order4','Location','SouthEast')
title('Order Plot')
xlabel('Number of time steps')
ylabel('Error')

outputFolder = fullfile(script_dir, 'Test Results');

if ~exist(outputFolder, 'dir')
    mkdir(outputFolder);
end

outputFile = fullfile(outputFolder, 'ExpH_Adr2d_Order');
print(gcf, outputFile, '-depsc');

rmpath('../integrators','../phipmsimuliom');