clc;
close all;
clear all;

addpath('../integrators','../phipmsimuliom');

global m X T1 D1 D2

%% ========================================================================
% 1. Space and time intervals
% ========================================================================

a = 0;
b = 1;

t0    = 0;
t_end = 1;

tspan = [t0, t_end];

m = 400;

delta_x = (b - a)/m;

x = linspace(a, b, m + 1);
X = x(2:end-1)';

%% ========================================================================
% 2. Spatial discretization
% ========================================================================

e = ones(m-1, 1);

D2 = spdiags( ...
    [e/delta_x^2, -2*e/delta_x^2, e/delta_x^2], ...
    [-1, 0, 1], m-1, m-1);

D1 = spdiags( ...
    [-e/(2*delta_x), e/(2*delta_x)], ...
    [-1, 1], m-1, m-1);

Aspace = D2;

% Augmented linear operator for U = [tau; u]
L = blkdiag(0, Aspace);

A_al = eye(m-1);

%% ========================================================================
% 3. Integral matrix
% ========================================================================

T_simson = ones(m-1, m-1);

for j = 2:2:m-1
    T_simson(:, j) = 4*ones(m-1, 1);

    if j + 1 <= m-1
        T_simson(:, j+1) = 2*ones(m-1, 1);
    end
end

T_simson = (delta_x/3)*T_simson;

T1 = blkdiag(0, T_simson);

%% ========================================================================
% 4. Initial condition
% ========================================================================

U0_space = X - X.^2;
U0 = [t0; U0_space];       % U(1) = tau, U(2:end) = u

%% ========================================================================
% 5. ExpH form
%
% We write the problem as
%
%     U' = L*U + Nfun(U).
%
% The ExpH solvers internally form
%
%     F_n = L*U_n + Nfun(U_n),
%     G_n = JN(U_n)*F_n.
% ========================================================================

Nfun = @(U) [ ...
    1; ...
    (11/6 + X - X.^2).*exp(U(1)) + T_simson*U(2:end) ...
];

JN = @(U) J_int(U);

Ffull = @(U) L*U + Nfun(U);
Fode  = @(t,U) L*U + Nfun(U);

%% ========================================================================
% 6. Reference solution by ode15s
% ========================================================================

options = odeset('RelTol', 2.22045e-14, 'AbsTol', 2.22045e-14);

[t_exac, exac_sol] = ode15s(Fode, tspan, U0, options);

exac_sol = exac_sol';
u_true = exac_sol(2:end, end);

%% ========================================================================
% 7. Time-step refinement
% ========================================================================

Nsteps = [4, 8, 16, 32];
h = (t_end - t0)./Nsteps;

tol = 1e-10;

ExpH2_err      = zeros(size(Nsteps));
ExpH3_c14_err  = zeros(size(Nsteps));
ExpH3_c12_err  = zeros(size(Nsteps));
ExpH4s3_err    = zeros(size(Nsteps));
ExpH5s3_err    = zeros(size(Nsteps));
ExpH5s4_err    = zeros(size(Nsteps));
ExpH5s5a_err   = zeros(size(Nsteps));

%% ========================================================================
% 8. Run ExpH methods
% ========================================================================

for i = 1:length(Nsteps)

    nSteps = Nsteps(i);

    fprintf('Running N = %d, h = %.4e\n', nSteps, h(i));

    [t, ExpH2_sol] = ExpH2( ...
        Nfun, L, JN, t0, t_end, U0, nSteps, tol);

    ExpH2_err(i) = norm(A_al*(u_true - ExpH2_sol(2:end)), inf);

    [t, ExpH3_c14_sol] = ExpH3( ...
        Nfun, L, JN, t0, t_end, U0, nSteps, tol, 1/4);

    ExpH3_c14_err(i) = norm(A_al*(u_true - ExpH3_c14_sol(2:end)), inf);

    [t, ExpH3_c12_sol] = ExpH3( ...
        Nfun, L, JN, t0, t_end, U0, nSteps, tol, 1/2);

    ExpH3_c12_err(i) = norm(A_al*(u_true - ExpH3_c12_sol(2:end)), inf);

    [t, ExpH4s3_sol] = ExpH4s3( ...
        Nfun, L, JN, t0, t_end, U0, nSteps, tol, [0.35, 0.8]);

    ExpH4s3_err(i) = norm(A_al*(u_true - ExpH4s3_sol(2:end)), inf);

    [t, ExpH5s3_sol] = ExpH5s3( ...
        Nfun, L, JN, t0, t_end, U0, nSteps, tol, 0.01);

    ExpH5s3_err(i) = norm(A_al*(u_true - ExpH5s3_sol(2:end)), inf);

    [t, ExpH5s4_sol] = ExpH5s4( ...
        Nfun, L, JN, t0, t_end, U0, nSteps, tol, 0.1, 0.1);

    ExpH5s4_err(i) = norm(A_al*(u_true - ExpH5s4_sol(2:end)), inf);

    [t, ExpH5s5a_sol] = ExpH5s5a( ...
        Nfun, L, JN, t0, t_end, U0, nSteps, tol);

    ExpH5s5a_err(i) = norm(A_al*(u_true - ExpH5s5a_sol(2:end)), inf);

end

%% ========================================================================
% 9. Observed convergence orders
% ========================================================================

ExpH2_order     = zeros(1, length(Nsteps)-1);
ExpH3_c14_order = zeros(1, length(Nsteps)-1);
ExpH3_c12_order = zeros(1, length(Nsteps)-1);
ExpH4s3_order   = zeros(1, length(Nsteps)-1);
ExpH5s3_order   = zeros(1, length(Nsteps)-1);
ExpH5s4_order   = zeros(1, length(Nsteps)-1);
ExpH5s5a_order  = zeros(1, length(Nsteps)-1);

for i = 1:length(Nsteps)-1

    ExpH2_order(i)     = log(ExpH2_err(i)/ExpH2_err(i+1))/log(2);
    ExpH3_c14_order(i) = log(ExpH3_c14_err(i)/ExpH3_c14_err(i+1))/log(2);
    ExpH3_c12_order(i) = log(ExpH3_c12_err(i)/ExpH3_c12_err(i+1))/log(2);
    ExpH4s3_order(i)   = log(ExpH4s3_err(i)/ExpH4s3_err(i+1))/log(2);
    ExpH5s3_order(i)   = log(ExpH5s3_err(i)/ExpH5s3_err(i+1))/log(2);
    ExpH5s4_order(i)   = log(ExpH5s4_err(i)/ExpH5s4_err(i+1))/log(2);
    ExpH5s5a_order(i)  = log(ExpH5s5a_err(i)/ExpH5s5a_err(i+1))/log(2);

end

disp('Observed orders:')
ExpH2_order
ExpH3_c14_order
ExpH3_c12_order
ExpH4s3_order
ExpH5s3_order
ExpH5s4_order
ExpH5s5a_order

%% ========================================================================
% 10. Order plot
% ========================================================================

set(0, 'DefaultTextFontSize', 15);
set(0, 'DefaultAxesFontSize', 15);

figure(1);

loglog(h, ExpH2_err,     'o-', 'LineWidth', 2, 'Color', '#0072BD', 'MarkerSize', 10);
hold on;
loglog(h, ExpH3_c14_err, 'v-', 'LineWidth', 2, 'Color', '#D95319', 'MarkerSize', 10);
loglog(h, ExpH3_c12_err, '*-', 'LineWidth', 2, 'Color', '#EDB120', 'MarkerSize', 12);
loglog(h, ExpH4s3_err,   '>-', 'LineWidth', 2, 'Color', '#7E2F8E', 'MarkerSize', 10);
loglog(h, ExpH5s3_err,   '^-', 'LineWidth', 2, 'Color', '#4DBEEE', 'MarkerSize', 10);
loglog(h, ExpH5s4_err,   's-', 'LineWidth', 2, 'Color', '#A2142F', 'MarkerSize', 10);
loglog(h, ExpH5s5a_err,  'o-', 'LineWidth', 2, 'Color', '#77AC30', 'MarkerSize', 10);

% Reference slopes
loglog(h, h.^2,          '-.', 'LineWidth', 1.2, 'Color', '#77AC30');
loglog(h, h.^3/6,        '-+', 'LineWidth', 1.2, 'Color', '#77AC30');
loglog(h, h.^3.5*0.005,   '-x', 'LineWidth', 1.2, 'Color', '#77AC30');
loglog(h, h.^4.5*0.0004, '--', 'LineWidth', 1.2, 'Color', '#77AC30');

grid on;

k  = gca;   
xt = [0, flip(h)]; 
yt = [1e-10 1e-9,1e-8, 1e-7, 1e-6, 1e-5, 1e-4 1e-3]; 
axis([0 max(h) ExpH5s5a_err(end) ExpH2_err(1)]) %


set(k,'YTick',yt) 
set(k,'XTick',xt )
xticklabels({'0','32','16','8','4'})
set(k,'ticklength',2*get(gca,'ticklength'))
set(k, 'OuterPosition', [0 0 1 1])
legend( ...
    'ExpH2', ...
    'ExpH3', ...
    'ExpH4s2', ...
    'ExpH4s3', ...
    'ExpH5s3', ...
    'ExpH5s4', ...
    'ExpH5s5', ...
    'Slope 2', ...
    'Slope 3', ...
    'Slope 3.5', ...
    'Slope 4.5', ...
    'Location', 'eastoutside');
title('Order Plot')
xlabel('Number of time steps')
ylabel('Error')

print('ExpH_HocOst_integral_Order', '-depsc');

rmpath('../integrators','../phipmsimuliom');

%% ========================================================================
% 11. Local functions
% ========================================================================

function [Jacobian] = J_int(U)

global X T1

    Jacobian = T1;

    Jacobian(2:end, 1) = (11/6 + X - X.^2).*exp(U(1));

end

function out = H1_norm(U, x)

global D1

    out = sqrt(norm(U, 2)^2 + norm(D1*U, 2)^2);

end

function [t, uFinal, Uhist] = ExpH5s5a(Nfun, L, JN, t0, tEnd, u0, nSteps, tol)
%EXPH5S5A Fifth-order five-stage ExpH method with custom nodes.
%
%   This is the ExpH version of your previous TDexp5s5a.
%
%   It solves
%
%       u' = L*u + Nfun(u).
%
%   The solver internally forms
%
%       F_n = L*u_n + Nfun(u_n),
%       G_n = JN(u_n)*F_n.

    h = (tEnd - t0)/nSteps;
    t = linspace(t0, tEnd, nSteps + 1);

    u = u0(:);
    mloc = length(u);
    zero = zeros(mloc, 1);

    saveTrajectory = (nargout >= 3);

    if saveTrajectory
        Uhist = zeros(mloc, nSteps + 1);
        Uhist(:, 1) = u;
    end

    c2 = 1/4;
    c3 = 1/4;
    c4 = 1/2;
    c5 = 1;

    d33 =  c4*c5       / (c3*(c3 - c4)*(c3 - c5));
    d34 = -2*(c4 + c5) / (c3*(c3 - c4)*(c3 - c5));
    d35 =  6           / (c3*(c3 - c4)*(c3 - c5));

    d43 =  c3*c5       / (c4*(c4 - c3)*(c4 - c5));
    d44 = -2*(c3 + c5) / (c4*(c4 - c3)*(c4 - c5));
    d45 =  6           / (c4*(c4 - c3)*(c4 - c5));

    d53 =  c3*c4       / (c5*(c5 - c3)*(c5 - c4));
    d54 = -2*(c3 + c4) / (c5*(c5 - c3)*(c5 - c4));
    d55 =  6           / (c5*(c5 - c3)*(c5 - c4));

    for k = 1:nSteps

        % Step data
        Nu = Nfun(u);

        % Full vector field
        Fn = L*u + Nu;

        % Hermite derivative
        Gn = JN(u)*Fn;

        % Stage U_{n2}
        inc2 = phipm_simul_iom( ...
            c2*h, L, [zero, Fn, Gn], tol, 1, 2);

        Un2 = u + inc2;

        F2 = L*Un2 + Nfun(Un2);
        Hn2 = JN(Un2)*F2 - Gn;

        % Stages U_{n3}, U_{n4}, U_{n5}
        stageIncs = phipm_simul_iom( ...
            [c3, c4, c5]*h, L, [zero, Fn, Gn, Hn2/(c2*h)], tol, 1, 2);

        Un3 = u + stageIncs(:, 1);
        Un4 = u + stageIncs(:, 2);
        Un5 = u + stageIncs(:, 3);

        F3 = L*Un3 + Nfun(Un3);
        F4 = L*Un4 + Nfun(Un4);
        F5 = L*Un5 + Nfun(Un5);

        Hn3 = JN(Un3)*F3 - Gn;
        Hn4 = JN(Un4)*F4 - Gn;
        Hn5 = JN(Un5)*F5 - Gn;

        % Final Hermite interpolation coefficients
        B3 = d33*Hn3 + d43*Hn4 + d53*Hn5;
        B4 = d34*Hn3 + d44*Hn4 + d54*Hn5;
        B5 = d35*Hn3 + d45*Hn4 + d55*Hn5;

        % Final ExpH update
        finalIncrement = phipm_simul_iom( ...
            h, L, [zero, Fn, Gn, B3/h, B4/h^2, B5/h^3], tol, 1, 2);

        u = u + finalIncrement;

        if saveTrajectory
            Uhist(:, k + 1) = u;
        end
    end

    uFinal = u;

end