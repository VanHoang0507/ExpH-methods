clc;
close all;
clear all;

%% Add paths robustly

addpath('../integrators')
addpath('../phipmsimuliom')

%% ========================================================================
% 1. Space and time intervals
% ========================================================================

a = 0;
b = 1;

t0    = 0;
t_end = 1;

m = 400;

delta_x = (b - a)/m;

x = linspace(a, b, m + 1);
X = x(2:end-1)';

%% ========================================================================
% 2. Spatial discretization
% ========================================================================

e = ones(m-1,1);

Aspace = spdiags( ...
    [e/delta_x^2, -2*e/delta_x^2, e/delta_x^2], ...
    [-1 0 1], m-1, m-1);

% Augmented matrix for U = [tau; u]
L = blkdiag(sparse(1,1), Aspace);

%% ========================================================================
% 3. Initial condition
% ========================================================================

s = sin(pi*X);

U0 = [t0; s];

%% ========================================================================
% 4. ExpH form: U' = L*U + Nfun(U)
% ========================================================================

Nfun = @(U) N(U, s, Aspace);

JN = @(U) J(U, s, Aspace);

u_true = @(x,t) sin(pi*x).*exp(t);


%% ========================================================================
% 5. Time-step refinement
% ========================================================================

Nsteps = [2 4 8 16];
h = (t_end - t0)./Nsteps;

tol = 1e-10;

ExpH2_err      = zeros(size(Nsteps));
ExpH3_c14_err  = zeros(size(Nsteps));
ExpH3_c12_err  = zeros(size(Nsteps));
ExpH4s3_err    = zeros(size(Nsteps));
ExpH5s3_err    = zeros(size(Nsteps));
ExpH5s4_err    = zeros(size(Nsteps));
ExpH5s5_err    = zeros(size(Nsteps));

u_exact_final = u_true(X, t_end);

%% ========================================================================
% 6. Run methods
% ========================================================================

for i = 1:length(Nsteps)

    nSteps = Nsteps(i);

    fprintf('\nRunning N = %d, h = %.4e\n', nSteps, h(i));

    [~, ExpH2_sol] = ExpH2( ...
        Nfun, L, JN, t0, t_end, U0, nSteps, tol);

    ExpH2_err(i) = norm( ...
        u_exact_final - ExpH2_sol(2:end), inf);

    [~, ExpH3_c14_sol] = ExpH3( ...
        Nfun, L, JN, t0, t_end, U0, nSteps, tol, 1/4);

    ExpH3_c14_err(i) = norm( ...
        u_exact_final - ExpH3_c14_sol(2:end), inf);

    [~, ExpH3_c12_sol] = ExpH3( ...
        Nfun, L, JN, t0, t_end, U0, nSteps, tol, 1/2);

    ExpH3_c12_err(i) = norm( ...
        u_exact_final - ExpH3_c12_sol(2:end), inf);

    [~, ExpH4s3_sol] = ExpH4s3( ...
        Nfun, L, JN, t0, t_end, U0, nSteps, tol, [0.35, 0.8]);

    ExpH4s3_err(i) = norm( ...
        u_exact_final - ExpH4s3_sol(2:end), inf);

    [~, ExpH5s3_sol] = ExpH5s3( ...
        Nfun, L, JN, t0, t_end, U0, nSteps, tol, 0.1);

    ExpH5s3_err(i) = norm( ...
        u_exact_final - ExpH5s3_sol(2:end), inf);

    [~, ExpH5s4_sol] = ExpH5s4( ...
        Nfun, L, JN, t0, t_end, U0, nSteps, tol, 1/6, 1/5);

    ExpH5s4_err(i) = norm( ...
        u_exact_final - ExpH5s4_sol(2:end), inf);

    [~, ExpH5s5_sol] = ExpH5s5( ...
        Nfun, L, JN, t0, t_end, U0, nSteps, tol);

    ExpH5s5_err(i) = norm( ...
        u_exact_final - ExpH5s5_sol(2:end), inf);
end

%% ========================================================================
% 7. Compute observed orders
% ========================================================================

compute_order = @(err) ...
    log(err(1:end-1)./err(2:end)) ...
    ./log(h(1:end-1)./h(2:end));

ExpH2_order      = compute_order(ExpH2_err);
ExpH3_c14_order  = compute_order(ExpH3_c14_err);
ExpH3_c12_order  = compute_order(ExpH3_c12_err);
ExpH4s3_order    = compute_order(ExpH4s3_err);
ExpH5s3_order    = compute_order(ExpH5s3_err);
ExpH5s4_order    = compute_order(ExpH5s4_err);
ExpH5s5_order    = compute_order(ExpH5s5_err);

disp('Observed orders:')

ExpH2_order
ExpH3_c14_order
ExpH3_c12_order
ExpH4s3_order
ExpH5s3_order
ExpH5s4_order
ExpH5s5_order

%% ========================================================================
% 8. Order plot
% ========================================================================

set(0, 'DefaultTextFontSize', 15);
set(0, 'DefaultAxesFontSize', 15);

figure(1);

loglog(h, ExpH2_err,     'o-', 'LineWidth', 2, ...
    'Color', '#0072BD', 'MarkerSize', 10);
hold on;

loglog(h, ExpH3_c14_err, 'v-', 'LineWidth', 2, ...
    'Color', '#D95319', 'MarkerSize', 10);

loglog(h, ExpH3_c12_err, '*-', 'LineWidth', 2, ...
    'Color', '#EDB120', 'MarkerSize', 12);

loglog(h, ExpH4s3_err,   '>-', 'LineWidth', 2, ...
    'Color', '#7E2F8E', 'MarkerSize', 10);

loglog(h, ExpH5s3_err,   '^-', 'LineWidth', 2, ...
    'Color', '#4DBEEE', 'MarkerSize', 10);

loglog(h, ExpH5s4_err,   's-', 'LineWidth', 2, ...
    'Color', '#A2142F', 'MarkerSize', 10);

loglog(h, ExpH5s5_err,   'o-', 'LineWidth', 2, ...
    'Color', '#77AC30', 'MarkerSize', 10);

% Reference slopes
loglog(h, h.^2 * ExpH2_err(1), '-.', 'LineWidth', 1.2, ...
    'Color', '#77AC30');

loglog(h, h.^3 * ExpH3_c14_err(1)*4, '-+', 'LineWidth', 1.2, ...
    'Color', '#77AC30');

loglog(h, h.^4 * ExpH4s3_err(1)*4, '-x', 'LineWidth', 1.2, ...
    'Color', '#77AC30');

loglog(h, h.^5 * ExpH5s5_err(1) , '--', 'LineWidth', 1.2, ...
    'Color', '#77AC30');

grid on;

k = gca;

xt = [0, flip(h)];
yt = [1e-10, 1e-9, 1e-8, 1e-7, ...
      1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1];

axis([0, max(h), ExpH5s5_err(end), ExpH2_err(1)])

set(k, 'YTick', yt);
set(k, 'XTick', xt);

% Same labeling idea:
% h = [1/2,1/4,1/8,1/16] corresponds to N = [2,4,8,16]
xticklabels({'0','16','8','4','2'});

set(k, 'TickLength', 2*get(gca,'TickLength'));
set(k, 'OuterPosition', [0 0 1 1]);

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
    'Slope 4', ...
    'Slope 5', ...
    'Location', 'eastoutside');

title('Order Plot');
xlabel('Number of time steps');
ylabel('Error');

print('ExpH_HocOst_Order','-depsc');

rmpath('../integrators')
rmpath('../phipmsimuliom')

%% ========================================================================
% Local nonlinear function
% ========================================================================

function value = N(U, s, Aspace)

    tau = U(1);
    u   = U(2:end);

    z = exp(tau)*s;

    phi_h = ...
        z ...
        - Aspace*z ...
        - 1./(1 + z.^2);

    value = [ ...
        1;
        phi_h + 1./(1 + u.^2)
    ];

end

%% ========================================================================
% Local Jacobian
% ========================================================================

function Jacobian = J(U, s, Aspace)

    tau = U(1);
    u   = U(2:end);

    n = length(u);

    z = exp(tau)*s;

    % Derivative with respect to u
    Ju = spdiags( ...
        -2*u./(1 + u.^2).^2, ...
        0, n, n);

    Jacobian = blkdiag(sparse(1,1), Ju);

    Jacobian(2:end,1) = ...
        z ...
        - Aspace*z ...
        + 2*z.^2./(1 + z.^2).^2;

end
