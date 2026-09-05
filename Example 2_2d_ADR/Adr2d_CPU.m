% Consider Advection Diffusion Reaction Model:
%
%   u_t = epsilon(u_xx + u_yy) - alpha(u_x + u_y)
%         + gamma*u*(u - 1/2)*(1 - u)
%
% on [0,1]^2 with homogeneous Neumann boundary condition.

close all;
clear;
clc;

addpath('../integrators','../phipmsimuliom');

%% ========================================================================
% 1. Problem parameters
% ========================================================================

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
% Old notation from Adr_2d_function:
%
%   F(U) = A*U + g(U)
%
% New ExpH notation:
%
%   U' = L*U + Nfun(U)
% ========================================================================

[F, A, g, J, Jn, U0, h] = Adr_2d_function(epsilon, alpha, gamma, n);

L    = A;
Nfun = g;
JN   = J;

%% ========================================================================
% 3. Reference solution
% ========================================================================

script_dir = fileparts(mfilename('fullpath'));

load(fullfile(script_dir, 'Data', 'Adr_2d_sol_100.mat'));

exac_sol = exac_sol(:);
U0       = U0(:);

%% ========================================================================
% 4. Time-step lists for CPU comparison
% ========================================================================

% m1 = [6, 11, 17, 28, 58 100 200];    % ExpH4s3
% m2 = [8, 15, 29, 54, 98, 180, 300];    % ExpH3, c2 = 1/2
% m3 = [10, 17, 30, 60, 110, 200, 350];  % ExpRB42
% m4 = [9, 17, 29, 50, 88, 150, 280];    % ExpRB43
% m5 = [8, 18, 35, 64, 114, 200 370 ];   % ExpRK4s5

m1 = [ 17, 28, 58 100 200];    % ExpH4s3
m2 = [ 29, 54, 98, 180, 300];    % ExpH3, c2 = 1/2
m3 = [ 30, 60, 110, 200, 350];  % ExpRB42
m4 = [ 29, 50, 88, 150, 280];    % ExpRB43
m5 = [ 35, 64, 114, 200 370 ];   % ExpRK4s5

numTests = length(m1);

CPU_T_ExpH4s3 = zeros(1, numTests);
CPU_T_ExpH3   = zeros(1, numTests);
CPU_T_ExpRB42 = zeros(1, numTests);
CPU_T_ExpRB43 = zeros(1, numTests);
CPU_T_ExpRK45 = zeros(1, numTests);

ExpH4s3_err = zeros(1, numTests);
ExpH3_err   = zeros(1, numTests);
ExpRB42_err = zeros(1, numTests);
ExpRB43_err = zeros(1, numTests);
ExpRK45_err = zeros(1, numTests);

%% ========================================================================
% 5. Run methods
% ========================================================================

for i = 1:numTests

    fprintf('\nTest %d of %d\n', i, numTests);

    % --------------------------------------------------------------------
    % ExpH4s3
    % --------------------------------------------------------------------
    fprintf('  ExpH4s3 with N = %d\n', m1(i));

    tic;
    [t, ExpH4s3_sol] = ExpH4s3( ...
        Nfun, L, JN, t0, tEnd, U0, m1(i), tol, [0.35, 0.8]);
    CPU_T_ExpH4s3(i) = toc;

    ExpH4s3_err(i) = norm(exac_sol - ExpH4s3_sol, inf);

    fprintf('  ExpH4s2 with N = %d\n', m2(i));

    tic;
    [t, ExpH3_sol] = ExpH3( ...
        Nfun, L, JN, t0, tEnd, U0, m2(i), tol, 1/2);
    CPU_T_ExpH3(i) = toc;

    ExpH3_err(i) = norm(exac_sol - ExpH3_sol, inf);

    % --------------------------------------------------------------------
    % ExpRB42
    % --------------------------------------------------------------------
    fprintf('  ExpRB42 with N = %d\n', m3(i));

    tic;
    [t, ExpRB42_sol] = expRB42( ...
        Nfun, L, JN, t0, tEnd, U0, m3(i), tol);
    CPU_T_ExpRB42(i) = toc;

    ExpRB42_err(i) = norm(exac_sol - ExpRB42_sol, inf);

    % --------------------------------------------------------------------
    % ExpRB43
    % --------------------------------------------------------------------
    fprintf('  ExpRB43 with N = %d\n', m4(i));

    tic;
    [t, ExpRB43_sol] = expRB43( ...
        Nfun, L, JN, t0, tEnd, U0, m4(i), tol);
    CPU_T_ExpRB43(i) = toc;

    ExpRB43_err(i) = norm(exac_sol - ExpRB43_sol, inf);

    % --------------------------------------------------------------------
    % ExpRK45
    % --------------------------------------------------------------------
    fprintf('  ExpRK45 with N = %d\n', m5(i));

    tic;
    [t, ExpRK45_sol] = ExpRK45( ...
        Nfun, L, t0, tEnd, U0, m5(i), tol);
    CPU_T_ExpRK45(i) = toc;

    ExpRK45_err(i) = norm(exac_sol - ExpRK45_sol, inf);

end

%% ========================================================================
% 6. Plot CPU time vs accuracy
% ========================================================================

set(0, 'DefaultTextFontSize', 15);
set(0, 'DefaultAxesFontSize', 15);

figure;

semilogy(CPU_T_ExpH4s3, ExpH4s3_err, 'o-', ...
    'LineWidth', 2, 'MarkerSize', 10);
hold on;

semilogy(CPU_T_ExpH3, ExpH3_err, '>-', ...
    'LineWidth', 2, 'MarkerSize', 10);

semilogy(CPU_T_ExpRB43, ExpRB43_err, 'v-', ...
    'LineWidth', 2, 'MarkerSize', 10);

semilogy(CPU_T_ExpRB42, ExpRB42_err, 'd-', ...
    'LineWidth', 2, 'MarkerSize', 10);

semilogy(CPU_T_ExpRK45, ExpRK45_err, 's-', ...
    'LineWidth', 2, 'MarkerSize', 10);

grid on;
axis([min(CPU_T_ExpH3) max(CPU_T_ExpRK45) ExpRB43_err(end) ExpH4s3_err(1)])
legend( ...
    'ExpH4s3', ...
    'ExpH4s2', ...
    'exprb43', ...
    'exprb42', ...
    'expRK4s5', ...
    'Location', 'NorthEast');

title('CPU Time vs Accuracy');
xlabel('CPU Time');
ylabel('Error');

outputFolder = fullfile(script_dir, 'test results');

if ~exist(outputFolder, 'dir')
    mkdir(outputFolder);
end

outputFile = fullfile(outputFolder, 'ExpH_Adr2d_CPU.eps');
print(gcf, outputFile, '-depsc');

%% ========================================================================
% 7. Display error and CPU tables
% ========================================================================

fprintf('\nCPU-time and error table:\n');
fprintf('--------------------------------------------------------------------------------\n');
fprintf('Method        N values                    CPU times                     Errors\n');
fprintf('--------------------------------------------------------------------------------\n');

fprintf('ExpH4s3       ');
fprintf('%5d ', m1);
fprintf('\n              ');
fprintf('%.3e ', CPU_T_ExpH4s3);
fprintf('\n              ');
fprintf('%.3e ', ExpH4s3_err);
fprintf('\n\n');

fprintf('ExpH4s2  ');
fprintf('%5d ', m2);
fprintf('\n              ');
fprintf('%.3e ', CPU_T_ExpH3);
fprintf('\n              ');
fprintf('%.3e ', ExpH3_err);
fprintf('\n\n');

fprintf('ExpRB42       ');
fprintf('%5d ', m3);
fprintf('\n              ');
fprintf('%.3e ', CPU_T_ExpRB42);
fprintf('\n              ');
fprintf('%.3e ', ExpRB42_err);
fprintf('\n\n');

fprintf('ExpRB43       ');
fprintf('%5d ', m4);
fprintf('\n              ');
fprintf('%.3e ', CPU_T_ExpRB43);
fprintf('\n              ');
fprintf('%.3e ', ExpRB43_err);
fprintf('\n\n');

fprintf('ExpRK45       ');
fprintf('%5d ', m5);
fprintf('\n              ');
fprintf('%.3e ', CPU_T_ExpRK45);
fprintf('\n              ');
fprintf('%.3e ', ExpRK45_err);
fprintf('\n');

fprintf('--------------------------------------------------------------------------------\n');

%% ========================================================================
% 8. Speedup table normalized by ExpRK45 = 1
% ========================================================================
%
% Definition:
%
%   speedup(method) = CPU time of ExpRK45 / CPU time of method
%
% Therefore:
%
%   ExpRK45 speedup = 1.
%
% If ExpH4s3 speedup = 2.5, then ExpH4s3 is 2.5 times faster than ExpRK45.

Speedup_ExpH4s3 = CPU_T_ExpRK45 ./ CPU_T_ExpH4s3;
Speedup_ExpH4s2 = CPU_T_ExpRK45 ./ CPU_T_ExpH3;
Speedup_ExpRB43 = CPU_T_ExpRK45 ./ CPU_T_ExpRB43;
Speedup_ExpRB42 = CPU_T_ExpRK45 ./ CPU_T_ExpRB42;
Speedup_ExpRK45 = ones(size(CPU_T_ExpRK45));

%% Display speedup table in command window

fprintf('\nSpeedup table normalized by ExpRK45 = 1:\n');
fprintf('--------------------------------------------------------------------------------\n');
fprintf(' Test     N1     N2     N3     N4     N5     ExpH4s3   ExpH4s2   exprb43   exprb42   rxpRK4s5\n');
fprintf('--------------------------------------------------------------------------------\n');

for i = 1:numTests
    fprintf('%4d   %5d  %5d  %5d  %5d  %5d   %8.3f  %8.3f  %8.3f  %8.3f  %8.3f\n', ...
        i, ...
        m1(i), m2(i), m4(i), m3(i), m5(i), ...
        Speedup_ExpH4s3(i), ...
        Speedup_ExpH4s2(i), ...
        Speedup_ExpRB43(i), ...
        Speedup_ExpRB42(i), ...
        Speedup_ExpRK45(i));
end

fprintf('--------------------------------------------------------------------------------\n');
fprintf(' Avg                                   %8.3f  %8.3f  %8.3f  %8.3f  %8.3f\n', ...
    mean(Speedup_ExpH4s3), ...
    mean(Speedup_ExpH4s2), ...
    mean(Speedup_ExpRB43), ...
    mean(Speedup_ExpRB42), ...
    mean(Speedup_ExpRK45));
fprintf('--------------------------------------------------------------------------------\n');

%% MATLAB table version

SpeedupTable = table( ...
    (1:numTests)', ...
    m1(:), ...
    m2(:), ...
    m4(:), ...
    m3(:), ...
    m5(:), ...
    Speedup_ExpH4s3(:), ...
    Speedup_ExpH4s2(:), ...
    Speedup_ExpRB43(:), ...
    Speedup_ExpRB42(:), ...
    Speedup_ExpRK45(:), ...
    'VariableNames', { ...
        'Test', ...
        'N_ExpH4s3', ...
        'N_ExpH4s2', ...
        'N_ExpRB43', ...
        'N_ExpRB42', ...
        'N_ExpRK45', ...
        'Speedup_ExpH4s3', ...
        'Speedup_ExpH4s2', ...
        'Speedup_ExpRB43', ...
        'Speedup_ExpRB42', ...
        'Speedup_ExpRK45' ...
    });

disp('MATLAB speedup table:')
disp(SpeedupTable)

%% Average speedup table

AverageSpeedupTable = table( ...
    mean(Speedup_ExpH4s3), ...
    mean(Speedup_ExpH4s2), ...
    mean(Speedup_ExpRB43), ...
    mean(Speedup_ExpRB42), ...
    mean(Speedup_ExpRK45), ...
    'VariableNames', { ...
        'Avg_ExpH4s3', ...
        'Avg_ExpH4s2', ...
        'Avg_ExpRB43', ...
        'Avg_ExpRB42', ...
        'Avg_ExpRK45' ...
    });

disp('Average speedup table:')
disp(AverageSpeedupTable)

%% Save tables to CSV files

% Save tables in the "test results" folder
writetable(SpeedupTable, ...
    fullfile(outputFolder, 'ExpH_Adr2d_speedup_table.csv'));

writetable(AverageSpeedupTable, ...
    fullfile(outputFolder, 'ExpH_Adr2d_average_speedup_table.csv'));
rmpath('../integrators','../phipmsimuliom');