%% ========================================================================
% Brusselator 3D: fourth-order convergence experiment
% ========================================================================

clear;
clc;
close all;

%% Paths
script_dir  = fileparts(mfilename('fullpath'));
project_dir = fileparts(script_dir);

addpath('../phipmsimuliom')
addpath('../KronPACK/src')

dataFolder = fullfile(script_dir, 'Data');
outputFolder = fullfile(script_dir, 'Test Results');

if ~exist(outputFolder, 'dir')
    mkdir(outputFolder);
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

Nsteps = [10, 20, 40, 80, 160];

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
    x{mu} = linspace(a(mu), b(mu), n(mu));
    h = (b(mu)-a(mu))/(n(mu)-1);

    D2{mu} = spdiags( ...
        ones(n(mu),1)*([1,-2,1]/h^2), -1:1, n(mu), n(mu));
    D2{mu}(1,1:2) = [-2,2]/h^2;
    D2{mu}(end,end-1:end) = [2,-2]/h^2;

    D1{mu} = spdiags( ...
        ones(n(mu),1)*([-1,0,1]/(2*h)), -1:1, n(mu), n(mu));
    D1{mu}(1,:) = 0;
    D1{mu}(end,:) = 0;

    A_sp{1}{mu} = -alphau*D1{mu} + deltau*D2{mu};
    A_sp{2}{mu} = -alphav*D1{mu} + deltav*D2{mu};
end

K = cell(2,1);
K{1} = kronsum(A_sp{1});
K{2} = kronsum(A_sp{2});

pn = prod(n);

%% ========================================================================
% 3. Initial value, vector field, and Jacobians
% ========================================================================
X = cell(1,d);
[X{:}] = ndgrid(x{:});

U0 = cell(2,1);
U0{1} = 1 + cos(2*pi*X{1}).*cos(2*pi*X{2}).*cos(2*pi*X{3});
U0{2} = 3*ones(n);
Y0 = [U0{1}(:); U0{2}(:)];

g1 = @(t,u,v) -(a1u+1)*u + a2u + (u.*u).*v;
g2 = @(t,u,v) a1u*u - (u.*u).*v;

dg11 = @(t,u,v) -(a1u+1) + 2*(u.*v);
dg12 = @(t,u,v) u.*u;
dg21 = @(t,u,v) a1u - 2*(u.*v);
dg22 = @(t,u,v) -u.*u;

Kfun = @(Y) [K{1}*Y(1:pn); K{2}*Y(pn+1:2*pn)];

Nfun = @(t,Y) [ ...
    g1(t,Y(1:pn),Y(pn+1:2*pn)); ...
    g2(t,Y(1:pn),Y(pn+1:2*pn))];

Ffun = @(t,Y) Kfun(Y) + Nfun(t,Y);

Jfun = @(t,Y) [ ...
    K{1} + spdiags(dg11(t,Y(1:pn),Y(pn+1:2*pn)),0,pn,pn), ...
           spdiags(dg12(t,Y(1:pn),Y(pn+1:2*pn)),0,pn,pn); ...
           spdiags(dg21(t,Y(1:pn),Y(pn+1:2*pn)),0,pn,pn), ...
    K{2} + spdiags(dg22(t,Y(1:pn),Y(pn+1:2*pn)),0,pn,pn)];

JN = @(t,Y) [ ...
    spdiags(dg11(t,Y(1:pn),Y(pn+1:2*pn)),0,pn,pn), ...
    spdiags(dg12(t,Y(1:pn),Y(pn+1:2*pn)),0,pn,pn); ...
    spdiags(dg21(t,Y(1:pn),Y(pn+1:2*pn)),0,pn,pn), ...
    spdiags(dg22(t,Y(1:pn),Y(pn+1:2*pn)),0,pn,pn)];

%% ========================================================================
% 4. Load extrapolated reference solution
% ========================================================================
referenceFile = fullfile(dataFolder, 'brusselator_3D_Uref.mat');

if ~isfile(referenceFile)
    error(['Reference solution was not found. Run the reference script ', ...
           'with method = ''ref_sol'' first.\nExpected file:\n%s'], ...
           referenceFile);
end

S = load(referenceFile, 'Uref', 'T', 'n');

if S.T ~= T || ~isequal(S.n,n)
    error('The reference solution uses a different final time or grid.');
end

Urefvec = [S.Uref{1}(:); S.Uref{2}(:)];

%% ========================================================================
% 5. Run both methods on the common refinement sequence
% ========================================================================
ExpRB42_err = zeros(size(Nsteps));
ExpH4s2_err = zeros(size(Nsteps));
CPU_T_ExpRB42 = zeros(size(Nsteps));
CPU_T_ExpH4s2 = zeros(size(Nsteps));

for i = 1:numel(Nsteps)
    nsteps = Nsteps(i);
    fprintf('\nN = %d\n',nsteps);

    tic;
    [~,Yfinal] = ExpRB42_local( ...
        Ffun,Jfun,0,T,Y0,nsteps,tol_phipm_simul_iom);
    CPU_T_ExpRB42(i) = toc;
    ExpRB42_err(i) = norm(Yfinal-Urefvec,inf);

    fprintf('ExpRB42:  error = %.3e, CPU = %.2f s\n', ...
        ExpRB42_err(i),CPU_T_ExpRB42(i));

    tic;
    [~,Yfinal] = ExpH4s2_local( ...
        Nfun,Kfun,JN,0,T,Y0,nsteps, ...
        tol_phipm_simul_iom,0.5);
    CPU_T_ExpH4s2(i) = toc;
    ExpH4s2_err(i) = norm(Yfinal-Urefvec,inf);

    fprintf('ExpH4s2:  error = %.3e, CPU = %.2f s\n', ...
        ExpH4s2_err(i),CPU_T_ExpH4s2(i));
end

%% ========================================================================
% 6. Observed convergence orders
% ========================================================================
ExpRB42_order = log(ExpRB42_err(1:end-1)./ExpRB42_err(2:end))/log(2);
ExpH4s2_order = log(ExpH4s2_err(1:end-1)./ExpH4s2_err(2:end))/log(2);

fprintf('\nExpRB42 pairwise orders: ');
fprintf('%.3f ',ExpRB42_order);
fprintf('\nExpH4s2 pairwise orders: ');
fprintf('%.3f ',ExpH4s2_order);
fprintf('\n');

%% ========================================================================
% 7. Construct and save the CSV table
% ========================================================================
ExpRB42_order_at_N = [NaN,ExpRB42_order]';
ExpH4s2_order_at_N = [NaN,ExpH4s2_order]';

ResultTable = table( ...
    Nsteps(:), ...
    ExpRB42_err(:), ...
    CPU_T_ExpRB42(:), ...
    ExpRB42_order_at_N, ...
    ExpH4s2_err(:), ...
    CPU_T_ExpH4s2(:), ...
    ExpH4s2_order_at_N, ...
    'VariableNames', { ...
        'N', ...
        'ExpRB42_Error', ...
        'ExpRB42_CPU_sec', ...
        'ExpRB42_Order', ...
        'ExpH4s2_Error', ...
        'ExpH4s2_CPU_sec', ...
        'ExpH4s2_Order'});

disp(ResultTable);

writetable(ResultTable,fullfile(outputFolder, ...
    'Brusselator3D_Order_results.csv'));

%% ========================================================================
% 8. Order plot
% ========================================================================
set(0,'DefaultTextFontSize',15);
set(0,'DefaultAxesFontSize',15);

color_ExpH4s2 = [0.8500 0.3250 0.0980];
color_ExpRB42 = [0.4940 0.1840 0.5560];
color_Order4  = [0.9290 0.6940 0.1250];

hPlot = T./Nsteps;
figOrder = figure;

loglog(hPlot,ExpH4s2_err,'>-', ...
    'Color',color_ExpH4s2, ...
    'MarkerEdgeColor',color_ExpH4s2, ...
    'MarkerFaceColor','none', ...
    'LineWidth',2, ...
    'MarkerSize',10);
hold on;

loglog(hPlot,ExpRB42_err,'d-', ...
    'Color',color_ExpRB42, ...
    'MarkerEdgeColor',color_ExpRB42, ...
    'MarkerFaceColor','none', ...
    'LineWidth',2, ...
    'MarkerSize',10);

ref4 = ExpH4s2_err(1)*(hPlot/hPlot(1)).^4;
loglog(hPlot,ref4,'--', ...
    'Color',color_Order4, ...
    'LineWidth',2);

grid on;
ax = gca;
ax.XTick = sort(hPlot);
ax.XTickLabel = string(fliplr(Nsteps));
ax.TickLength = 2*ax.TickLength;
ax.OuterPosition = [0 0 1 1];

xlim([min(hPlot),max(hPlot)]);
allErrors = [ExpRB42_err,ExpH4s2_err,ref4];
ylim([0.5*min(allErrors),2*max(allErrors)]);

legend('ExpH4s2','exprb42','Order 4','Location','SouthEast');
title('Order Plot');
xlabel('Number of time steps');
ylabel('Error');

print(figOrder,fullfile(outputFolder, ...
    'Brusselator3D_Order.eps'),'-depsc');

%% ========================================================================
% 9. Save MATLAB data
% ========================================================================
save(fullfile(dataFolder,'Brusselator3D_Order_data.mat'), ...
    'Nsteps','ExpRB42_err','CPU_T_ExpRB42','ExpRB42_order', ...
    'ExpH4s2_err','CPU_T_ExpH4s2','ExpH4s2_order', ...
    'ResultTable','T','n','tol_phipm_simul_iom');

rmpath('../phipmsimuliom')
rmpath('../KronPACK/src')

%% ========================================================================
% Local ExpH4s2 implementation
% ========================================================================
function [t,uFinal] = ExpH4s2_local(Nfun,L,JN,t0,tEnd,u0,nSteps,tol,c2)
    h = (tEnd-t0)/nSteps;
    t = linspace(t0,tEnd,nSteps+1);
    u = u0(:);
    zeroVec = zeros(size(u));

    for k = 1:nSteps
        Nu = Nfun(t(k),u);
        Fn = L(u) + Nu;
        Gn = JN(t(k),u)*Fn;

        stageIncrement = phipm_simul_iom( ...
            c2*h, L, [zeroVec,Fn,Gn], tol, 1, 2);
        uStage = u + stageIncrement;

        FStage = L(uStage) + Nfun(t(k)+c2*h,uStage);
        Hn2 = JN(t(k)+c2*h,uStage)*FStage - Gn;

        finalIncrement = phipm_simul_iom( ...
            h, L, [zeroVec,Fn,Gn,Hn2/(c2*h)], tol, 1, 2);
        u = u + finalIncrement;
    end

    uFinal = u;
end

%% ========================================================================
% Local ExpRB42 implementation
% ========================================================================
function [tgrid,yFinal] = ExpRB42_local(Ffun,Jfun,t0,tEnd,y0,nsteps,tol)
    h = (tEnd-t0)/nsteps;
    tgrid = linspace(t0,tEnd,nsteps+1);
    y = y0(:);
    zeroVec = zeros(size(y));

    for k = 1:nsteps
        tn = tgrid(k);
        Fn = Ffun(tn,y);
        Jn = Jfun(tn,y);

        alpha = (3/4)*h;
        incStage = alpha*phipm_simul_iom( ...
            1, alpha*Jn, [zeroVec,Fn], tol, 1, 2);
        yStage = y + incStage;

        FStage = Ffun(tn+alpha,yStage);
        Dn2 = FStage-Fn-Jn*(yStage-y);

        incFinal = h*phipm_simul_iom( ...
            1, h*Jn, [zeroVec,Fn,zeroVec,(32/9)*Dn2], tol, 1, 2);
        y = y + incFinal;
    end

    yFinal = y;
end

