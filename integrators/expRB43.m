function [t, uFinal, U] = ExpRB43(Nfun, L, JN, t0, tEnd, u0, nSteps, tol)
%EXPRB43 Fourth-order exponential Rosenbrock method.
%
%   Solves the autonomous semilinear problem
%
%       u'(t) = L*u(t) + N(u(t)),     u(t0) = u0.
%
%   At each step, the method uses the local Rosenbrock linearization
%
%       J_n = L + N'(u_n),
%
%   and the nonlinear remainder
%
%       g_n(u) = N(u) - N'(u_n)u.
%
%   The full right-hand side is
%
%       F_n = L*u_n + N(u_n).
%
%   Inputs:
%       Nfun    - nonlinear function handle, Nfun(u)
%       L       - linear matrix/operator
%       JN      - Jacobian of N, JN(u)
%       t0      - initial time
%       tEnd    - final time
%       u0      - initial condition
%       nSteps  - number of time steps
%       tol     - tolerance for phipm_simul_iom
%
%   Outputs:
%       t       - time grid
%       uFinal  - numerical solution at tEnd
%       U       - optional full numerical trajectory

    if nargin < 8 || isempty(tol)
        tol = 1e-10;
    end

    if nSteps <= 0 || floor(nSteps) ~= nSteps
        error('nSteps must be a positive integer.');
    end

    h = (tEnd - t0) / nSteps;
    t = linspace(t0, tEnd, nSteps + 1);

    u = u0(:);
    m = length(u);
    zeroVec = zeros(m, 1);

    saveTrajectory = (nargout >= 3);

    if saveTrajectory
        U = zeros(m, nSteps + 1);
        U(:, 1) = u;
    end

    for k = 1:nSteps

        % ---------------------------------------------------------------
        % Step data at u_n
        % ---------------------------------------------------------------

        Nu = Nfun(u);
        JNu = JN(u);

        % Full right-hand side:
        %
        % F_n = L*u_n + N(u_n)
        Fn = L*u + Nu;

        % Rosenbrock Jacobian:
        %
        % J_n = L + N'(u_n)
        Jstep = L + JNu;

        % Nonlinear Rosenbrock remainder at u_n:
        %
        % g_n(u_n) = N(u_n) - N'(u_n)u_n
        gn_u = Nu - JNu*u;

        % ---------------------------------------------------------------
        % Internal stages:
        %
        % U_{n2} = u_n + (h/2)*phi_1((h/2)J_n)F_n,
        %
        % U_{n3} = u_n + h*phi_1(hJ_n)F_n.
        % ---------------------------------------------------------------

        stageIncrements = phipm_simul_iom( ...
            [1/2, 1]*h, Jstep, [zeroVec, Fn], tol, 1, 2);

        Un2 = u + stageIncrements(:, 1);
        Un3 = u + stageIncrements(:, 2);

        % ---------------------------------------------------------------
        % Rosenbrock nonlinear differences:
        %
        % D_{ni} = g_n(U_{ni}) - g_n(u_n),
        %
        % where
        %
        % g_n(v) = N(v) - N'(u_n)v.
        % ---------------------------------------------------------------

        Dn2 = Nfun(Un2) - JNu*Un2 - gn_u;
        Dn3 = Nfun(Un3) - JNu*Un3 - gn_u;

        % ---------------------------------------------------------------
        % Final update:
        %
        % u_{n+1} = u_n
        %          + h*phi_1(hJ_n)F_n
        %          + h*phi_3(hJ_n)(16D_{n2} - 2D_{n3})
        %          + h*phi_4(hJ_n)(-48D_{n2} + 12D_{n3}).
        % ---------------------------------------------------------------

        B3 =  16*Dn2 -  2*Dn3;
        B4 = -48*Dn2 + 12*Dn3;

        finalIncrement = h*phipm_simul_iom( ...
            1, h*Jstep, [zeroVec, Fn, zeroVec, B3, B4], tol, 1, 2);

        u = u + finalIncrement;

        if saveTrajectory
            U(:, k + 1) = u;
        end
    end

    uFinal = u;
end