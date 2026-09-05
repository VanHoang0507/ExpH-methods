function [t, uFinal, U] = ExpRB42(Nfun, L, JN, t0, tEnd, u0, nSteps, tol)
%EXPRB42 Fourth-order two-stage exponential Rosenbrock method.
%
%   Solves the autonomous semilinear problem
%
%       u'(t) = L*u(t) + N(u(t)),     u(t0) = u0.
%
%   At each time step, the Rosenbrock linearization is
%
%       J_n = L + N'(u_n),
%
%   and the nonlinear Rosenbrock remainder is
%
%       g_n(u) = N(u) - N'(u_n)u.
%
%   The full vector field is
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

        % Full vector field:
        %
        % F_n = L*u_n + N(u_n)
        Fn = L*u + Nu;

        % Rosenbrock Jacobian:
        %
        % J_n = L + N'(u_n)
        Jstep = L + JNu;

        % Rosenbrock remainder at u_n:
        %
        % g_n(u_n) = N(u_n) - N'(u_n)u_n
        gn_u = Nu - JNu*u;

        % ---------------------------------------------------------------
        % Internal stage:
        %
        % U_{n2} = u_n + (3/4)h phi_1((3/4)hJ_n)F_n.
        % ---------------------------------------------------------------

        Un2 = u + (3/4)*h*phipm_simul_iom( ...
            1, (3/4)*h*Jstep, [zeroVec, Fn], tol, 1, 2);

        % Rosenbrock nonlinear difference:
        %
        % D_{n2} = g_n(U_{n2}) - g_n(u_n).
        Dn2 = Nfun(Un2) - JNu*Un2 - gn_u;

        % ---------------------------------------------------------------
        % Final update:
        %
        % u_{n+1} = u_n
        %          + h phi_1(hJ_n)F_n
        %          + h phi_3(hJ_n)(32/9)D_{n2}.
        % ---------------------------------------------------------------

        u = u + h*phipm_simul_iom( ...
            1, h*Jstep, [zeroVec, Fn, zeroVec, (32/9)*Dn2], tol, 1, 2);

        if saveTrajectory
            U(:, k + 1) = u;
        end
    end

    uFinal = u;
end