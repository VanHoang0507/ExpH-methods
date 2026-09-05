function [t, uFinal, U] = ExpRK45(Nfun, L, t0, tEnd, u0, nSteps, tol)
%EXPRK45 Five-stage exponential Runge--Kutta method.
%
%   Solves the autonomous semilinear problem
%
%       u'(t) = L*u(t) + N(u(t)),     u(t0) = u0.
%
%   Define the full vector field
%
%       F(u) = L*u + N(u).
%
%   This method uses nonlinear differences
%
%       D_i = N(U_i) - N(u_n),
%
%   not Hermite derivative data.
%
%   Inputs:
%       Nfun    - nonlinear function handle, Nfun(u)
%       L       - linear matrix/operator
%       t0      - initial time
%       tEnd    - final time
%       u0      - initial condition vector
%       nSteps  - number of time steps
%       tol     - tolerance for phipm_simul_iom
%
%   Outputs:
%       t       - time grid
%       uFinal  - numerical solution at tEnd
%       U       - optional full numerical trajectory

    if nargin < 7 || isempty(tol)
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
        % Data at u_n
        % ---------------------------------------------------------------
        N0 = Nfun(u);

        % Full vector field:
        %
        % F_n = L*u_n + N(u_n)
        Fn = L*u + N0;

        % ---------------------------------------------------------------
        % Stage U_{n2}
        %
        % U_{n2} = u_n + 1/2*h*phi_1(1/2*h*L)F_n.
        % ---------------------------------------------------------------
        Un2 = u + 0.5*h*phipm_simul_iom( ...
            1, 0.5*h*L, [zeroVec, Fn], tol, 1, 2);

        Dn2 = Nfun(Un2) - N0;

        % ---------------------------------------------------------------
        % Stage U_{n3}
        % ---------------------------------------------------------------
        Un3 = Un2 + h*phipm_simul_iom( ...
            1, 0.5*h*L, [zeroVec, zeroVec, Dn2], tol, 1, 2);

        Dn3 = Nfun(Un3) - N0;

        % ---------------------------------------------------------------
        % Stage U_{n4}
        % ---------------------------------------------------------------
        Un4 = u + h*phipm_simul_iom( ...
            1, h*L, [zeroVec, Fn, Dn2 + Dn3], tol, 1, 2);

        Dn4 = Nfun(Un4) - N0;

        % ---------------------------------------------------------------
        % Stage U_{n5}
        % ---------------------------------------------------------------
        V2 = 0.25*(2*Dn2 + 2*Dn3 - Dn4);
        V3 = 0.5*(-Dn2 - Dn3 + Dn4);

        W2 = 0.25*(Dn2 + Dn3 - Dn4);
        W3 = -Dn2 - Dn3 + Dn4;

        Un5 = u ...
            + h*phipm_simul_iom( ...
                1, 0.5*h*L, [zeroVec, 0.5*Fn, V2, V3], tol, 1, 2) ...
            + h*phipm_simul_iom( ...
                1, h*L, [zeroVec, zeroVec, W2, W3], tol, 1, 2);

        Dn5 = Nfun(Un5) - N0;

        % ---------------------------------------------------------------
        % Final update
        % ---------------------------------------------------------------
        B2 = -Dn4 + 4*Dn5;
        B3 =  4*Dn4 - 8*Dn5;

        u = u + h*phipm_simul_iom( ...
            1, h*L, [zeroVec, Fn, B2, B3], tol, 1, 2);

        if saveTrajectory
            U(:, k + 1) = u;
        end
    end

    uFinal = u;
end