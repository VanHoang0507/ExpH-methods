function [t, uFinal, U] = ExpH2(Nfun, L, JN, t0, tEnd, u0, nSteps, tol)
%EXPH2 Second-order exponential Hermite method.
%   Solves the autonomous semilinear problem
%
%       u'(t) = L*u(t) + N(u(t)),     u(t0) = u0.
%
%   The method is
%
%       F_n      = L*u_n + N(u_n),
%       dNFn     = N'(u_n) F_n,
%
%       u_{n+1} = u_n + h*phi_1(hL) F_n + h^2*phi_2(hL) dNFn.
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
%
%   Outputs:
%       t       - time grid
%       uFinal  - numerical solution at tEnd
%       U       - optional full solution history

    if nargin < 8 || isempty(tol)
        tol = 1e-10;
    end

    if nSteps <= 0 || floor(nSteps) ~= nSteps
        error('nSteps must be a positive integer.');
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
        Nu = Nfun(u);

        % Full vector field:
        % F_n = L*u_n + N(u_n)
        Fn = L*u + Nu;

        % Hermite derivative data:
        % d/dt N(u(t_n)) = N'(u_n)F_n
        dNFn = JN(u) * Fn;

        % phipm_simul_iom computes
        % h*phi_1(hL)Fn + h^2*phi_2(hL)dNFn
        rhs = [zeroVec, Fn, dNFn];

        u = u + phipm_simul_iom(h, L, rhs, tol, 1, 2);

        if saveSolution
            U(:, k + 1) = u;
        end
    end

    uFinal = u;
end