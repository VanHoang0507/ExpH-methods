function [t, uFinal, U] = ExpH3(Nfun, L, JN, t0, tEnd, u0, nSteps, tol, c2)
%EXPH3 Third-order exponential Hermite method.
%
%   Solves
%
%       u'(t) = L*u(t) + N(u(t)),     u(t0) = u0.
%
%   Define the full vector field
%
%       F(u) = L*u + N(u).
%
%   The ExpH3 scheme is
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

    if nSteps <= 0 || floor(nSteps) ~= nSteps
        error('nSteps must be a positive integer.');
    end

    if c2 <= 0 || c2 > 1
        error('c2 must satisfy 0 < c2 <= 1.');
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
        Nu = Nfun(u);
        Fn = L*u + Nu;

        % Nonlinear Hermite derivative:
        %
        % G_n = N'(u_n)F_n
        Gn = JN(u) * Fn;

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
        NStage = Nfun(uStage);
        FStage = L*uStage + NStage;

        % Difference term:
        %
        % H_{n2} = N'(U_{n2})F(U_{n2}) - N'(u_n)F(u_n)
        Hn2 = JN(uStage)*FStage - Gn;

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