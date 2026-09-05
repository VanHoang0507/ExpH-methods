function [t, uFinal, U] = ExpH5s4(Nfun, L, JN, t0, tEnd, u0, nSteps, tol, c2, c3)
%EXPH5S4 Fifth-order four-stage exponential Hermite method.
%
%   Solves the autonomous semilinear problem
%
%       u'(t) = L*u(t) + N(u(t)),     u(t0) = u0.
%
%   Define the full vector field
%
%       F(u) = L*u + N(u).
%
%   The method uses the Hermite derivative
%
%       G_n = N'(u_n)F(u_n).
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
%       c2      - second stage node
%       c3      - third stage node
%
%   Outputs:
%       t       - time grid
%       uFinal  - numerical solution at tEnd
%       U       - optional full numerical trajectory

    if nargin < 8 || isempty(tol)
        tol = 1e-10;
    end

    if nargin < 9 || isempty(c2)
        c2 = 1/2;
    end

    if nargin < 10 || isempty(c3)
        c3 = 3/4;
    end

    if nSteps <= 0 || floor(nSteps) ~= nSteps
        error('nSteps must be a positive integer.');
    end

    if c2 <= 0 || c3 <= 0
        error('Stage nodes c2 and c3 must be positive.');
    end

    if abs(2*c3 - 1) < 100*eps
        error('c3 cannot be 1/2 because c4 becomes singular.');
    end

    c4 = (5*c3 - 3) / (10*c3 - 5);

    if c4 <= 0
        error('The computed stage node c4 must be positive.');
    end

    if abs(c3 - c4) < 100*eps
        error('Invalid stage nodes: c3 and c4 must be different.');
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

        % Full vector field:
        %
        % F_n = L*u_n + N(u_n)
        Fn = L*u + Nu;

        % Hermite derivative:
        %
        % G_n = N'(u_n)F_n
        Gn = JN(u) * Fn;

        % ---------------------------------------------------------------
        % Stage U_{n2}
        %
        % U_{n2} = u_n
        %        + c2*h*phi_1(c2*h*L)F_n
        %        + c2^2*h^2*phi_2(c2*h*L)G_n.
        % ---------------------------------------------------------------

        inc2 = phipm_simul_iom( ...
            c2*h, L, [zeroVec, Fn, Gn], tol, 1, 2);

        Un2 = u + inc2;

        F2 = L*Un2 + Nfun(Un2);

        Hn2 = JN(Un2)*F2 - Gn;

        % ---------------------------------------------------------------
        % Stage U_{n3}
        %
        % U_{n3} = u_n
        %        + c3*h*phi_1(c3*h*L)F_n
        %        + c3^2*h^2*phi_2(c3*h*L)G_n
        %        + (c3^3/c2)*h^2*phi_3(c3*h*L)H_{n2}.
        %
        % Since phipm_simul_iom(s,L,[0,0,0,V]) gives
        %
        %        s^3*phi_3(sL)V,
        %
        % we use V = H_{n2}/(c2*h).
        % ---------------------------------------------------------------

        inc3 = phipm_simul_iom( ...
            c3*h, L, [zeroVec, Fn, Gn, Hn2/(c2*h)], tol, 1, 2);

        Un3 = u + inc3;

        F3 = L*Un3 + Nfun(Un3);

        Hn3 = JN(Un3)*F3 - Gn;

        % ---------------------------------------------------------------
        % Stage U_{n4}
        %
        % U_{n4} = u_n
        %        + c4*h*phi_1(c4*h*L)F_n
        %        + c4^2*h^2*phi_2(c4*h*L)G_n
        %        + (c4^3/c2)*h^2*phi_3(c4*h*L)H_{n2}.
        % ---------------------------------------------------------------

        inc4 = phipm_simul_iom( ...
            c4*h, L, [zeroVec, Fn, Gn, Hn2/(c2*h)], tol, 1, 2);

        Un4 = u + inc4;

        F4 = L*Un4 + Nfun(Un4);

        Hn4 = JN(Un4)*F4 - Gn;

        % ---------------------------------------------------------------
        % Final Hermite coefficients
        % ---------------------------------------------------------------

        denom3 = c3*c4 - c3^2;
        denom4 = c4*c3 - c4^2;

        B3 =  c4*Hn3/denom3 + c3*Hn4/denom4;
        B4 = -2*Hn3/denom3 - 2*Hn4/denom4;

        % ---------------------------------------------------------------
        % Final update:
        %
        % u_{n+1} = u_n
        %          + h*phi_1(hL)F_n
        %          + h^2*phi_2(hL)G_n
        %          + h^2*phi_3(hL)B3
        %          + h^2*phi_4(hL)B4.
        %
        % phipm_simul_iom(h,L,[0,Fn,Gn,V3,V4]) gives
        %
        %        h*phi_1(hL)F_n
        %      + h^2*phi_2(hL)G_n
        %      + h^3*phi_3(hL)V3
        %      + h^4*phi_4(hL)V4.
        %
        % Therefore
        %
        %        V3 = B3/h,
        %        V4 = B4/h^2.
        % ---------------------------------------------------------------

        finalIncrement = phipm_simul_iom( ...
            h, L, [zeroVec, Fn, Gn, B3/h, B4/h^2], tol, 1, 2);

        u = u + finalIncrement;

        if saveTrajectory
            U(:, k + 1) = u;
        end
    end

    uFinal = u;
end