function [t, uFinal, U] = ExpH4s3(Nfun, L, JN, t0, tEnd, u0, nSteps, tol, c)
%EXPH4S3 Fourth-order three-stage exponential Hermite method.
%
%   Solves the autonomous semilinear problem
%
%       u'(t) = L*u(t) + N(u(t)),     u(t0) = u0.
%
%   Define the full vector field
%
%       F(u) = L*u + N(u).
%
%   The method uses
%
%       G_n = N'(u_n)F(u_n),
%
%   and the stage differences
%
%       H_{n2} = N'(U_{n2})F(U_{n2}) - G_n,
%       H_{n3} = N'(U_{n3})F(U_{n3}) - G_n.
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
%       c       - stage nodes [c2, c3], default c = [1/2, 1]
%
%   Outputs:
%       t       - time grid
%       uFinal  - numerical solution at tEnd
%       U       - optional full solution history

    if nargin < 8 || isempty(tol)
        tol = 1e-10;
    end

    if nargin < 9 || isempty(c)
        c = [1/2, 1];
    end

    if numel(c) ~= 2
        error('c must be a vector with two entries: c = [c2, c3].');
    end

    c2 = c(1);
    c3 = c(2);

    if nSteps <= 0 || floor(nSteps) ~= nSteps
        error('nSteps must be a positive integer.');
    end

    if c2 <= 0 || c3 <= 0 || abs(c2 - c3) < eps
        error('The stage nodes must satisfy c2 > 0, c3 > 0, and c2 ~= c3.');
    end

    h = (tEnd - t0) / nSteps;
    t = linspace(t0, tEnd, nSteps + 1);

    u = u0(:);
    m = length(u);
    zeroVec = zeros(m, 1);

    saveSolution = (nargout >= 3);

    if saveSolution
        U = zeros(m, nSteps + 1);
        U(:, 1) = u;
    end

    for k = 1:nSteps

        % Full vector field:
        % F_n = L*u_n + N(u_n)
        Nu = Nfun(u);
        Fn = L*u + Nu;

        % Hermite nonlinear derivative:
        % G_n = N'(u_n)F_n
        Gn = JN(u) * Fn;

        % ---------------------------------------------------------------
        % Stage values:
        %
        % U_{n2} = u_n + c2*h*phi_1(c2*h*L)F_n + c2^2*h^2*phi_2(c2*h*L)G_n,
        %
        % U_{n3} = u_n + c3*h*phi_1(c3*h*L)F_n + c3^2*h^2*phi_2(c3*h*L)G_n.
        % ---------------------------------------------------------------
        stageIncrements = phipm_simul_iom( ...
            [c2, c3]*h, L, [zeroVec, Fn, Gn], tol, 1, 2);

        Un2 = u + stageIncrements(:, 1);
        Un3 = u + stageIncrements(:, 2);

        % Full vector fields at the stage values
        N2 = Nfun(Un2);
        F2 = L*Un2 + N2;

        N3 = Nfun(Un3);
        F3 = L*Un3 + N3;

        % Stage Hermite differences:
        %
        % H_{n2} = N'(U_{n2})F(U_{n2}) - N'(u_n)F(u_n)
        % H_{n3} = N'(U_{n3})F(U_{n3}) - N'(u_n)F(u_n)
        Hn2 = JN(Un2)*F2 - Gn;
        Hn3 = JN(Un3)*F3 - Gn;

        % Coefficients for the phi_3 and phi_4 correction terms
        denom3 = c3^2 - c2*c3;
        denom2 = c2^2 - c2*c3;

        B3 = -c2*Hn3/denom3 - c3*Hn2/denom2;
        B4 =  2*Hn3/denom3 + 2*Hn2/denom2;

        % ---------------------------------------------------------------
        % Final update in phi-form:
        %
        % u_{n+1} = u_n
        %          + h*phi_1(hL)F_n
        %          + h^2*phi_2(hL)G_n
        %          + h^2*phi_3(hL)B3
        %          + h^2*phi_4(hL)B4.
        %
        % Since phipm_simul_iom(h,L,[0,Fn,Gn,V3,V4]) computes
        %
        %     h*phi_1(hL)Fn
        %   + h^2*phi_2(hL)Gn
        %   + h^3*phi_3(hL)V3
        %   + h^4*phi_4(hL)V4,
        %
        % we pass
        %
        %     V3 = B3/h,
        %     V4 = B4/h^2.
        % ---------------------------------------------------------------
        finalIncrement = phipm_simul_iom( ...
            h, L, [zeroVec, Fn, Gn, B3/h, B4/h^2], tol, 1, 2);

        u = u + finalIncrement;

        if saveSolution
            U(:, k + 1) = u;
        end
    end

    uFinal = u;
end