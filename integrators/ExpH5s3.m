function [t, uFinal, U] = ExpH5s3(Nfun, L, JN, t0, tEnd, u0, nSteps, tol, c2)
%EXPH5S3 Fifth-order three-stage exponential Hermite method.
%
%   Solves the autonomous semilinear problem
%
%       u'(t) = L*u(t) + N(u(t)),     u(t0) = u0.
%
%   Define the full vector field
%
%       F(u) = L*u + N(u).
%
%   The method uses the nonlinear Hermite derivative
%
%       G_n = N'(u_n)F(u_n).
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
%       c2      - second stage node
%
%   Outputs:
%       t       - time grid
%       uFinal  - numerical solution at tEnd
%       U       - optional full solution history

    if nargin < 8 || isempty(tol)
        tol = 1e-10;
    end

    if nargin < 9 || isempty(c2)
        c2 = 1;
    end

    if nSteps <= 0 || floor(nSteps) ~= nSteps
        error('nSteps must be a positive integer.');
    end

    if c2 <= 0
        error('c2 must be positive.');
    end

    if abs(2*c2 - 1) < 100*eps
        error('c2 cannot be 1/2 because c3 is singular.');
    end

    % Third stage node and coefficients
    c3 = (5*c2 - 3) / (10*c2 - 5);

    if c3 <= 0 || abs(c3 - c2) < 100*eps
        error('Invalid stage nodes: require c3 > 0 and c3 ~= c2.');
    end

    b20 = (c3/6 - 1/12) / (c2*(c3 - c2));
    b30 = (c2/6 - 1/12) / (c3*(c2 - c3));

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

        % ---------------------------------------------------------------
        % Data at u_n
        % ---------------------------------------------------------------
        Nu = Nfun(u);

        % Full vector field:
        %
        % F_n = L*u_n + N(u_n)
        Fn = L*u + Nu;

        % Nonlinear Hermite derivative:
        %
        % G_n = N'(u_n)F_n
        Gn = JN(u) * Fn;

        % ---------------------------------------------------------------
        % Base stage increments:
        %
        % U_{n2}^{base} = c2*h*phi_1(c2*h*L)F_n + c2^2*h^2*phi_2(c2*h*L)G_n,
        %
        % U_{n3}^{base} = c3*h*phi_1(c3*h*L)F_n + c3^2*h^2*phi_2(c3*h*L)G_n.
        % ---------------------------------------------------------------
        stageBase = phipm_simul_iom( ...
            [c2, c3]*h, L, [zeroVec, Fn, Gn], tol, 1, 2);

        Un2 = u + stageBase(:, 1);

        % ---------------------------------------------------------------
        % H_{n2}
        % ---------------------------------------------------------------
        N2 = Nfun(Un2);
        F2 = L*Un2 + N2;

        Hn2 = JN(Un2)*F2 - Gn;

        % ---------------------------------------------------------------
        % Third stage correction
        %
        % The old code used terms of the form
        %
        %   h^2 * c3^3/c2 * phi_3(c3*h*L)H_{n2}
        %
        % and
        %
        %   h^2 * c2^2*(b20/b30)*phi_3(c2*h*L)H_{n2}.
        %
        % Since phipm_simul_iom(s,L,[0,0,0,V]) gives
        %
        %   s^3*phi_3(sL)V,
        %
        % the vectors must be scaled by 1/h.
        % ---------------------------------------------------------------
        corr31 = phipm_simul_iom( ...
            c3*h, L, [zeroVec, zeroVec, zeroVec, Hn2/(c2*h)], tol, 1, 2);

        corr32 = phipm_simul_iom( ...
            c2*h, L, [zeroVec, zeroVec, zeroVec, (b20/b30)*Hn2/(c2*h)], tol, 1, 2);

        Un3 = u + stageBase(:, 2) + corr31 + corr32;

        % ---------------------------------------------------------------
        % H_{n3}
        % ---------------------------------------------------------------
        N3 = Nfun(Un3);
        F3 = L*Un3 + N3;

        Hn3 = JN(Un3)*F3 - Gn;

        % ---------------------------------------------------------------
        % Final correction coefficients
        % ---------------------------------------------------------------
        denom2 = c2*c3 - c2^2;
        denom3 = c2*c3 - c3^2;

        B3 =  c3*Hn2/denom2 + c2*Hn3/denom3;
        B4 = -2*Hn2/denom2 - 2*Hn3/denom3;

        % ---------------------------------------------------------------
        % Final update:
        %
        % u_{n+1} = u_n + h*phi_1(hL)F_n + h^2*phi_2(hL)G_n
        %          + h^2*phi_3(hL)B3 + h^2*phi_4(hL)B4.
        %
        % Since phipm_simul_iom(h,L,[0,Fn,Gn,V3,V4]) gives
        %
        %   h*phi_1(hL)Fn + h^2*phi_2(hL)Gn + h^3*phi_3(hL)V3
        % + h^4*phi_4(hL)V4,
        %
        % we pass
        %
        %   V3 = B3/h,
        %   V4 = B4/h^2.
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