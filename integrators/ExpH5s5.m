function [t, uFinal, U] = ExpH5s5(Nfun, L, JN, t0, tEnd, u0, nSteps, tol)
%EXPH5S5 Fifth-order five-stage exponential Hermite method.
%
%   Solves the autonomous semilinear problem
%
%       u'(t) = L*u(t) + N(u(t)),     u(t0) = u0.
%
%   Define the full right-hand side
%
%       F(u) = L*u + N(u),
%
%   and the Hermite derivative data
%
%       G_n = N'(u_n)F(u_n).
%
%   This method uses the ExpH form
%
%       u_{n+1} = u_n
%               + h*phi_1(hL)F_n
%               + h^2*phi_2(hL)G_n
%               + h^2*phi_3(hL)B_3
%               + h^2*phi_4(hL)B_4
%               + h^2*phi_5(hL)B_5.
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

    % Stage nodes
    c2 = 1/2;
    c3 = 1/2;
    c4 = 2/3;
    c5 = 1;

    % Coefficients for the final Hermite interpolation correction
    d33 =  c4*c5       / (c3*(c3 - c4)*(c3 - c5));
    d34 = -2*(c4 + c5) / (c3*(c3 - c4)*(c3 - c5));
    d35 =  6           / (c3*(c3 - c4)*(c3 - c5));

    d43 =  c3*c5       / (c4*(c4 - c3)*(c4 - c5));
    d44 = -2*(c3 + c5) / (c4*(c4 - c3)*(c4 - c5));
    d45 =  6           / (c4*(c4 - c3)*(c4 - c5));

    d53 =  c3*c4       / (c5*(c5 - c3)*(c5 - c4));
    d54 = -2*(c3 + c4) / (c5*(c5 - c3)*(c5 - c4));
    d55 =  6           / (c5*(c5 - c3)*(c5 - c4));

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

        Nu = Nfun(u);

        % Full right-hand side:
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
        % Stages U_{n3}, U_{n4}, U_{n5}
        %
        % For i = 3,4,5:
        %
        % U_{ni} = u_n
        %        + ci*h*phi_1(ci*h*L)F_n
        %        + ci^2*h^2*phi_2(ci*h*L)G_n
        %        + (ci^3/c2)*h^2*phi_3(ci*h*L)H_{n2}.
        %
        % Since phipm_simul_iom(s,L,[0,Fn,Gn,V3]) gives
        %
        %        s*phi_1(sL)Fn
        %      + s^2*phi_2(sL)Gn
        %      + s^3*phi_3(sL)V3,
        %
        % we pass
        %
        %        V3 = H_{n2}/(c2*h).
        % ---------------------------------------------------------------

        stageIncs = phipm_simul_iom( ...
            [c3, c4, c5]*h, L, [zeroVec, Fn, Gn, Hn2/(c2*h)], tol, 1, 2);

        Un3 = u + stageIncs(:, 1);
        Un4 = u + stageIncs(:, 2);
        Un5 = u + stageIncs(:, 3);

        F3 = L*Un3 + Nfun(Un3);
        F4 = L*Un4 + Nfun(Un4);
        F5 = L*Un5 + Nfun(Un5);

        Hn3 = JN(Un3)*F3 - Gn;
        Hn4 = JN(Un4)*F4 - Gn;
        Hn5 = JN(Un5)*F5 - Gn;

        % ---------------------------------------------------------------
        % Final Hermite correction coefficients
        % ---------------------------------------------------------------

        B3 = d33*Hn3 + d43*Hn4 + d53*Hn5;
        B4 = d34*Hn3 + d44*Hn4 + d54*Hn5;
        B5 = d35*Hn3 + d45*Hn4 + d55*Hn5;

        % ---------------------------------------------------------------
        % Final update:
        %
        % u_{n+1} = u_n
        %          + h*phi_1(hL)F_n
        %          + h^2*phi_2(hL)G_n
        %          + h^2*phi_3(hL)B3
        %          + h^2*phi_4(hL)B4
        %          + h^2*phi_5(hL)B5.
        %
        % phipm_simul_iom(h,L,[0,Fn,Gn,V3,V4,V5]) gives
        %
        %        h*phi_1(hL)Fn
        %      + h^2*phi_2(hL)Gn
        %      + h^3*phi_3(hL)V3
        %      + h^4*phi_4(hL)V4
        %      + h^5*phi_5(hL)V5.
        %
        % Therefore
        %
        %        V3 = B3/h,
        %        V4 = B4/h^2,
        %        V5 = B5/h^3.
        % ---------------------------------------------------------------

        finalIncrement = phipm_simul_iom( ...
            h, L, [zeroVec, Fn, Gn, B3/h, B4/h^2, B5/h^3], tol, 1, 2);

        u = u + finalIncrement;

        if saveTrajectory
            U(:, k + 1) = u;
        end
    end

    uFinal = u;
end