# ExpH: Exponential Hermite-Type Methods #

#### Vu Thai Luan<sup>1</sup> and Nguyen Van Hoang<sup>1</sup> ####

<sup>1</sup>Department of Mathematics & Statistics, Texas Tech University,  
1108 Memorial Circle, Lubbock, Texas 79409, USA

This repository contains MATLAB implementations and numerical experiments associated with the manuscript:

**Vu Thai Luan and Nguyen Van Hoang, “Exponential Hermite-Type Methods,” 2026.**

Exponential Hermite-type (`ExpH`) methods are explicit exponential integrators for stiff semilinear parabolic problems of the form

$$ u'(t)=Lu(t)+N(u(t))=:F(u(t)), \qquad u(t_0)=u_0, $$

where \(L\) represents the stiff linear differential operator and \(N\) is a sufficiently smooth nonlinear function.

Unlike standard exponential Runge-Kutta methods, which construct their stages using only nonlinear function values, ExpH methods approximate the nonlinear term in the variation-of-constants formula using both its value and its derivative along the solution:

$$ \frac{d}{dt}N(u(t)) = N'(u(t))F(u(t)). $$

We therefore define

$$ G(u)=N'(u)F(u). $$

The resulting Hermite-type approximation is integrated exactly against the exponential kernel. Consequently, the methods involve only \(\varphi\)-functions of the fixed linear operator \(L\). This distinguishes ExpH methods from exponential Rosenbrock methods, whose matrix functions are evaluated at step-dependent Jacobians.

The general \(s\)-stage ExpH method is

$$
\begin{aligned}
U_{ni} &={}&u_n+c_i h\varphi_1(c_i hL)F(u_n)
+c_i^2h^2\varphi_2(c_i hL)G(u_n)+h^2\sum_{j=2}^{i-1}a_{ij}(hL)H_{nj}, \qquad i=2,\ldots,s, \\ 
u_{n+1} &={}&u_n+h\varphi_1(hL)F(u_n) +h^2\varphi_2(hL)G(u_n)+h^2\sum_{i=2}^{s}b_i(hL)H_{ni},
\end{aligned}
$$
where
$$
H_{ni}=G(U_{ni})-G(u_n).
$$

The matrix functions are defined by

```math
\varphi_0(z)=e^z,
\qquad
\varphi_k(z)
=
\int_0^1 e^{(1-\theta)z}
\frac{\theta^{k-1}}{(k-1)!}\,d\theta,
\qquad k\geq 1.
```

ExpH methods of orders two through five are implemented in this repository:

- second-order `ExpH2`,
- third-order `ExpH3`,
- fourth-order `ExpH4s2` and `ExpH4s3`,
- fifth-order `ExpH5s3`, `ExpH5s4`, and `ExpH5s5`.

The Hermite-type construction significantly simplifies the stiff order theory. Methods up to order five require only four stiff order conditions, compared with sixteen stiff order conditions for standard exponential Runge-Kutta methods. This reduction permits the construction of high-order explicit schemes with relatively few stages.

The numerical experiments compare the fourth-order methods `ExpH4s2` and `ExpH4s3` with the exponential Runge-Kutta method `expRK4s5` and the exponential Rosenbrock methods `exprb42` and `exprb43`.

The test problems considered in this repository include:

1. a one-dimensional stiff semilinear parabolic problem with a known exact solution;
2. a two-dimensional advection-diffusion-reaction problem;
3. a three-dimensional Brusselator system.
4. a one-dimensional semilinear parabolic problem with a nonlocal integral term and a known exact solution.

The first problem is used to verify the convergence orders of all ExpH methods from orders two through five. The two- and three-dimensional problems are used to compare the accuracy and computational efficiency of the ExpH methods with established exponential Runge-Kutta and exponential Rosenbrock schemes. The final problem, which contains a nonlocal nonlinear term, is used to verify the convergence orders of the ExpH methods within the analytic-semigroup framework.

Linear combinations of \(\varphi\)-functions applied to vectors are evaluated using the adaptive Krylov-subspace routine `phipm_simul_iom`. The numerical experiments confirm the predicted convergence orders and demonstrate that the proposed ExpH methods can achieve higher accuracy at substantially lower computational cost than the comparison methods.
## Numerical experiments


## Dependencies

This repository uses external routines for tensor operations and Krylov-based evaluations of matrix-function actions.

Some tensor operations use KronPACK, which is distributed under the MIT License.

The routine `phipm_simul_iom` is used to compute linear combinations of phi-functions acting on vectors. This routine is distributed under the BSD 3-Clause License.

The original copyright and license notices of third-party routines are retained in the corresponding source files. See `ThirdPartyNotices.md` for details.

## License

The original code developed in this repository is distributed under the MIT License.

Third-party components remain under their respective licenses.

