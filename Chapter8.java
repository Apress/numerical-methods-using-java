/*
 * Copyright (c) NM LTD.
 * https://nm.dev/
 * 
 * THIS SOFTWARE IS LICENSED, NOT SOLD.
 * 
 * YOU MAY USE THIS SOFTWARE ONLY AS DESCRIBED IN THE LICENSE.
 * IF YOU ARE NOT AWARE OF AND/OR DO NOT AGREE TO THE TERMS OF THE LICENSE,
 * DO NOT USE THIS SOFTWARE.
 * 
 * THE SOFTWARE IS PROVIDED "AS IS", WITH NO WARRANTY WHATSOEVER,
 * EITHER EXPRESS OR IMPLIED, INCLUDING, WITHOUT LIMITATION,
 * ANY WARRANTIES OF ACCURACY, ACCESSIBILITY, COMPLETENESS,
 * FITNESS FOR A PARTICULAR PURPOSE, MERCHANTABILITY, NON-INFRINGEMENT, 
 * TITLE AND USEFULNESS.
 * 
 * IN NO EVENT AND UNDER NO LEGAL THEORY,
 * WHETHER IN ACTION, CONTRACT, NEGLIGENCE, TORT, OR OTHERWISE,
 * SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR
 * ANY CLAIMS, DAMAGES OR OTHER LIABILITIES,
 * ARISING AS A RESULT OF USING OR OTHER DEALINGS IN THE SOFTWARE.
 */
package dev.nm.nmj;

import dev.nm.algebra.linear.vector.doubles.Vector;
import dev.nm.analysis.differentialequation.pde.finitedifference.PDESolutionGrid2D;
import dev.nm.analysis.differentialequation.pde.finitedifference.PDESolutionTimeSpaceGrid1D;
import dev.nm.analysis.differentialequation.pde.finitedifference.PDESolutionTimeSpaceGrid2D;
import dev.nm.analysis.differentialequation.pde.finitedifference.elliptic.dim2.IterativeCentralDifference;
import dev.nm.analysis.differentialequation.pde.finitedifference.elliptic.dim2.PoissonEquation2D;
import dev.nm.analysis.differentialequation.pde.finitedifference.hyperbolic.dim1.ExplicitCentralDifference1D;
import dev.nm.analysis.differentialequation.pde.finitedifference.hyperbolic.dim1.WaveEquation1D;
import dev.nm.analysis.differentialequation.pde.finitedifference.hyperbolic.dim2.ExplicitCentralDifference2D;
import dev.nm.analysis.differentialequation.pde.finitedifference.hyperbolic.dim2.WaveEquation2D;
import dev.nm.analysis.differentialequation.pde.finitedifference.parabolic.dim1.heatequation.CrankNicolsonHeatEquation1D;
import dev.nm.analysis.differentialequation.pde.finitedifference.parabolic.dim1.heatequation.HeatEquation1D;
import dev.nm.analysis.differentialequation.pde.finitedifference.parabolic.dim2.AlternatingDirectionImplicitMethod;
import dev.nm.analysis.differentialequation.pde.finitedifference.parabolic.dim2.HeatEquation2D;
import dev.nm.analysis.function.rn2r1.AbstractBivariateRealFunction;
import dev.nm.analysis.function.rn2r1.AbstractTrivariateRealFunction;
import dev.nm.analysis.function.rn2r1.BivariateRealFunction;
import dev.nm.analysis.function.rn2r1.TrivariateRealFunction;
import dev.nm.analysis.function.rn2r1.univariate.AbstractUnivariateRealFunction;
import dev.nm.number.DoubleUtils;
import static java.lang.Math.*;

/**
 * Numerical Methods Using Java: For Data Science, Analysis, and Engineering
 *
 * @author haksunli
 * @see
 * https://www.amazon.com/Numerical-Methods-Using-Java-Engineering/dp/1484267966
 * https://nm.dev/
 */
public class Chapter8 {

    public static void main(String[] args) {
        System.out.println("Chapter 8 demos");

        Chapter8 chapter8 = new Chapter8();
        chapter8.solve_Poisson_equation_0010();
        chapter8.solve_Poisson_equation_0020();
        chapter8.solve_wave_equation_1D();
        chapter8.solve_wave_equation_2D();
        chapter8.solve_heat_equation_1D();
        chapter8.solve_heat_equation_2D();
    }

    private void solve_heat_equation_2D() {
        System.out.println("solve a 2-dimensional heat equation");

        // solution domain
        final double a = 4.0, b = 4.0;
        // time domain
        final double T = 5000;

        // heat equation coefficient
        final double beta = 1e-4;

        // initial condition
        final BivariateRealFunction f
                = new AbstractBivariateRealFunction() {
            @Override
            public double evaluate(double x1, double x2) {
                return 0.;
            }
        };

        // boundary condition
        final TrivariateRealFunction g
                = new AbstractTrivariateRealFunction() {
            @Override
            public double evaluate(double t, double x, double y) {
                return exp(y) * cos(x) - exp(x) * cos(y);
            }
        };
        final HeatEquation2D PDE = new HeatEquation2D(beta, T, a, b, f, g);

        AlternatingDirectionImplicitMethod adi
                = new AlternatingDirectionImplicitMethod(1e-5);
        PDESolutionTimeSpaceGrid2D soln = adi.solve(PDE, 50, 39, 39);

        int t = 50;
        int x = 1;
        int y = 1;
        System.out.println(String.format("u(%d,%d,%d) = %f", t, x, y, soln.u(t, x, y)));
        t = 50;
        x = 1;
        y = 16;
        System.out.println(String.format("u(%d,%d,%d) = %f", t, x, y, soln.u(t, x, y)));
        t = 50;
        x = 1;
        y = 31;
        System.out.println(String.format("u(%d,%d,%d) = %f", t, x, y, soln.u(t, x, y)));

        t = 50;
        x = 16;
        y = 1;
        System.out.println(String.format("u(%d,%d,%d) = %f", t, x, y, soln.u(t, x, y)));
        t = 50;
        x = 16;
        y = 16;
        System.out.println(String.format("u(%d,%d,%d) = %f", t, x, y, soln.u(t, x, y)));
        t = 50;
        x = 16;
        y = 31;
        System.out.println(String.format("u(%d,%d,%d) = %f", t, x, y, soln.u(t, x, y)));

        t = 50;
        x = 31;
        y = 1;
        System.out.println(String.format("u(%d,%d,%d) = %f", t, x, y, soln.u(t, x, y)));
        t = 50;
        x = 31;
        y = 16;
        System.out.println(String.format("u(%d,%d,%d) = %f", t, x, y, soln.u(t, x, y)));
        t = 50;
        x = 31;
        y = 31;
        System.out.println(String.format("u(%d,%d,%d) = %f", t, x, y, soln.u(t, x, y)));
    }

    private void solve_heat_equation_1D() {
        System.out.println("solve a 1-dimensional heat equation");

        HeatEquation1D pde = new HeatEquation1D(
                1e-5, // heat equation coefficient
                1., 6000., // solution domain bounds
                new AbstractUnivariateRealFunction() {
            @Override
            public double evaluate(double x) {
                return 2.0 * x + sin(2.0 * PI * x); // initial condition
            }
        },
                0., new AbstractUnivariateRealFunction() {
            @Override
            public double evaluate(double t) {
                return 0.; // boundary condition at x = 0
            }
        },
                0., new AbstractUnivariateRealFunction() {
            @Override
            public double evaluate(double t) {
                return 2.; // boundary condition at x = 6000
            }
        });

        // c_k are 0 for Dirichlet boundary conditions
        int m = 50;
        int n = 39;

        PDESolutionTimeSpaceGrid1D soln
                = new CrankNicolsonHeatEquation1D().solve(pde, m, n);

        int t = 0;
        int x = 1;
        System.out.println(String.format("u(%d,%d) = %f", t, x, soln.u(t, x)));
        t = 0;
        x = 16;
        System.out.println(String.format("u(%d,%d) = %f", t, x, soln.u(t, x)));
        t = 0;
        x = 31;
        System.out.println(String.format("u(%d,%d) = %f", t, x, soln.u(t, x)));

        t = 15;
        x = 1;
        System.out.println(String.format("u(%d,%d) = %f", t, x, soln.u(t, x)));
        t = 15;
        x = 16;
        System.out.println(String.format("u(%d,%d) = %f", t, x, soln.u(t, x)));
        t = 15;
        x = 31;
        System.out.println(String.format("u(%d,%d) = %f", t, x, soln.u(t, x)));

        t = 30;
        x = 1;
        System.out.println(String.format("u(%d,%d) = %f", t, x, soln.u(t, x)));
        t = 30;
        x = 16;
        System.out.println(String.format("u(%d,%d) = %f", t, x, soln.u(t, x)));
        t = 30;
        x = 31;
        System.out.println(String.format("u(%d,%d) = %f", t, x, soln.u(t, x)));

        t = 45;
        x = 1;
        System.out.println(String.format("u(%d,%d) = %f", t, x, soln.u(t, x)));
        t = 45;
        x = 16;
        System.out.println(String.format("u(%d,%d) = %f", t, x, soln.u(t, x)));
        t = 45;
        x = 31;
        System.out.println(String.format("u(%d,%d) = %f", t, x, soln.u(t, x)));
    }

    private void solve_wave_equation_2D() {
        System.out.println("solve a 2-dimensional wave equation");

        double c2 = 1. / 4; // wave speed squared
        double T = 2., a = 2., b = 2.; // the solution domain bounds
        WaveEquation2D pde = new WaveEquation2D(
                c2, T, a, b,
                new AbstractBivariateRealFunction() {
            @Override
            public double evaluate(double x, double y) {
                return 0.1 * sin(PI * x) * sin(PI * y / 2.);
            }
        },
                new AbstractBivariateRealFunction() {
            @Override
            public double evaluate(double x, double y) {
                return 0.;
            }
        });

        int m = 40; // dt = T/m
        int n = 39; // dx = a/n
        int p = 39; // dy = b/p
        PDESolutionTimeSpaceGrid2D soln = new ExplicitCentralDifference2D().solve(pde, m, n, p);

        int t = 40; // t index
        int x = 1; // x index
        int y = 1; // y index
        System.out.println(String.format("u(%d,%d,%d) = %f", t, x, y, soln.u(t, x, y)));
        t = 40; // t index
        x = 1; // x index
        y = 16; // y index
        System.out.println(String.format("u(%d,%d,%d) = %f", t, x, y, soln.u(t, x, y)));
        t = 40; // t index
        x = 1; // x index
        y = 31; // y index
        System.out.println(String.format("u(%d,%d,%d) = %f", t, x, y, soln.u(t, x, y)));

        t = 40; // t index
        x = 16; // x index
        y = 1; // y index
        System.out.println(String.format("u(%d,%d,%d) = %f", t, x, y, soln.u(t, x, y)));
        t = 40; // t index
        x = 16; // x index
        y = 16; // y index
        System.out.println(String.format("u(%d,%d,%d) = %f", t, x, y, soln.u(t, x, y)));
        t = 40; // t index
        x = 16; // x index
        y = 31; // y index
        System.out.println(String.format("u(%d,%d,%d) = %f", t, x, y, soln.u(t, x, y)));

        t = 40; // t index
        x = 31; // x index
        y = 1; // y index
        System.out.println(String.format("u(%d,%d,%d) = %f", t, x, y, soln.u(t, x, y)));
        t = 40; // t index
        x = 31; // x index
        y = 16; // y index
        System.out.println(String.format("u(%d,%d,%d) = %f", t, x, y, soln.u(t, x, y)));
        t = 40; // t index
        x = 31; // x index
        y = 31; // y index
        System.out.println(String.format("u(%d,%d,%d) = %f", t, x, y, soln.u(t, x, y)));
    }

    private void solve_wave_equation_1D() {
        System.out.println("solve a 1-dimensional wave equation");

        final double c2 = 4.0; // c^2
        final double T = 1.0; // time upper bond
        final double a = 2.0; // x upper bound
        WaveEquation1D pde
                = new WaveEquation1D(
                        c2,
                        T,
                        a,
                        new AbstractUnivariateRealFunction() {
                    @Override
                    public double evaluate(double x) {
                        return 0.1 * sin(PI * x); // 0.1 * sin(π x)
                    }
                },
                        new AbstractUnivariateRealFunction() {
                    @Override
                    public double evaluate(double x) {
                        return 0.2 * PI * sin(PI * x); // 0.2π * sin(π x)
                    }
                });

        int m = 80; // dt = T/m
        int n = 39; // dx = a/n
        PDESolutionTimeSpaceGrid1D soln = new ExplicitCentralDifference1D().solve(pde, m, n);

        int t = 0; // time index
        int x = 1; // x index
        System.out.println(String.format("u(%d,%d) = %f", t, x, soln.u(t, x)));
        t = 0;
        x = 16;
        System.out.println(String.format("u(%d,%d) = %f", t, x, soln.u(t, x)));
        t = 0;
        x = 31;
        System.out.println(String.format("u(%d,%d) = %f", t, x, soln.u(t, x)));

        t = 20;
        x = 1;
        System.out.println(String.format("u(%d,%d) = %f", t, x, soln.u(t, x)));
        t = 20;
        x = 16;
        System.out.println(String.format("u(%d,%d) = %f", t, x, soln.u(t, x)));
        t = 20;
        x = 31;
        System.out.println(String.format("u(%d,%d) = %f", t, x, soln.u(t, x)));

        t = 40;
        x = 1;
        System.out.println(String.format("u(%d,%d) = %f", t, x, soln.u(t, x)));
        t = 40;
        x = 16;
        System.out.println(String.format("u(%d,%d) = %f", t, x, soln.u(t, x)));
        t = 40;
        x = 31;
        System.out.println(String.format("u(%d,%d) = %f", t, x, soln.u(t, x)));

        t = 60;
        x = 1;
        System.out.println(String.format("u(%d,%d) = %f", t, x, soln.u(t, x)));
        t = 60;
        x = 16;
        System.out.println(String.format("u(%d,%d) = %f", t, x, soln.u(t, x)));
        t = 60;
        x = 31;
        System.out.println(String.format("u(%d,%d) = %f", t, x, soln.u(t, x)));
    }

    private void solve_Poisson_equation_0020() {
        System.out.println("solve a 2-dimensional Poisson equation");

        BivariateRealFunction ZERO // a constant zero function, f = 0
                = new AbstractBivariateRealFunction() {
            @Override
            public double evaluate(double x, double y) {
                return 0;
            }

            @Override
            public Double evaluate(Vector x) {
                return 0.;
            }
        };

        // the boundary conditions
        final double EPSION = 1e-8;
        BivariateRealFunction g = new AbstractBivariateRealFunction() {
            @Override
            public double evaluate(double x, double y) {
                if (DoubleUtils.isZero(x, EPSION) || DoubleUtils.isZero(y, EPSION)) {
                    return 0;
                } else if (DoubleUtils.equal(x, 0.5, EPSION)) {
                    return 200. * y;
                } else if (DoubleUtils.equal(y, 0.5, EPSION)) {
                    return 200. * x;
                }

                // not reachable; don't matter
                return Double.NaN;
            }
        };

        double a = 0.5; // width of the x-dimension
        double b = 0.5; // height of the y-dimension
        PoissonEquation2D pde = new PoissonEquation2D(a, b, ZERO, g);
        IterativeCentralDifference solver = new IterativeCentralDifference(
                EPSION, // precision
                40); // max number of iterations
        PDESolutionGrid2D soln = solver.solve(pde, 4, 4);
        int k = 1, j = 1; // node indices
        double u_11 = soln.u(k, j);
        System.out.println(String.format("u_%d,%d = u(%f,%f): %f", k, j, soln.x(k), soln.y(j), u_11));
        k = 1;
        j = 2;
        double u_12 = soln.u(k, j);
        System.out.println(String.format("u_%d,%d = u(%f,%f): %f", k, j, soln.x(k), soln.y(j), u_12));
        k = 2;
        j = 1;
        double u_21 = soln.u(k, j);
        System.out.println(String.format("u_%d,%d = u(%f,%f): %f", k, j, soln.x(k), soln.y(j), u_21));
        k = 2;
        j = 2;
        double u_22 = soln.u(k, j);
        System.out.println(String.format("u_%d,%d = u(%f,%f): %f", k, j, soln.x(k), soln.y(j), u_22));
        k = 3;
        j = 3;
        double u_33 = soln.u(k, j);
        System.out.println(String.format("u_%d,%d = u(%f,%f): %f", k, j, soln.x(k), soln.y(j), u_33));
        k = 4;
        j = 4;
        double u_44 = soln.u(k, j);
        System.out.println(String.format("u_%d,%d = u(%f,%f): %f", k, j, soln.x(k), soln.y(j), u_44));
        k = 5;
        j = 5;
        double u_55 = soln.u(k, j);
        System.out.println(String.format("u_%d,%d = u(%f,%f): %f", k, j, soln.x(k), soln.y(j), u_55));
    }

    private void solve_Poisson_equation_0010() {
        System.out.println("solve a 2-dimensional Poisson equation");

        BivariateRealFunction ZERO // a constant zero function, f = 0
                = new AbstractBivariateRealFunction() {
            @Override
            public double evaluate(double x, double y) {
                return 0;
            }

            @Override
            public Double evaluate(Vector x) {
                return 0.;
            }
        };

        // the boundary conditions
        BivariateRealFunction g = new AbstractBivariateRealFunction() {
            @Override
            public double evaluate(double x, double y) {
                return log((1. + x) * (1. + x) + y * y);
            }
        };

        double a = 1.; // width of the x-dimension
        double b = 1.; // height of the y-dimension
        PoissonEquation2D pde = new PoissonEquation2D(a, b, ZERO, g);
        IterativeCentralDifference solver = new IterativeCentralDifference(
                1e-8, // precision
                40); // max number of iterations
        PDESolutionGrid2D soln = solver.solve(pde, 2, 2);
        int k = 1, j = 1; // node indices
        double u_11 = soln.u(k, j); // x = 0.3, y = 0.3
        System.out.println(String.format("u_%d,%d = u(%f,%f): %f", k, j, soln.x(k), soln.y(j), u_11));
        k = 1;
        j = 2;
        double u_12 = soln.u(k, j); // x = 0.3, y = 0.6
        System.out.println(String.format("u_%d,%d = u(%f,%f): %f", k, j, soln.x(k), soln.y(j), u_12));
        k = 2;
        j = 1;
        double u_21 = soln.u(k, j); // x = 0.6, y = 0.3
        System.out.println(String.format("u_%d,%d = u(%f,%f): %f", k, j, soln.x(k), soln.y(j), u_21));
        k = 2;
        j = 2;
        double u_22 = soln.u(k, j); // x = 0.6, y = 0.6
        System.out.println(String.format("u_%d,%d = u(%f,%f): %f", k, j, soln.x(k), soln.y(j), u_22));
    }

}
