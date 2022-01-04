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

import dev.nm.algebra.linear.matrix.doubles.Matrix;
import dev.nm.algebra.linear.matrix.doubles.matrixtype.dense.DenseMatrix;
import dev.nm.algebra.linear.vector.doubles.Vector;
import dev.nm.algebra.linear.vector.doubles.dense.DenseVector;
import dev.nm.analysis.function.matrix.RntoMatrix;
import dev.nm.analysis.function.polynomial.Polynomial;
import dev.nm.analysis.function.rn2r1.AbstractBivariateRealFunction;
import dev.nm.analysis.function.rn2r1.RealScalarFunction;
import dev.nm.analysis.function.rn2rm.RealVectorFunction;
import dev.nm.analysis.function.special.gamma.LogGamma;
import dev.nm.solver.IterativeSolution;
import dev.nm.solver.multivariate.unconstrained.BruteForceMinimizer;
import dev.nm.solver.multivariate.unconstrained.DoubleBruteForceMinimizer;
import dev.nm.solver.multivariate.unconstrained.c2.conjugatedirection.ConjugateGradientMinimizer;
import dev.nm.solver.multivariate.unconstrained.c2.conjugatedirection.FletcherReevesMinimizer;
import dev.nm.solver.multivariate.unconstrained.c2.conjugatedirection.PowellMinimizer;
import dev.nm.solver.multivariate.unconstrained.c2.conjugatedirection.ZangwillMinimizer;
import dev.nm.solver.multivariate.unconstrained.c2.quasinewton.BFGSMinimizer;
import dev.nm.solver.multivariate.unconstrained.c2.quasinewton.DFPMinimizer;
import dev.nm.solver.multivariate.unconstrained.c2.quasinewton.HuangMinimizer;
import dev.nm.solver.multivariate.unconstrained.c2.quasinewton.PearsonMinimizer;
import dev.nm.solver.multivariate.unconstrained.c2.quasinewton.QuasiNewtonMinimizer;
import dev.nm.solver.multivariate.unconstrained.c2.quasinewton.RankOneMinimizer;
import dev.nm.solver.multivariate.unconstrained.c2.steepestdescent.FirstOrderMinimizer;
import dev.nm.solver.multivariate.unconstrained.c2.steepestdescent.GaussNewtonMinimizer;
import dev.nm.solver.multivariate.unconstrained.c2.steepestdescent.NewtonRaphsonMinimizer;
import dev.nm.solver.multivariate.unconstrained.c2.steepestdescent.SteepestDescentMinimizer;
import dev.nm.solver.problem.C2OptimProblem;
import dev.nm.solver.problem.C2OptimProblemImpl;
import dev.nm.solver.problem.OptimProblem;
import dev.nm.root.univariate.UnivariateMinimizer;
import dev.nm.root.univariate.bracketsearch.BracketSearchMinimizer;
import dev.nm.root.univariate.bracketsearch.BrentMinimizer;
import dev.nm.root.univariate.bracketsearch.FibonaccMinimizer;
import dev.nm.root.univariate.bracketsearch.GoldenMinimizer;
import static java.lang.Math.pow;
import static java.lang.Math.sqrt;
import java.util.ArrayList;
import java.util.List;

/**
 * Numerical Methods Using Java: For Data Science, Analysis, and Engineering
 *
 * @author haksunli
 * @see
 * https://www.amazon.com/Numerical-Methods-Using-Java-Engineering/dp/1484267966
 * https://nm.dev/
 */
public class Chapter9 {

    public static void main(String[] args) throws Exception {
        System.out.println("Chapter 9 demos");

        Chapter9 chapter9 = new Chapter9();
        chapter9.solve_by_brute_force_search_1();
        chapter9.solve_by_brute_force_search_2();
        chapter9.solve_by_brute_force_search_3();
        chapter9.solve_by_brute_force_search_4();
        chapter9.solve_loggamma_by_bracketing();
        chapter9.solve_by_steepest_descent();
        chapter9.solve_by_Newton_Raphson();
        chapter9.solve_by_Gauss_Newton();
        chapter9.solve_by_conjugate_direction_methods();
        chapter9.solve_by_quasi_Newton();
    }

    public void solve_by_brute_force_search_1() throws Exception {
        System.out.println("solve uniivariate function by brute force search");

        // define the optimization problem using an objective function
        OptimProblem problem = new OptimProblem() {

            @Override
            public int dimension() {
                return 1;
            }

            @Override
            public RealScalarFunction f() {
                return new RealScalarFunction() {

                    // the objective function
                    @Override
                    public Double evaluate(Vector v) {
                        double x = v.get(1);
                        Polynomial polynomial = new Polynomial(1, 0, -4); // f(x) = x^2 - 4
                        double fx = polynomial.evaluate(x);
                        return fx;
                    }

                    @Override
                    public int dimensionOfDomain() {
                        return 1;
                    }

                    @Override
                    public int dimensionOfRange() {
                        return 1;
                    }
                };
            }
        };

        // set up the solver to use and the solution
        DoubleBruteForceMinimizer solver = new DoubleBruteForceMinimizer(false);
        BruteForceMinimizer.Solution soln = solver.solve(problem);

        // for brute force search, we need to explicitly enumerate the values in the domain
        List<Vector> domain = new ArrayList<>();
        domain.add(new DenseVector(-2.));
        domain.add(new DenseVector(-1.));
        domain.add(new DenseVector(0.)); // the minimizer
        domain.add(new DenseVector(1.));
        domain.add(new DenseVector(2.));
        soln.setDomain(domain);

        System.out.println(String.format("f(%s) = %f", soln.minimizer(), soln.min()));
    }

    public void solve_by_brute_force_search_2() throws Exception {
        System.out.println("solve multivariate function by brute force search");

        OptimProblem problem = new OptimProblem() {

            @Override
            public int dimension() {
                return 2;
            }

            @Override
            public RealScalarFunction f() {
                return new RealScalarFunction() {

                    @Override
                    public Double evaluate(Vector v) {
                        double x = v.get(1);
                        double y = v.get(2);

                        double fx = x * x + y * y;
                        return fx;
                    }

                    @Override
                    public int dimensionOfDomain() {
                        return 2;
                    }

                    @Override
                    public int dimensionOfRange() {
                        return 1;
                    }
                };
            }
        };

        DoubleBruteForceMinimizer bf = new DoubleBruteForceMinimizer(true);
        BruteForceMinimizer.Solution soln = bf.solve(problem);
        List<Vector> domain = new ArrayList<>();
        domain.add(new DenseVector(-2., -2.));
        domain.add(new DenseVector(-1., -1.));
        domain.add(new DenseVector(0., 0.)); // the minimizer
        domain.add(new DenseVector(1., 1.));
        domain.add(new DenseVector(2., 2.));
        soln.setDomain(domain);

        System.out.println(String.format("f(%s) = %f", soln.minimizer(), soln.min()));
    }

    public void solve_by_brute_force_search_3() throws Exception {
        System.out.println("solve uniivariate function by brute force search");

        // set up the solver to use and the solution
        DoubleBruteForceMinimizer solver = new DoubleBruteForceMinimizer(false);
        BruteForceMinimizer.Solution soln
                = solver.solve(new C2OptimProblemImpl(new Polynomial(1, 0, -4))); // f(x) = x^2 - 4

        // for brute force search, we need to explicitly enumerate the values in the domain
        List<Vector> domain = new ArrayList<>();
        domain.add(new DenseVector(-2.));
        domain.add(new DenseVector(-1.));
        domain.add(new DenseVector(0.)); // the minimizer
        domain.add(new DenseVector(1.));
        domain.add(new DenseVector(2.));
        soln.setDomain(domain);

        System.out.println(String.format("f(%s) = %f", soln.minimizer(), soln.min()));
    }

    public void solve_by_brute_force_search_4() throws Exception {
        System.out.println("solve multivariate function by brute force search");

        DoubleBruteForceMinimizer bf = new DoubleBruteForceMinimizer(true);
        BruteForceMinimizer.Solution soln = bf.solve(
                new C2OptimProblemImpl(
                        new AbstractBivariateRealFunction() {
                    @Override
                    public double evaluate(double x, double y) {
                        double fx = x * x + y * y;
                        return fx;
                    }
                }));

        List<Vector> domain = new ArrayList<>();
        domain.add(new DenseVector(-2., -2.));
        domain.add(new DenseVector(-1., -1.));
        domain.add(new DenseVector(0., 0.)); // the minimizer
        domain.add(new DenseVector(1., 1.));
        domain.add(new DenseVector(2., 2.));
        soln.setDomain(domain);

        System.out.println(String.format("f(%s) = %f", soln.minimizer(), soln.min()));
    }

    public void solve_loggamma_by_bracketing() throws Exception {
        System.out.println("solve loggamma function by bracketing");

        LogGamma logGamma = new LogGamma(); // the log-gamma function

        BracketSearchMinimizer solver1 = new FibonaccMinimizer(1e-8, 15);
        UnivariateMinimizer.Solution soln1 = solver1.solve(logGamma);
        double x_min_1 = soln1.search(0, 5);
        System.out.println(String.format("f(%f) = %f", x_min_1, logGamma.evaluate(x_min_1)));

        BracketSearchMinimizer solver2 = new GoldenMinimizer(1e-8, 15);
        UnivariateMinimizer.Solution soln2 = solver2.solve(logGamma);
        double x_min_2 = soln2.search(0, 5);
        System.out.println(String.format("f(%f) = %f", x_min_2, logGamma.evaluate(x_min_2)));

        BracketSearchMinimizer solver3 = new BrentMinimizer(1e-8, 10);
        UnivariateMinimizer.Solution soln3 = solver3.solve(logGamma);
        double x_min_3 = soln3.search(0, 5);
        System.out.println(String.format("f(%f) = %f", x_min_3, logGamma.evaluate(x_min_3)));
    }

    public void solve_by_steepest_descent() throws Exception {
        System.out.println("solve multivariate function by steepest-descent");

        // the objective function
        // the global minimizer is at x = [0,0,0,0]
        RealScalarFunction f = new RealScalarFunction() {

            @Override
            public Double evaluate(Vector x) {
                double x1 = x.get(1);
                double x2 = x.get(2);
                double x3 = x.get(3);
                double x4 = x.get(4);

                double result = pow(x1 - 4 * x2, 4);
                result += 12 * pow(x3 - x4, 4);
                result += 3 * pow(x2 - 10 * x3, 2);
                result += 55 * pow(x1 - 2 * x4, 2);

                return result;
            }

            @Override
            public int dimensionOfDomain() {
                return 4;
            }

            @Override
            public int dimensionOfRange() {
                return 1;
            }
        };

        // the gradient function
        RealVectorFunction g = new RealVectorFunction() {

            @Override
            public Vector evaluate(Vector x) {
                double x1 = x.get(1);
                double x2 = x.get(2);
                double x3 = x.get(3);
                double x4 = x.get(4);

                double[] result = new double[4];
                result[0] = 4 * pow(x1 - 4 * x2, 3) + 110 * (x1 - 2 * x4);
                result[1] = -16 * pow(x1 - 4 * x2, 3) + 6 * (x2 - 10 * x3);
                result[2] = 48 * pow(x3 - x4, 3) - 60 * (x2 - 10 * x3);
                result[3] = -48 * pow(x3 - x4, 3) - 220 * (x1 - 2 * x4);
                return new DenseVector(result);
            }

            @Override
            public int dimensionOfDomain() {
                return 4;
            }

            @Override
            public int dimensionOfRange() {
                return 4;
            }
        };

        C2OptimProblem problem = new C2OptimProblemImpl(f, g); // only gradient information
        SteepestDescentMinimizer firstOrderMinimizer
                = new FirstOrderMinimizer(
                        FirstOrderMinimizer.Method.IN_EXACT_LINE_SEARCH, // FirstOrderMinimizer.Method.ANALYTIC
                        1e-8,
                        40000
                );
        IterativeSolution<Vector> soln = firstOrderMinimizer.solve(problem);

        Vector xmin = soln.search(new DenseVector(new double[]{1, -1, -1, 1}));
        double f_xmin = f.evaluate(xmin);
        System.out.println(String.format("f(%s) = %f", xmin.toString(), f_xmin));
    }

    public void solve_by_Newton_Raphson() throws Exception {
        System.out.println("solve multivariate function by Newton-Raphson");

        // the objective function
        // the global minimizer is at x = [0,0,0,0]
        RealScalarFunction f = new RealScalarFunction() {

            @Override
            public Double evaluate(Vector x) {
                double x1 = x.get(1);
                double x2 = x.get(2);
                double x3 = x.get(3);
                double x4 = x.get(4);

                double result = pow(x1 - 4 * x2, 4);
                result += 12 * pow(x3 - x4, 4);
                result += 3 * pow(x2 - 10 * x3, 2);
                result += 55 * pow(x1 - 2 * x4, 2);

                return result;
            }

            @Override
            public int dimensionOfDomain() {
                return 4;
            }

            @Override
            public int dimensionOfRange() {
                return 1;
            }
        };

        // the gradient function
        RealVectorFunction g = new RealVectorFunction() {

            @Override
            public Vector evaluate(Vector x) {
                double x1 = x.get(1);
                double x2 = x.get(2);
                double x3 = x.get(3);
                double x4 = x.get(4);

                double[] result = new double[4];
                result[0] = 4 * pow(x1 - 4 * x2, 3) + 110 * (x1 - 2 * x4);
                result[1] = -16 * pow(x1 - 4 * x2, 3) + 6 * (x2 - 10 * x3);
                result[2] = 48 * pow(x3 - x4, 3) - 60 * (x2 - 10 * x3);
                result[3] = -48 * pow(x3 - x4, 3) - 220 * (x1 - 2 * x4);
                return new DenseVector(result);
            }

            @Override
            public int dimensionOfDomain() {
                return 4;
            }

            @Override
            public int dimensionOfRange() {
                return 4;
            }
        };

        C2OptimProblem problem = new C2OptimProblemImpl(f, g); // use numerical Hessian
        SteepestDescentMinimizer newtonRaphsonMinimizer
                = new NewtonRaphsonMinimizer(
                        1e-8,
                        20
                );
        IterativeSolution<Vector> soln = newtonRaphsonMinimizer.solve(problem);

        Vector xmin = soln.search(new DenseVector(new double[]{1, -1, -1, 1}));
        double f_xmin = f.evaluate(xmin);
        System.out.println(String.format("f(%s) = %f", xmin.toString(), f_xmin));
    }

    public void solve_by_Gauss_Newton() throws Exception {
        System.out.println("solve multivariate function by Gauss-Newton");

        // the objective function
        //  the global minimizer is at x = [0,0,0,0]
        RealVectorFunction f = new RealVectorFunction() {

            @Override
            public Vector evaluate(Vector x) {
                double x1 = x.get(1);
                double x2 = x.get(2);
                double x3 = x.get(3);
                double x4 = x.get(4);

                double[] fx = new double[4];
                fx[0] = pow(x1 - 4 * x2, 2);
                fx[1] = sqrt(12) * pow(x3 - x4, 2);
                fx[2] = sqrt(3) * (x2 - 10 * x3);
                fx[3] = sqrt(55) * (x1 - 2 * x4);

                return new DenseVector(fx);
            }

            @Override
            public int dimensionOfDomain() {
                return 4;
            }

            @Override
            public int dimensionOfRange() {
                return 4;
            }
        };

        // the Jacobian
        RntoMatrix J = new RntoMatrix() {

            @Override
            public Matrix evaluate(Vector x) {
                double x1 = x.get(1);
                double x2 = x.get(2);
                double x3 = x.get(3);
                double x4 = x.get(4);

                Matrix Jx = new DenseMatrix(4, 4);

                double value = 2 * (x1 - 4 * x2);
                Jx.set(1, 1, value);

                value = -8 * (x1 - 4 * x2);
                Jx.set(1, 2, value);

                value = 2 * sqrt(12) * (x3 - x4);
                Jx.set(2, 3, value);
                Jx.set(2, 4, -value);

                Jx.set(3, 2, sqrt(3));
                Jx.set(3, 3, -10 * sqrt(3));

                Jx.set(4, 1, sqrt(55));
                Jx.set(4, 4, -2 * sqrt(55));

                return Jx;
            }

            @Override
            public int dimensionOfDomain() {
                return 4;
            }

            @Override
            public int dimensionOfRange() {
                return 1;
            }
        };

        GaussNewtonMinimizer optim1 = new GaussNewtonMinimizer(1e-8, 10);

        IterativeSolution<Vector> soln = optim1.solve(f, J);//analytical gradient

        Vector xmin = soln.search(new DenseVector(new double[]{1, -1, -1, 1}));
        System.out.println(String.format("f(%s) = %s", xmin.toString(), f.evaluate(xmin).toString()));
    }

    public void solve_by_conjugate_direction_methods() throws Exception {
        System.out.println("solve multivariate function by conjugate-direction methods");

        /**
         * The Himmelblau function: f(x) = (x1^2 + x2 - 11)^2 + (x1 + x2^2 -
         * 7)^2
         */
        RealScalarFunction f = new RealScalarFunction() {
            @Override
            public Double evaluate(Vector x) {
                double x1 = x.get(1);
                double x2 = x.get(2);

                double result = pow(x1 * x1 + x2 - 11, 2);
                result += pow(x1 + x2 * x2 - 7, 2);

                return result;
            }

            @Override
            public int dimensionOfDomain() {
                return 2;
            }

            @Override
            public int dimensionOfRange() {
                return 1;
            }
        };

        RealVectorFunction g = new RealVectorFunction() {
            @Override
            public Vector evaluate(Vector x) {
                double x1 = x.get(1);
                double x2 = x.get(2);

                double w1 = x1 * x1 + x2 - 11;
                double w2 = x1 + x2 * x2 - 7;

                double[] result = new double[2];
                result[0] = 4 * w1 * x1 + 2 * w2;
                result[1] = 2 * w1 + 4 * w2 * x2;
                return new DenseVector(result);
            }

            @Override
            public int dimensionOfDomain() {
                return 2;
            }

            @Override
            public int dimensionOfRange() {
                return 2;
            }
        };
        C2OptimProblemImpl problem = new C2OptimProblemImpl(f, g);

        ConjugateGradientMinimizer ConjugateGradientMinimizer
                = new ConjugateGradientMinimizer(1e-16, 40);
        IterativeSolution<Vector> soln1 = ConjugateGradientMinimizer.solve(problem);
        Vector xmin1 = soln1.search(new DenseVector(new double[]{6, 6}));
        double f_xmin1 = f.evaluate(xmin1);
        System.out.println(String.format("f(%s) = %.16f", xmin1.toString(), f_xmin1));

        ConjugateGradientMinimizer fletcherReevesMinimizer
                = new FletcherReevesMinimizer(1e-16, 20);
        IterativeSolution<Vector> soln2 = fletcherReevesMinimizer.solve(problem);
        Vector xmin2 = soln2.search(new DenseVector(new double[]{6, 6}));
        double f_xmin2 = f.evaluate(xmin2);
        System.out.println(String.format("f(%s) = %.16f", xmin2.toString(), f_xmin2));

        SteepestDescentMinimizer powellMinimizer
                = new PowellMinimizer(1e-16, 20);
        IterativeSolution<Vector> soln3 = powellMinimizer.solve(problem);
        Vector xmin3 = soln3.search(new DenseVector(new double[]{6, 6}));
        double f_xmin3 = f.evaluate(xmin3);
        System.out.println(String.format("f(%s) = %.16f", xmin3.toString(), f_xmin3));

        SteepestDescentMinimizer zangwillMinimizer
                = new ZangwillMinimizer(1e-16, 1e-16, 20);
        IterativeSolution<Vector> soln4 = zangwillMinimizer.solve(problem);
        Vector xmin4 = soln4.search(new DenseVector(new double[]{6, 6}));
        double f_xmin4 = f.evaluate(xmin4);
        System.out.println(String.format("f(%s) = %.16f", xmin4.toString(), f_xmin4));
    }

    public void solve_by_quasi_Newton() throws Exception {
        System.out.println("solve multivariate function by quasi-Newton");

        /**
         * The Himmelblau function: f(x) = (x1^2 + x2 - 11)^2 + (x1 + x2^2 -
         * 7)^2
         */
        RealScalarFunction f = new RealScalarFunction() {
            @Override
            public Double evaluate(Vector x) {
                double x1 = x.get(1);
                double x2 = x.get(2);

                double result = pow(x1 * x1 + x2 - 11, 2);
                result += pow(x1 + x2 * x2 - 7, 2);

                return result;
            }

            @Override
            public int dimensionOfDomain() {
                return 2;
            }

            @Override
            public int dimensionOfRange() {
                return 1;
            }
        };

        RealVectorFunction g = new RealVectorFunction() {
            @Override
            public Vector evaluate(Vector x) {
                double x1 = x.get(1);
                double x2 = x.get(2);

                double w1 = x1 * x1 + x2 - 11;
                double w2 = x1 + x2 * x2 - 7;

                double[] result = new double[2];
                result[0] = 4 * w1 * x1 + 2 * w2;
                result[1] = 2 * w1 + 4 * w2 * x2;
                return new DenseVector(result);
            }

            @Override
            public int dimensionOfDomain() {
                return 2;
            }

            @Override
            public int dimensionOfRange() {
                return 2;
            }
        };
        C2OptimProblemImpl problem = new C2OptimProblemImpl(f, g);

        QuasiNewtonMinimizer rankOneMinimizer = new RankOneMinimizer(1e-16, 15);
        IterativeSolution<Vector> soln1 = rankOneMinimizer.solve(problem);
        Vector xmin = soln1.search(new DenseVector(new double[]{6, 6}));
        double f_xmin = f.evaluate(xmin);
        System.out.println(String.format("f(%s) = %.16f", xmin.toString(), f_xmin));

        QuasiNewtonMinimizer dfpMinimizer = new DFPMinimizer(1e-16, 15);
        IterativeSolution<Vector> soln2 = dfpMinimizer.solve(problem);
        xmin = soln2.search(new DenseVector(new double[]{6, 6}));
        f_xmin = f.evaluate(xmin);
        System.out.println(String.format("f(%s) = %.16f", xmin.toString(), f_xmin));

        QuasiNewtonMinimizer bfgsMinimizer = new BFGSMinimizer(false, 1e-16, 15);
        IterativeSolution<Vector> soln3 = bfgsMinimizer.solve(problem);
        xmin = soln3.search(new DenseVector(new double[]{6, 6}));
        f_xmin = f.evaluate(xmin);
        System.out.println(String.format("f(%s) = %.16f", xmin.toString(), f_xmin));

        QuasiNewtonMinimizer huangMinimizer = new HuangMinimizer(0, 1, 0, 1, 1e-16, 15);
        IterativeSolution<Vector> soln4 = huangMinimizer.solve(problem);
        xmin = soln4.search(new DenseVector(new double[]{6, 6}));
        f_xmin = f.evaluate(xmin);
        System.out.println(String.format("f(%s) = %.16f", xmin.toString(), f_xmin));

        QuasiNewtonMinimizer pearsonMinimizer = new PearsonMinimizer(1e-16, 15);
        IterativeSolution<Vector> soln5 = pearsonMinimizer.solve(problem);
        xmin = soln5.search(new DenseVector(new double[]{6, 6}));
        f_xmin = f.evaluate(xmin);
        System.out.println(String.format("f(%s) = %.16f", xmin.toString(), f_xmin));
    }

}
