/*
 * Copyright (c) Numerical Method Inc.
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
import dev.nm.algebra.linear.matrix.doubles.factorization.svd.SVD;
import dev.nm.algebra.linear.matrix.doubles.matrixtype.dense.DenseMatrix;
import dev.nm.algebra.linear.matrix.doubles.matrixtype.dense.triangle.SymmetricMatrix;
import dev.nm.algebra.linear.matrix.doubles.operation.MatrixMeasure;
import dev.nm.algebra.linear.vector.doubles.Vector;
import dev.nm.algebra.linear.vector.doubles.dense.DenseVector;
import dev.nm.analysis.function.rn2r1.AbstractBivariateRealFunction;
import dev.nm.analysis.function.rn2r1.AbstractRealScalarFunction;
import dev.nm.analysis.function.rn2r1.QuadraticFunction;
import dev.nm.analysis.function.rn2r1.RealScalarFunction;
import dev.nm.misc.PrecisionUtils;
import dev.nm.solver.IterativeSolution;
import dev.nm.solver.multivariate.constrained.constraint.EqualityConstraints;
import dev.nm.solver.multivariate.constrained.constraint.GreaterThanConstraints;
import dev.nm.solver.multivariate.constrained.constraint.general.GeneralEqualityConstraints;
import dev.nm.solver.multivariate.constrained.constraint.general.GeneralGreaterThanConstraints;
import dev.nm.solver.multivariate.constrained.constraint.general.GeneralLessThanConstraints;
import dev.nm.solver.multivariate.constrained.constraint.linear.BoxConstraints;
import dev.nm.solver.multivariate.constrained.constraint.linear.LinearEqualityConstraints;
import dev.nm.solver.multivariate.constrained.constraint.linear.LinearGreaterThanConstraints;
import dev.nm.solver.multivariate.constrained.constraint.linear.LinearLessThanConstraints;
import dev.nm.solver.multivariate.constrained.constraint.linear.LowerBoundConstraints;
import dev.nm.solver.multivariate.constrained.constraint.linear.NonNegativityConstraints;
import dev.nm.solver.multivariate.constrained.convex.sdp.pathfollowing.CentralPath;
import dev.nm.solver.multivariate.constrained.convex.sdp.pathfollowing.PrimalDualPathFollowingMinimizer;
import dev.nm.solver.multivariate.constrained.convex.sdp.problem.SDPDualProblem;
import dev.nm.solver.multivariate.constrained.convex.sdp.problem.SDPPrimalProblem;
import dev.nm.solver.multivariate.constrained.convex.sdp.socp.interiorpoint.PrimalDualInteriorPointMinimizer;
import dev.nm.solver.multivariate.constrained.convex.sdp.socp.interiorpoint.PrimalDualSolution;
import dev.nm.solver.multivariate.constrained.convex.sdp.socp.problem.SOCPGeneralConstraint;
import dev.nm.solver.multivariate.constrained.convex.sdp.socp.problem.SOCPGeneralProblem;
import dev.nm.solver.multivariate.constrained.convex.sdp.socp.qp.QPSimpleMinimizer;
import dev.nm.solver.multivariate.constrained.convex.sdp.socp.qp.QPSolution;
import dev.nm.solver.multivariate.constrained.convex.sdp.socp.qp.lp.LPMinimizer;
import dev.nm.solver.multivariate.constrained.convex.sdp.socp.qp.lp.problem.LPCanonicalProblem1;
import dev.nm.solver.multivariate.constrained.convex.sdp.socp.qp.lp.problem.LPCanonicalProblem2;
import dev.nm.solver.multivariate.constrained.convex.sdp.socp.qp.lp.problem.LPProblemImpl1;
import dev.nm.solver.multivariate.constrained.convex.sdp.socp.qp.lp.problem.LPStandardProblem;
import dev.nm.solver.multivariate.constrained.convex.sdp.socp.qp.lp.simplex.FerrisMangasarianWrightPhase1;
import dev.nm.solver.multivariate.constrained.convex.sdp.socp.qp.lp.simplex.SimplexTable;
import dev.nm.solver.multivariate.constrained.convex.sdp.socp.qp.lp.simplex.solution.LPBoundedMinimizer;
import dev.nm.solver.multivariate.constrained.convex.sdp.socp.qp.lp.simplex.solution.LPUnboundedMinimizer;
import dev.nm.solver.multivariate.constrained.convex.sdp.socp.qp.lp.simplex.solver.LPRevisedSimplexSolver;
import dev.nm.solver.multivariate.constrained.convex.sdp.socp.qp.lp.simplex.solver.LPTwoPhaseSolver;
import dev.nm.solver.multivariate.constrained.convex.sdp.socp.qp.problem.QPProblem;
import dev.nm.solver.multivariate.constrained.convex.sdp.socp.qp.solver.activeset.QPDualActiveSetMinimizer;
import dev.nm.solver.multivariate.constrained.convex.sdp.socp.qp.solver.activeset.QPPrimalActiveSetMinimizer;
import dev.nm.solver.multivariate.constrained.general.sqp.activeset.SQPActiveSetOnlyInequalityConstraintMinimizer;
import dev.nm.solver.multivariate.constrained.general.sqp.activeset.equalityconstraint.SQPASEVariation2;
import dev.nm.solver.multivariate.constrained.general.sqp.activeset.equalityconstraint.SQPActiveSetOnlyEqualityConstraint1Minimizer;
import static java.lang.Math.pow;
import java.util.Arrays;

/**
 * Numerical Methods Using Java: For Data Science, Analysis, and Engineering
 *
 * @author haksunli
 * @see
 * https://www.amazon.com/Numerical-Methods-Using-Java-Engineering/dp/1484267966
 * https://nm.dev/
 */
public class Chapter10 {

    public static void main(String[] args) throws Exception {
        System.out.println("Chapter 10 demos");

        Chapter10 chapter10 = new Chapter10();

        chapter10.equality_constraints();
        chapter10.inequality_constraints();
        chapter10.lp_problems();
        chapter10.example_3_1_1();
        chapter10.phase1();
        chapter10.lp_solver_1();
        chapter10.lp_solver_2();
        chapter10.lp_solver_3();
        chapter10.lp_solver_4();
        chapter10.lp_solver_5();
        chapter10.qp_solver_1();
        chapter10.qp_solver_2();
        chapter10.sdp_problems();
        chapter10.sdp_solver_1();
        chapter10.socp_solver_1();
        chapter10.sqp_solver_1();
        chapter10.sqp_solver_2();
    }

    /**
     * example 15.2 in
     * Andreas Antoniou, Wu-Sheng Lu
     */
    public void sqp_solver_2() throws Exception {
        System.out.println("solving an SQP problem");

        // objective function
        RealScalarFunction f = new RealScalarFunction() {
            @Override
            public Double evaluate(Vector x) {
                double x1 = x.get(1);
                double x2 = x.get(2);
                double x3 = x.get(3);
                double x4 = x.get(4);

                double fx = (x1 - x3) * (x1 - x3);
                fx += (x2 - x4) * (x2 - x4);
                fx /= 2;

                return fx;
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

        // inequality constraints
        GreaterThanConstraints greater
                = new GeneralGreaterThanConstraints(
                        // c1
                        new RealScalarFunction() {
                    @Override
                    public Double evaluate(Vector x) {
                        double x1 = x.get(1);
                        double x2 = x.get(2);
                        double x3 = x.get(3);
                        double x4 = x.get(4);

                        Matrix x12
                                = new DenseMatrix(
                                        new double[]{
                                            x1, x2},
                                        2, 1);

                        Matrix A
                                = new DenseMatrix(
                                        new double[][]{
                                            {0.25, 0},
                                            {0, 1}
                                        });
                        Matrix B = new DenseMatrix(
                                new double[]{
                                    0.5, 0},
                                2, 1);

                        Matrix FX = x12.t().multiply(A).multiply(x12);
                        FX = FX.scaled(-1);
                        FX = FX.add(x12.t().multiply(B));

                        double fx = FX.get(1, 1);
                        fx += 0.75;

                        return fx;
                    }

                    @Override
                    public int dimensionOfDomain() {
                        return 4;
                    }

                    @Override
                    public int dimensionOfRange() {
                        return 1;
                    }
                },
                        // c2
                        new RealScalarFunction() {
                    @Override
                    public Double evaluate(Vector x) {
                        double x1 = x.get(1);
                        double x2 = x.get(2);
                        double x3 = x.get(3);
                        double x4 = x.get(4);

                        Matrix x34
                                = new DenseMatrix(
                                        new double[]{
                                            x3, x4},
                                        2, 1);

                        Matrix A
                                = new DenseMatrix(
                                        new double[][]{
                                            {5, 3},
                                            {3, 5}
                                        });
                        Matrix B
                                = new DenseMatrix(
                                        new double[]{
                                            11. / 2, 13. / 2},
                                        2, 1);

                        Matrix FX = x34.t().multiply(A).multiply(x34);
                        FX = FX.scaled(-1. / 8);
                        FX = FX.add(x34.t().multiply(B));

                        double fx = FX.get(1, 1);
                        fx += -35. / 2;

                        return fx;
                    }

                    @Override
                    public int dimensionOfDomain() {
                        return 4;
                    }

                    @Override
                    public int dimensionOfRange() {
                        return 1;
                    }
                });

        /**
         * TODO: making the 2nd precision parameter 0 gives a better minimizer;
         * how to choose the precision parameters in general?
         */
        // construct an SQP solver
        SQPActiveSetOnlyInequalityConstraintMinimizer solver
                = new SQPActiveSetOnlyInequalityConstraintMinimizer(
                        1e-7, // epsilon1
                        1e-3, // epsilon2
                        10 // max number of iterations
                );
        // solving the SQP problem
        IterativeSolution<Vector> solution = solver.solve(f, greater);
        Vector x = solution.search(
                new DenseVector(1., 0.5, 2., 3.), // x0
                new DenseVector(1., 1.)); // μ0
        double fx = f.evaluate(x);
        // print out the solution
        System.out.println("x = " + x);
        System.out.println("fx = " + fx);
    }

    /**
     * example 15.1 in
     * Andreas Antoniou, Wu-Sheng Lu
     */
    public void sqp_solver_1() throws Exception {
        System.out.println("solving an SQP problem with only equality constraints");

        // objective function
        RealScalarFunction f = new RealScalarFunction() {
            @Override
            public Double evaluate(Vector x) {
                double x1 = x.get(1);
                double x2 = x.get(2);
                double x3 = x.get(3);

                double fx = -pow(x1, 4.);
                fx -= 2. * pow(x2, 4.);
                fx -= pow(x3, 4.);
                fx -= pow(x1 * x2, 2.);
                fx -= pow(x1 * x3, 2.);

                return fx;
            }

            @Override
            public int dimensionOfDomain() {
                return 3;
            }

            @Override
            public int dimensionOfRange() {
                return 1;
            }
        };

        // equality constraints
        EqualityConstraints equality_constraints
                = new GeneralEqualityConstraints(
                        new RealScalarFunction() {
                    @Override
                    public Double evaluate(Vector x) {
                        double x1 = x.get(1);
                        double x2 = x.get(2);
                        double x3 = x.get(3);

                        double fx = pow(x1, 4.);
                        fx += pow(x2, 4.);
                        fx += pow(x3, 4.);
                        fx -= 25.;

                        return fx; // a1
                    }

                    @Override
                    public int dimensionOfDomain() {
                        return 3;
                    }

                    @Override
                    public int dimensionOfRange() {
                        return 1;
                    }
                },
                        new RealScalarFunction() {
                    @Override
                    public Double evaluate(Vector x) {
                        double x1 = x.get(1);
                        double x2 = x.get(2);
                        double x3 = x.get(3);

                        double fx = 8. * pow(x1, 2.);
                        fx += 14. * pow(x2, 2.);
                        fx += 7. * pow(x3, 2.);
                        fx -= 56.;

                        return fx; // a2
                    }

                    @Override
                    public int dimensionOfDomain() {
                        return 3;
                    }

                    @Override
                    public int dimensionOfRange() {
                        return 1;
                    }
                });

        // construct an SQP solver
        SQPActiveSetOnlyEqualityConstraint1Minimizer solver
                = new SQPActiveSetOnlyEqualityConstraint1Minimizer(
                        (RealScalarFunction f1, EqualityConstraints equal) -> {
                            SQPASEVariation2 impl = new SQPASEVariation2(100., 0.01, 10);
                            impl.set(f1, equal);
                            return impl;
                        },
                        1e-8, // epsilon, threshold
                        20); // max number of iterations
        // solving an SQP problem
        IterativeSolution<Vector> solution
                = solver.solve(f, equality_constraints);
        Vector x = solution.search(
                new DenseVector(3., 1.5, 3.), // x0
                new DenseVector(-1., -1.)); // λ0
        double fx = f.evaluate(x);
        // print out the solution
        System.out.println("x = " + x);
        System.out.println("fx = " + fx);
    }

    /**
     * example 14.5 in
     * Andreas Antoniou, Wu-Sheng Lu
     */
    public void socp_solver_1() throws Exception {
        System.out.println("solving an SOCP problem");

        // the problem specifications
        Vector b = new DenseVector(1., 0., 0., 0., 0.);

        Matrix A1t
                = new DenseMatrix(
                        new double[][]{
                            {0, -1, 0, 1, 0},
                            {0, 0, 1, 0, -1}
                        });
        Matrix A2t
                = new DenseMatrix(
                        new double[][]{
                            {0, 0.5, 0, 0, 0},
                            {0, 0, 1, 0, 0}
                        });
        Matrix A3t
                = new DenseMatrix(
                        new double[][]{
                            {0, 0, 0, -0.7071, -0.7071},
                            {0, 0, 0, -0.3536, 0.3536}
                        });

        Vector b1 = b;
        Vector b2 = b.ZERO();
        Vector b3 = b.ZERO();

        Vector c1 = new DenseVector(2);//zero
        Vector c2 = new DenseVector(-0.5, 0.);
        Vector c3 = new DenseVector(4.2426, -0.7071);

        double[] d = new double[]{0., 1, 1};

        // construct an SOCP problem in standard form
        SOCPGeneralProblem problem
                = new SOCPGeneralProblem(
                        b,
                        Arrays.asList(
                                new SOCPGeneralConstraint(A1t.t(), c1, b1, d[0]),
                                new SOCPGeneralConstraint(A2t.t(), c2, b2, d[1]),
                                new SOCPGeneralConstraint(A3t.t(), c3, b3, d[2])
                        )
                );

        // an initial strictly feasible point
        Vector x0 = new DenseVector(1, 0, 0, 0.1, 0, 0, 0.1, 0, 0);
        Vector s0 = new DenseVector(3.7, 1, -3.5, 1, 0.25, 0.5, 1, -0.35355, -0.1767);
        Vector y0 = new DenseVector(-3.7, -1.5, -0.5, -2.5, -4);
        PrimalDualSolution initial = new PrimalDualSolution(x0, s0, y0);

        // solving an SOCP problem
        PrimalDualInteriorPointMinimizer solver
                = new PrimalDualInteriorPointMinimizer(0.00001, 20);
        IterativeSolution<PrimalDualSolution> solution
                = solver.solve(problem);
        solution.search(initial);

        // primal solution
        System.out.println("X = ");
        System.out.println(solution.minimizer().x);
        // dual solution
        System.out.println("y = ");
        System.out.println(solution.minimizer().y);
        System.out.println("S = ");
        System.out.println(solution.minimizer().s);
        System.out.println("minimum = " + solution.minimum());
    }

    /**
     * p.465 in
     * Andreas Antoniou, Wu-Sheng Lu
     */
    public void sdp_solver_1() throws Exception {
        System.out.println("solving an SDP problem");

        // define an SDP problem with matrices and vectors
        SymmetricMatrix A0
                = new SymmetricMatrix(
                        new double[][]{
                            {2},
                            {-0.5, 2},
                            {-0.6, 0.4, 3}
                        });
        SymmetricMatrix A1
                = new SymmetricMatrix(
                        new double[][]{
                            {0},
                            {1, 0},
                            {0, 0, 0}
                        });
        SymmetricMatrix A2
                = new SymmetricMatrix(
                        new double[][]{
                            {0},
                            {0, 0},
                            {1, 0, 0}
                        });
        SymmetricMatrix A3
                = new SymmetricMatrix(
                        new double[][]{
                            {0},
                            {0, 0},
                            {0, 1, 0}
                        });
        SymmetricMatrix A4 = A3.ONE();
        SymmetricMatrix C = A0.scaled(-1.);
        Vector b = new DenseVector(0., 0., 0., 1.);
        // construct an SDP problem
        SDPDualProblem problem
                = new SDPDualProblem(
                        b,
                        C,
                        new SymmetricMatrix[]{A1, A2, A3, A4});

        // the initial feasible point
        DenseMatrix X0
                = new DenseMatrix(
                        new double[][]{
                            {1. / 3., 0., 0.},
                            {0., 1. / 3., 0.},
                            {0., 0., 1. / 3.}
                        });
        Vector y0 = new DenseVector(0.2, 0.2, 0.2, -4.);
        DenseMatrix S0
                = new DenseMatrix(
                        new double[][]{
                            {2, 0.3, 0.4},
                            {0.3, 2, -0.6},
                            {0.4, -0.6, 1}
                        });
        // the initial central path
        CentralPath path0 = new CentralPath(X0, y0, S0);

        // solving SDP problem
        PrimalDualPathFollowingMinimizer solver
                = new PrimalDualPathFollowingMinimizer(
                        0.9, // γ
                        0.001); // ε
        IterativeSolution<CentralPath> solution = solver.solve(problem);
        CentralPath path = solution.search(path0);

        //the solution from the textbook is accurate up to epsilon
        //changing epsilon will change the answers
        // primal solution
        System.out.println("X = ");
        System.out.println(path.X);
        // dual solution
        System.out.println("y = ");
        System.out.println(path.y);
        System.out.println("S = ");
        System.out.println(path.S);
    }

    public void sdp_problems() {
        System.out.println("construct the primal and dual SDP problems");

        // the primal SDP matrices
        SymmetricMatrix C
                = new SymmetricMatrix(
                        new double[][]{
                            {1},
                            {2, 9},
                            {3, 0, 7}
                        });
        SymmetricMatrix A1
                = new SymmetricMatrix(
                        new double[][]{
                            {1},
                            {0, 3},
                            {1, 7, 5}
                        });
        SymmetricMatrix A2
                = new SymmetricMatrix(
                        new double[][]{
                            {0},
                            {2, 6},
                            {8, 0, 4}
                        });

        // construct the primal SDP problem
        SDPPrimalProblem primal
                = new SDPPrimalProblem(
                        C,
                        new SymmetricMatrix[]{A1, A2}
                );

        // the dual SDP vector and matrices
        Vector b = new DenseVector(11., 19.);
        // construct the primal SDP problem
        SDPDualProblem dual
                = new SDPDualProblem(
                        b,
                        C,
                        new SymmetricMatrix[]{A1, A2}
                );
    }

    /**
     * example 16.4 in Jorge Nocedal, Stephen Wright
     *
     * There is a detailed trace (for debugging) on p. 475.
     */
    public void qp_solver_2() throws Exception {
        System.out.println("solving an QP problem");

        // construct a quadratic function
        Matrix H = new DenseMatrix(new double[][]{
            {2, 0},
            {0, 2}
        });
        Vector p = new DenseVector(new double[]{-2, -5});
        QuadraticFunction f = new QuadraticFunction(H, p);

        // construct the linear inequality constraints
        Matrix A = new DenseMatrix(new double[][]{
            {1, -2},
            {-1, -2},
            {-1, 2},
            {1, 0},
            {0, 1}
        });
        Vector b = new DenseVector(new double[]{-2, -6, -2, 0, 0});
        LinearGreaterThanConstraints greater
                = new LinearGreaterThanConstraints(A, b);// x >= 0
        // construct the QP problem
        QPProblem problem = new QPProblem(f, null, greater);

        // construct a primal active set solver
        double epsion = Math.sqrt(PrecisionUtils.autoEpsilon(problem.f().Hessian()));
        QPPrimalActiveSetMinimizer solver1
                = new QPPrimalActiveSetMinimizer(
                        epsion, // precision
                        Integer.MAX_VALUE // max number of iterations
                );
        // solve the QP problem using the primal active set method
        QPPrimalActiveSetMinimizer.Solution solution1 = solver1.solve(problem);
        solution1.search(new DenseVector(2., 0.));
        // print out the solution
        System.out.println("minimizer = " + solution1.minimizer().minimizer());
        System.out.println("minimum = " + solution1.minimum());

        // solve the QP problem using the dual active set method
        QPDualActiveSetMinimizer solver2
                = new QPDualActiveSetMinimizer(
                        epsion, // precision
                        Integer.MAX_VALUE); // max number of iterations
        QPDualActiveSetMinimizer.Solution solution2 = solver2.solve(problem);
        solution2.search();
        // print out the solution
        System.out.println("minimizer = " + solution2.minimizer().minimizer());
        System.out.println("minimum = " + solution2.minimum());
    }

    /**
     * example 13.1 in
     * Andreas Antoniou, Wu-Sheng Lu, "Algorithm 13.1, Quadratic and Convex
     * Programming," Practical Optimization: Algorithms and Engineering
     * Applications.
     *
     * @throws Exception
     */
    public void qp_solver_1() throws Exception {
        System.out.println("solving an QP problem with only equality constraints");

        //construct the QP problem with only equality constraints
        DenseMatrix H = new DenseMatrix(
                new double[][]{
                    {1, 0, 0},
                    {0, 1, 0},
                    {0, 0, 0}
                });
        DenseVector p = new DenseVector(2, 1, -1);
        QuadraticFunction f = new QuadraticFunction(H, p);
        System.out.println("minimizing:");
        System.out.println(f);

        // equality constraints
        DenseMatrix A = new DenseMatrix(
                new double[][]{
                    {0, 1, 1}
                });
        DenseVector b = new DenseVector(1.);
        LinearEqualityConstraints Aeq = new LinearEqualityConstraints(A, b);

        // solve a QP problem with only equality constraints
        QPSolution soln = QPSimpleMinimizer.solve(f, Aeq);
        Vector x = soln.minimizer();
        double fx = f.evaluate(x);
        System.out.printf("f(%s) = %f%n", x, fx);
        System.out.printf("is unique = %b%n", soln.isUnique());
    }

    /**
     * Example 11.1.
     *
     * Applied Integer Programming: Modeling and Solution
     * by Der-San Chen, Robert G. Batson, Yu Dang.
     */
    public void lp_solver_5() throws Exception {
        System.out.println("solving an LP problem");

        // construct an LP problem
        LPProblemImpl1 problem = new LPProblemImpl1(
                new DenseVector(-5.0, 2.0), // c
                new LinearGreaterThanConstraints(
                        new DenseMatrix( // A1
                                new double[][]{
                                    {1.0, 3.0}}),
                        new DenseVector(9.0)), // b1
                new LinearLessThanConstraints(
                        new DenseMatrix( // A2
                                new double[][]{
                                    {-1.0, 2.0},
                                    {3.0, 2.0}}),
                        new DenseVector(5.0, 19.0)), // b2
                null,
                null);

        // solve the LP problem using the algebraic LP solver
        LPRevisedSimplexSolver solver = new LPRevisedSimplexSolver(1e-8);
        LPMinimizer solution = solver.solve(problem).minimizer();

        System.out.printf("minimum = %f%n", solution.minimum());
        System.out.printf("minimizer = %s%n", solution.minimizer());
    }

    /**
     * Example 11.1.
     *
     * Applied Integer Programming: Modeling and Solution
     * by Der-San Chen, Robert G. Batson, Yu Dang.
     */
    public void lp_solver_4() throws Exception {
        System.out.println("solving an LP problem");

        // construct an LP problem
        LPProblemImpl1 problem = new LPProblemImpl1(
                new DenseVector(-5.0, 2.0), // c
                new LinearGreaterThanConstraints(
                        new DenseMatrix( // A1
                                new double[][]{
                                    {1.0, 3.0}}),
                        new DenseVector(9.0)), // b1
                new LinearLessThanConstraints(
                        new DenseMatrix( // A2
                                new double[][]{
                                    {-1.0, 2.0},
                                    {3.0, 2.0}}),
                        new DenseVector(5.0, 19.0)), // b2
                null,
                null);

        // construct the simplex tableau for the LP problem
        SimplexTable table0 = new SimplexTable(problem);
        System.out.println("simplex tableau for the problem:");
        System.out.println(table0);

        // solve the LP problem using the 2-phase algorithm
        LPTwoPhaseSolver solver = new LPTwoPhaseSolver();
        LPMinimizer solution = solver.solve(problem).minimizer();

        System.out.printf("minimum = %f%n", solution.minimum());
        System.out.printf("minimizer = %s%n", solution.minimizer());
    }

    /**
     * Example 3-6-13 (b), pp. 84.
     *
     * Linear Programming with MATLAB
     * by Michael C. Ferris, Olvi L. Mangasarian, Stephen J. Wright.
     *
     * This case is found infeasible during phase 1.
     */
    public void lp_solver_3() throws Exception {
        System.out.println("solving an LP problem");

        // construct an LP problem
        LPProblemImpl1 problem = new LPProblemImpl1(
                new DenseVector(2.0, -1.0), // c
                new LinearGreaterThanConstraints(
                        new DenseMatrix( // A
                                new double[][]{{1.0, 0.0}}),
                        new DenseVector(6.0)), // b
                null,
                new LinearEqualityConstraints(
                        new DenseMatrix(
                                new double[][]{
                                    {-1.0, 0.0}}), // A
                        new DenseVector(-4.0)), // b
                new BoxConstraints(
                        2,
                        new BoxConstraints.Bound(
                                2,
                                Double.NEGATIVE_INFINITY,
                                Double.POSITIVE_INFINITY)) // x2 is free
        );

        // construct the simplex tableau for the LP problem
        SimplexTable table0 = new SimplexTable(problem);
        System.out.println("simplex tableau for the problem:");
        System.out.println(table0);

        // solve the LP problem using the 2-phase algorithm
        LPTwoPhaseSolver solver = new LPTwoPhaseSolver();
        LPMinimizer minimizer = solver.solve(problem).minimizer();
    }

    /**
     * Example 3-6-13 (c), pp. 84.
     *
     * Linear Programming with MATLAB
     * by Michael C. Ferris, Olvi L. Mangasarian, Stephen J. Wright.
     *
     * This case is founded unbound during phase 2.
     */
    public void lp_solver_2() throws Exception {
        System.out.println("solving an LP problem");

        // construct an LP problem
        LPProblemImpl1 problem = new LPProblemImpl1(
                new DenseVector(2.0, -1.0), // c
                new LinearGreaterThanConstraints(
                        new DenseMatrix( // A1
                                new double[][]{
                                    {1.0, 0.0}
                                }),
                        new DenseVector(-6.0)), // b1
                null,
                new LinearEqualityConstraints(
                        new DenseMatrix( // A2
                                new double[][]{
                                    {-1.0, 0.0} // b2
                                }),
                        new DenseVector(-4.0)),
                new BoxConstraints(
                        2,
                        new BoxConstraints.Bound(
                                2,
                                Double.NEGATIVE_INFINITY,
                                Double.POSITIVE_INFINITY)) // x2 is free
        );

        // construct the simplex tableau for the LP problem
        SimplexTable table0 = new SimplexTable(problem);
        System.out.println("simplex tableau for the problem:");
        System.out.println(table0);

        // solve the LP problem using the 2-phase algorithm
        LPTwoPhaseSolver solver = new LPTwoPhaseSolver();
        LPUnboundedMinimizer solution
                = (LPUnboundedMinimizer) solver.solve(problem).minimizer();

        System.out.printf("minimum = %f%n", solution.minimum());
        System.out.printf("minimizer = %s%n", solution.minimizer());
        System.out.printf("v = %s%n", solution.v());
    }

    /**
     * Example 3-4-1.
     *
     * Linear Programming with MATLAB
     * by Michael C. Ferris, Olvi L. Mangasarian, Stephen J. Wright.
     */
    public void lp_solver_1() throws Exception {
        System.out.println("solving an LP problem");

        // construct an LP problem
        LPProblemImpl1 problem = new LPProblemImpl1(
                new DenseVector(4.0, 5.0), // c
                new LinearGreaterThanConstraints(
                        new DenseMatrix( // A
                                new double[][]{
                                    {1.0, 1.0},
                                    {1.0, 2.0},
                                    {4.0, 2.0},
                                    {-1.0, -1.0},
                                    {-1.0, 1.0}
                                }),
                        new DenseVector(-1.0, 1.0, 8.0, -3.0, 1.0)), // b
                null, // less-than constraints
                null, // equality constraints
                null); // box constraints

        // construct the simplex tableau for the LP problem
        SimplexTable table0 = new SimplexTable(problem);
        System.out.println("simplex tableau for the problem:");
        System.out.println(table0);

        // solve the LP problem using the 2-phase algorithm
        LPTwoPhaseSolver solver = new LPTwoPhaseSolver();
        LPBoundedMinimizer solution
                = (LPBoundedMinimizer) solver.solve(problem).minimizer();

        System.out.printf("minimum = %f%n", solution.minimum());
        System.out.printf("minimizer = %s%n", solution.minimizer());
    }

    /**
     * Example 3-4-1.
     *
     * Linear Programming with MATLAB
     * by Michael C. Ferris, Olvi L. Mangasarian, Stephen J. Wright.
     */
    public void phase1() throws Exception {
        System.out.println("Phase 1 procedure");

        // construct an LP problem
        LPProblemImpl1 problem = new LPProblemImpl1(
                new DenseVector(4.0, 5.0), // c
                new LinearGreaterThanConstraints(
                        new DenseMatrix( // A
                                new double[][]{
                                    {1.0, 1.0},
                                    {1.0, 2.0},
                                    {4.0, 2.0},
                                    {-1.0, -1.0},
                                    {-1.0, 1.0}
                                }),
                        new DenseVector(-1.0, 1.0, 8.0, -3.0, 1.0)), // b
                null, // less-than constraints
                null, // equality constraints
                null); // box constraints
        SimplexTable table0 = new SimplexTable(problem);
        System.out.println("tableau for the original problem:");
        System.out.println(table0);

        FerrisMangasarianWrightPhase1 phase1
                = new FerrisMangasarianWrightPhase1(table0);
        SimplexTable table1 = phase1.process();
        System.out.println("tableau for the phase 1 problem:");
        System.out.println(table1);

        System.out.printf("minimum = %f%n", table1.minimum());
        System.out.printf("minimizer = %s%n", table1.minimizer());
    }

    /**
     * Example 3-1-1.
     *
     * Linear Programming with MATLAB
     * by Michael C. Ferris, Olvi L. Mangasarian, Stephen J. Wright.
     */
    public void example_3_1_1() {
        System.out.println("example 3.1.1");

        // construct an LP problem
        LPCanonicalProblem1 problem
                = new LPCanonicalProblem1(
                        new DenseVector(3., -6.), // c 
                        new DenseMatrix( // A
                                new double[][]{
                                    {1, 2},
                                    {2, 1},
                                    {1, -1},
                                    {1, -4},
                                    {-4, 1}
                                }),
                        new DenseVector(-1., 0, -1, -13, -23) // b
                );

        SimplexTable tableau = new SimplexTable(problem);
        System.out.println(tableau);

        tableau = tableau.swap(3, 2);
        System.out.println(tableau);

        tableau = tableau.swap(4, 1);
        System.out.println(tableau);

    }

    public void lp_problems() {
        System.out.println("LP problems of different forms");

        // construct an LP problem in standard form
        LPStandardProblem problem1 = new LPStandardProblem(
                new DenseVector(new double[]{-1.0, -1.0, 0, 0}), // c
                new LinearEqualityConstraints(
                        new DenseMatrix( // A
                                new double[][]{
                                    {7, 1, 1, 0},
                                    {-1, 1, 0, 1}
                                }),
                        new DenseVector(new double[]{15.0, 1.0}) // b
                ));
        System.out.println(problem1);

        // construct an LP problem in canonical form 1
        LPCanonicalProblem1 problem2
                = new LPCanonicalProblem1(
                        new DenseVector(new double[]{-1.0, -1.0, 0, 0}), // c
                        new DenseMatrix( // A
                                new double[][]{
                                    {7, 1, 1, 0},
                                    {-1, 1, 0, 1},
                                    {-7, -1, -1, 0},
                                    {1, -1, 0, -1}
                                }),
                        new DenseVector(new double[]{15.0, 1.0, -15.0, -1.0}) // b
                );
        System.out.println(problem2);

        // construct an LP problem in canonical form 2
        LPCanonicalProblem2 problem3
                = new LPCanonicalProblem2(
                        new DenseVector(new double[]{-1.0, -1.0, 0, 0}), // c
                        new DenseMatrix( // A
                                new double[][]{
                                    {-7, -1, -1, 0},
                                    {1, -1, 0, -1},
                                    {7, 1, 1, 0},
                                    {-1, 1, 0, 1}
                                }),
                        new DenseVector(new double[]{-15.0, -1.0, 15.0, 1.0}) // b
                );
        System.out.println(problem3);

        // construct an LP problem in canonical form 2
        LPCanonicalProblem2 problem4
                = new LPCanonicalProblem2(
                        new DenseVector(new double[]{1.0, 1.0}), // c
                        new DenseMatrix( // A
                                new double[][]{
                                    {1, 1},
                                    {1, 2},
                                    {0, 3}
                                }),
                        new DenseVector(new double[]{150, 170, 180}) // b
                );
        System.out.println(problem4);
    }

    public void inequality_constraints() {
        System.out.println("inequality constraints");

        GeneralGreaterThanConstraints c_gr
                = new GeneralGreaterThanConstraints(
                        new AbstractBivariateRealFunction() {
                    @Override
                    public double evaluate(double x1, double x2) {
                        double c = x2 - x1 * x1;
                        return c;
                    }
                });

        GeneralLessThanConstraints c_less = new GeneralLessThanConstraints(
                new AbstractBivariateRealFunction() {
            @Override
            public double evaluate(double x1, double x2) {
                double c = x1 * x1 - x2;
                return c;
            }
        });

        // w_i >= 0 mean no short selling
        Matrix A = new DenseMatrix(new double[][]{
            {1, 0, 0},
            {0, 1, 0},
            {0, 0, 1}
        });
        Vector b1 = new DenseVector(new double[]{0, 0, 0}); // 3 stocks
        LinearGreaterThanConstraints no_short_selling1
                = new LinearGreaterThanConstraints(A, b1);// w >= 0
        System.out.println(no_short_selling1);

        LowerBoundConstraints no_short_selling2
                = new LowerBoundConstraints(3, 0.);
        System.out.println(no_short_selling2);

        NonNegativityConstraints no_short_selling3
                = new NonNegativityConstraints(3);
        System.out.println(no_short_selling3);

        // w_i <= 0.2
        Vector b2 = new DenseVector(new double[]{0.2, 0.2, 0.2}); // 3 stocks
        LinearLessThanConstraints maximum_exposure
                = new LinearLessThanConstraints(A, b2);// w >= 0
        System.out.println(maximum_exposure);

    }

    public void equality_constraints() {
        System.out.println("equality constraints");

        // non-linear constraints
        GeneralEqualityConstraints a
                = new GeneralEqualityConstraints(
                        // the first equality constraint
                        new AbstractRealScalarFunction(3) { // the domain dimension
                    @Override
                    public Double evaluate(Vector x) {
                        double x1 = x.get(1);
                        double x3 = x.get(3);

                        double a1 = -x1 + x3 + 1;
                        return a1;
                    }
                },
                        // the second equality constraint
                        new AbstractRealScalarFunction(3) { // the domain dimension
                    @Override
                    public Double evaluate(Vector x) {
                        double x1 = x.get(1);
                        double x2 = x.get(2);

                        double a2 = x1 * x1 + x2 * x2 - 2. * x1;
                        return a2;
                    }
                }
                );

        /** Example 10.2, p. 270. Practical Optimization: Algorithms and
         * Engineering Applications. Andreas Antoniou, Wu-Sheng Lu */
        // linear constraints
        Matrix A = new DenseMatrix(new double[][]{
            {1, -2, 3, 2},
            {0, 2, -1, 0},
            {2, -10, 9, 4}
        });
        Vector b = new DenseVector(new double[]{4, 1, 5});
        LinearEqualityConstraints A_eq
                = new LinearEqualityConstraints(A, b);
        System.out.println("original equality constraints: ");
        System.out.println(A_eq);

        // do SVD decomposition to reduce the eqaulity constraints
        SVD svd = new SVD(A, true);
        Matrix U = svd.U();
        System.out.println("U = ");
        System.out.println(U);
        Matrix D = svd.D();
        System.out.println("D = ");
        System.out.println(D);
        Matrix V = svd.V();
        System.out.println("V = ");
        System.out.println(V);

        // check if the original equality constraints are reducible
        double epsilon = 1e-8; // the precision parameter under which is considered 0
        boolean isReducible = A_eq.isReducible(epsilon);
        System.out.println(isReducible);
        int r = MatrixMeasure.rank(
                A,
                epsilon
        );
        System.out.printf("rank of A = %d%n", r);

        // construct a new set of reduced constraints
        LinearEqualityConstraints A_eq_hat
                = A_eq.getReducedLinearEqualityConstraints(epsilon);
        System.out.println("reduced equality constraints: ");
        System.out.println(A_eq_hat);
    }
}
