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
import dev.nm.algebra.linear.vector.doubles.dense.DenseVector;
import dev.nm.analysis.function.rn2r1.AbstractBivariateRealFunction;
import dev.nm.analysis.function.rn2r1.AbstractRealScalarFunction;
import dev.nm.analysis.function.rn2r1.RealScalarFunction;
import dev.nm.misc.algorithm.stopcondition.AfterIterations;
import dev.nm.misc.algorithm.stopcondition.StopCondition;
import dev.nm.solver.IterativeSolution;
import dev.nm.solver.multivariate.constrained.constraint.general.GeneralEqualityConstraints;
import dev.nm.solver.multivariate.constrained.constraint.general.GeneralLessThanConstraints;
import dev.nm.solver.multivariate.constrained.general.penaltymethod.PenaltyMethodMinimizer;
import dev.nm.solver.multivariate.constrained.integer.IPProblem;
import dev.nm.solver.multivariate.constrained.integer.IPProblemImpl1;
import dev.nm.solver.multivariate.constrained.problem.ConstrainedOptimProblemImpl1;
import dev.nm.solver.multivariate.geneticalgorithm.minimizer.deoptim.DEOptim;
import dev.nm.solver.multivariate.geneticalgorithm.minimizer.deoptim.Rand1Bin;
import dev.nm.solver.multivariate.geneticalgorithm.minimizer.deoptim.constrained.IntegralConstrainedCellFactory;
import dev.nm.solver.multivariate.geneticalgorithm.minimizer.simplegrid.SimpleCellFactory;
import dev.nm.solver.multivariate.geneticalgorithm.minimizer.simplegrid.SimpleGridMinimizer;
import dev.nm.solver.multivariate.unconstrained.IterativeMinimizer;
import dev.nm.solver.multivariate.unconstrained.annealing.GeneralizedSimulatedAnnealingMinimizer;
import dev.nm.solver.multivariate.unconstrained.c2.quasinewton.BFGSMinimizer;
import dev.nm.solver.problem.C2OptimProblemImpl;
import dev.nm.solver.problem.OptimProblem;
import dev.nm.stat.random.rng.univariate.RandomLongGenerator;
import dev.nm.stat.random.rng.univariate.uniform.UniformRNG;

/**
 * Numerical Methods Using Java: For Data Science, Analysis, and Engineering
 *
 * @author haksunli
 * @see
 * https://www.amazon.com/Numerical-Methods-Using-Java-Engineering/dp/1484267966
 * https://nm.dev/
 */
public class Chapter11 {

    public static void main(String[] args) throws Exception {
        System.out.println("Chapter 11 demos");

        Chapter11 chapter11 = new Chapter11();
        chapter11.penalty_method();
        chapter11.genetic_algorithm();
        chapter11.differential_evolution();
        chapter11.simulated_annealing();
    }

    public void simulated_annealing() throws Exception {
        System.out.println("simulated annealing");

        // the objective function to minimize
        final RealScalarFunction f
                = new AbstractRealScalarFunction(2) {

            @Override
            public Double evaluate(Vector x) {
                double x1 = x.get(1);
                double x2 = x.get(2);
                // (4 - 2.1*x(1)^2 + x(1)^4/3)*x(1)^2
                double term1
                        = (4.0 - 2.1 * Math.pow(x1, 2.0) + Math.pow(x1, 4.0) / 3.0)
                        * Math.pow(x1, 2.0);
                // x(1)*x(2)
                double term2 = x1 * x2;
                // (-4 + 4*x(2)^2)*x(2)^2
                double term3 = (-4.0 + 4.0 * Math.pow(x2, 2.0)) * Math.pow(x2, 2.0);
                return term1 + term2 + term3;
            }
        };

        // construct an optimization problem
        final OptimProblem problem = new OptimProblem() {

            @Override
            public RealScalarFunction f() {
                return f;
            }

            @Override
            public int dimension() {
                return 2;
            }
        };

        // stop after 5000 iterations
        StopCondition stopCondition = new AfterIterations(5000);
        // an instance of a simulated annealing solver
        IterativeMinimizer<OptimProblem> solver
                = new GeneralizedSimulatedAnnealingMinimizer(
                        2, // dimension of the objective function
                        stopCondition
                );
        IterativeSolution<Vector> soln = solver.solve(problem);
        Vector x0 = new DenseVector(0.5, 0.5); // the initial guess
        Vector xmin = soln.search(x0);
        double fxmin = f.evaluate(xmin); // the mimimum
        System.out.println(String.format("f(%s) = %f", xmin, fxmin));
    }

    public void differential_evolution() throws Exception {
        System.out.println("Differential Evolution");

        // the objective function to minimize
        RealScalarFunction f
                = new AbstractBivariateRealFunction() {
            @Override
            public double evaluate(double x, double y) {
                return (x - 1) * (x - 1) + (y - 3) * (y - 3); // (x - 1)^2 + (y - 3)^2
            }
        };

        // construct an integer programming problem
        final IPProblem problem
                // both x and y need to be integers
                = new IPProblemImpl1(f, null, null, new int[]{1, 2}, 0);

        // a uniform random number generator
        final RandomLongGenerator rng = new UniformRNG();
        rng.seed(123456798L);

        // construct an instance of a genetic algorithm solver
        DEOptim solver = new DEOptim(
                () -> new IntegralConstrainedCellFactory(
                        new Rand1Bin(0.5, 0.5, rng), // the DE operator
                        new IntegralConstrainedCellFactory.SomeIntegers(problem)), // specify the integral constraints
                rng, // a uniform random number generator
                1e-15, // a precision parameter
                100, // the maximum number of iterations
                20 // the maximum number of iterations of no improvement
        );

        IterativeSolution<Vector> soln = solver.solve(problem);
        Vector xmin = soln.search(new Vector[]{
            // the boundaries: [-10, 10], [-10, 10]
            new DenseVector(-10.0, 10.0),
            new DenseVector(10.0, -10.0),
            new DenseVector(10.0, 10.0),
            new DenseVector(-10.0, -10.0)
        });
        double fxmin = f.evaluate(xmin); // the mimimum
        System.out.println(String.format("f(%s) = %f", xmin, fxmin));
    }

    public void genetic_algorithm() throws Exception {
        System.out.println("genetic algorithm");

        // the objective function to minimize
        RealScalarFunction f
                = new AbstractBivariateRealFunction() {
            @Override
            public double evaluate(double x, double y) {
                return x * x + y * y; // x^2 + y^2
            }
        };

        // a uniform random number generator
        RandomLongGenerator rng = new UniformRNG();
        rng.seed(123456798L);

        // construct an instance of the genetic algorithm solver
        SimpleGridMinimizer solver
                = new SimpleGridMinimizer(
                        // define the encodinng, crossover and mutation operator
                        () -> new SimpleCellFactory(
                                0.1, // the convergence rate
                                rng),
                        rng, // source of randomness for the GA
                        1e-15, // a precision parameter
                        500, // the maximum number of iterations
                        500 // the maximum number of iterations of no improvement
                );

        // run the solver to solve the optimization problem
        IterativeSolution<Vector> soln
                = solver.solve(new C2OptimProblemImpl(f));
        Vector xmin = soln.search(new Vector[]{ // the minimizer
            // the boundaries: [-10, 10], [-10, 10]
            new DenseVector(-10.0, 10.0),
            new DenseVector(10.0, -10.0),
            new DenseVector(10.0, 10.0),
            new DenseVector(-10.0, -10.0)
        });
        double fxmin = f.evaluate(xmin); // the mimimum
        System.out.println(String.format("f(%s) = %f", xmin, fxmin));
    }

    public void penalty_method() throws Exception {
        System.out.println("penalty method");

        RealScalarFunction f = new AbstractBivariateRealFunction() {
            @Override
            public double evaluate(double x, double y) {
                // f = (x+1)^2 + (y+1)^2
                return (x + 1) * (x + 1) + (y + 1) * (y + 1);
            }
        };

        RealScalarFunction c1 = new AbstractBivariateRealFunction() {
            @Override
            public double evaluate(double x, double y) {
                // y = 0
                return y;
            }
        };

        RealScalarFunction c2 = new AbstractBivariateRealFunction() {
            @Override
            public double evaluate(double x, double y) {
                // x >= 1
                return 1 - x;
            }
        };

        ConstrainedOptimProblemImpl1 problem
                = new ConstrainedOptimProblemImpl1(
                        f,
                        new GeneralEqualityConstraints(c1), // y = 0
                        new GeneralLessThanConstraints(c2)); // x >= 1

        double M = 1e30; // the penalty factor
        PenaltyMethodMinimizer optim
                = new PenaltyMethodMinimizer(
                        PenaltyMethodMinimizer.DEFAULT_PENALTY_FUNCTION_FACTORY,
                        M,
                        // the solver to solve the equivalent unconstrained optimization problem
                        new BFGSMinimizer(false, 1e-8, 200)
                );
        IterativeSolution<Vector> soln = optim.solve(problem);

        Vector xmin = soln.search( // the minimizer
                new DenseVector(new double[]{0, 0}) // an initial guess
        );
        double fxmin = f.evaluate(xmin); // the mimimum
        System.out.println(String.format("f(%s) = %f", xmin, fxmin));

        // alternatively
        System.out.println(String.format("f(%s) = %f", soln.minimizer(), soln.minimum()));
    }
}
