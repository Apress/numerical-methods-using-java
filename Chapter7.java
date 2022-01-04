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
import dev.nm.analysis.differentialequation.ode.ivp.problem.DerivativeFunction;
import dev.nm.analysis.differentialequation.ode.ivp.problem.ODE1stOrder;
import dev.nm.analysis.differentialequation.ode.ivp.solver.EulerMethod;
import dev.nm.analysis.differentialequation.ode.ivp.solver.ODESolution;
import dev.nm.analysis.differentialequation.ode.ivp.solver.ODESolver;
import dev.nm.analysis.differentialequation.ode.ivp.solver.multistep.adamsbashforthmoulton.ABMPredictorCorrector1;
import dev.nm.analysis.differentialequation.ode.ivp.solver.multistep.adamsbashforthmoulton.ABMPredictorCorrector2;
import dev.nm.analysis.differentialequation.ode.ivp.solver.multistep.adamsbashforthmoulton.ABMPredictorCorrector3;
import dev.nm.analysis.differentialequation.ode.ivp.solver.multistep.adamsbashforthmoulton.ABMPredictorCorrector4;
import dev.nm.analysis.differentialequation.ode.ivp.solver.multistep.adamsbashforthmoulton.ABMPredictorCorrector5;
import dev.nm.analysis.differentialequation.ode.ivp.solver.multistep.adamsbashforthmoulton.AdamsBashforthMoulton;
import dev.nm.analysis.differentialequation.ode.ivp.solver.rungekutta.RungeKutta;
import dev.nm.analysis.differentialequation.ode.ivp.solver.rungekutta.RungeKutta1;
import dev.nm.analysis.differentialequation.ode.ivp.solver.rungekutta.RungeKutta2;
import dev.nm.analysis.differentialequation.ode.ivp.solver.rungekutta.RungeKutta3;
import dev.nm.analysis.differentialequation.ode.ivp.solver.rungekutta.RungeKuttaStepper;
import dev.nm.analysis.function.rn2r1.univariate.AbstractUnivariateRealFunction;
import dev.nm.analysis.function.rn2r1.univariate.UnivariateRealFunction;
import dev.nm.analysis.function.rn2rm.AbstractRealVectorFunction;
import dev.nm.analysis.function.rn2rm.RealVectorFunction;
import java.io.IOException;
import static java.lang.Math.exp;

/**
 * Numerical Methods Using Java: For Data Science, Analysis, and Engineering
 *
 * @author haksunli
 * @see
 * https://www.amazon.com/Numerical-Methods-Using-Java-Engineering/dp/1484267966
 * https://nm.dev/
 */
public class Chapter7 {

    public static void main(String[] args) throws IOException {
        System.out.println("Chapter 7 demos");

        Chapter7 chapter7 = new Chapter7();
        chapter7.EulerMethod();
        chapter7.RungeKutta();
        chapter7.ABM();
        chapter7.system_of_ODEs();
        chapter7.higher_order_ODEs();
    }

    private void higher_order_ODEs() {
        System.out.println("solve a higher-order ODE using Euler's method");

        // define the equivalent system of ODEs to solve
        DerivativeFunction dY = new DerivativeFunction() {

            @Override
            public Vector evaluate(double t, Vector v) {
                double y_1 = v.get(1);
                double y_2 = v.get(2);

                double dy_1 = y_2;
                double dy_2 = y_2 + 6. * y_1;
                return new DenseVector(new double[]{dy_1, dy_2});
            }

            @Override
            public int dimension() {
                return 2;
            }
        };
        // initial condition, y1(0) = 1, y2(0) = 2
        Vector Y0 = new DenseVector(1., 2.);

        double t0 = 0, t1 = 1.; // solution domain
        double h = 0.1; // step size

        // the analytical solution
        UnivariateRealFunction f
                = new AbstractUnivariateRealFunction() {
            @Override
            public double evaluate(double t) {
                double f = 0.8 * exp(3. * t);
                f += 0.2 * exp(-2. * t);
                return f;
            }
        };

        // define an IVP
        ODE1stOrder ivp = new ODE1stOrder(dY, Y0, t0, t1);
        // construt an ODE solver using Euler's method
//        ODESolver solver = new EulerMethod(h);
        // construt an ODE solver using the third order Runge-Kutta formula
        RungeKuttaStepper stepper3 = new RungeKutta3();
        ODESolver solver = new RungeKutta(stepper3, h);

        // solve the ODE
        ODESolution soln = solver.solve(ivp);
        // print out the solution function, y, at discrete points
        double[] t = soln.x();
        Vector[] v = soln.y();
        for (int i = 0; i < t.length; ++i) {
            double y1 = v[i].get(1); // the numerical solution
            System.out.println(String.format(
                    "y(%f) = %f vs %f",
                    t[i],
                    y1,
                    f.evaluate(t[i])
            ));
        }
    }

    private void system_of_ODEs() {
        System.out.println("solve a system of ODEs using Euler's method");

        // define the system of ODEs to solve
        DerivativeFunction dY = new DerivativeFunction() {

            @Override
            public Vector evaluate(double t, Vector v) {
                double x = v.get(1);
                double y = v.get(2);

                double dx = 3. * x - 4. * y;
                double dy = 4. * x - 7. * y;
                return new DenseVector(new double[]{dx, dy});
            }

            @Override
            public int dimension() {
                return 2;
            }
        };
        // initial condition, x0=y0=1
        Vector Y0 = new DenseVector(1., 1.);

        double x0 = 0, x1 = 1.; // solution domain
        double h = 0.1; // step size

        // the analytical solution
        RealVectorFunction F
                = new AbstractRealVectorFunction(1, 2) {
            @Override
            public Vector evaluate(Vector v) {
                double t = v.get(1);
                double x = 2. / 3 * exp(t) + 1. / 3 * exp(-5. * t);
                double y = 1. / 3 * exp(t) + 2. / 3 * exp(-5. * t);
                return new DenseVector(x, y);
            }
        };

        // define an IVP
        ODE1stOrder ivp = new ODE1stOrder(dY, Y0, x0, x1);
        // construt an ODE solver using Euler's method
        ODESolver solver = new EulerMethod(h);
        // solve the ODE
        ODESolution soln = solver.solve(ivp);
        // print out the solution function, y, at discrete points
        double[] t = soln.x();
        Vector[] v = soln.y();
        for (int i = 0; i < t.length; ++i) {
            System.out.println(String.format(
                    "y(%f) = %s vs %s",
                    t[i],
                    v[i],
                    F.evaluate(new DenseVector(t[i]))
            ));
        }
    }

    private void ABM() {
        System.out.println("solve an ODE using Adams-Bashforth methods");

        // define the ODE to solve
        DerivativeFunction dy = new DerivativeFunction() {

            @Override
            public Vector evaluate(double x, Vector v) {
                double y = v.get(1);
                double dy = y - x + 1;
                return new DenseVector(dy);
            }

            @Override
            public int dimension() {
                return 1;
            }
        };
        // initial condition, y0=1
        Vector y0 = new DenseVector(1.);

        double x0 = 0, x1 = 1.; // solution domain
        double h = 0.1; // step size

        // the analytical solution
        UnivariateRealFunction y = new AbstractUnivariateRealFunction() {
            @Override
            public double evaluate(double x) {
                double y = exp(x) + x;
                return y;
            }
        };

        // define an IVP
        ODE1stOrder ivp = new ODE1stOrder(dy, y0, x0, x1);

        // using first order Adams-Bashforth formula
        ODESolver solver1 = new AdamsBashforthMoulton(new ABMPredictorCorrector1(), h);
        ODESolution soln1 = solver1.solve(ivp);

        // using second order Adams-Bashforth formula
        ODESolver solver2 = new AdamsBashforthMoulton(new ABMPredictorCorrector2(), h);
        ODESolution soln2 = solver2.solve(ivp);

        // using third order Adams-Bashforth formula
        ODESolver solver3 = new AdamsBashforthMoulton(new ABMPredictorCorrector3(), h);
        ODESolution soln3 = solver3.solve(ivp);

        // using forth order Adams-Bashforth formula
        ODESolver solver4 = new AdamsBashforthMoulton(new ABMPredictorCorrector4(), h);
        ODESolution soln4 = solver4.solve(ivp);

        // using fifth order Adams-Bashforth formula
        ODESolver solver5 = new AdamsBashforthMoulton(new ABMPredictorCorrector5(), h);
        ODESolution soln5 = solver5.solve(ivp);

        double[] x = soln1.x();
        Vector[] y1 = soln1.y();
        Vector[] y2 = soln2.y();
        Vector[] y3 = soln3.y();
        Vector[] y4 = soln4.y();
        Vector[] y5 = soln5.y();
        for (int i = 0; i < x.length; ++i) {
            double yx = y.evaluate(x[i]); // the analytical solution
            double diff1 = yx - y1[i].get(1); // the first order error
            double diff2 = yx - y2[i].get(1); // the second order error
            double diff3 = yx - y3[i].get(1); // the third order error
            double diff4 = yx - y4[i].get(1); // the forth order error
            double diff5 = yx - y5[i].get(1); // the fifth order error
            System.out.println(
                    String.format("y(%f) = %s (%.16f); = %s (%.16f); = %s (%.16f); = %s (%.16f); = %s (%.16f)",
                            x[i], y1[i], diff1,
                            y2[i], diff2,
                            y3[i], diff3,
                            y4[i], diff4,
                            y5[i], diff5
                    ));
        }
    }

    private void RungeKutta() {
        System.out.println("solve an ODE using Runge-Kutta methods");

        // define the ODE to solve
        DerivativeFunction dy = new DerivativeFunction() {

            @Override
            public Vector evaluate(double x, Vector v) {
                double y = v.get(1);
                double dy = y - x + 1;
                return new DenseVector(dy);
            }

            @Override
            public int dimension() {
                return 1;
            }
        };
        // initial condition, y0=1
        Vector y0 = new DenseVector(1.);

        double x0 = 0, x1 = 1.; // solution domain
        double h = 0.1; // step size

        // the analytical solution
        UnivariateRealFunction y = new AbstractUnivariateRealFunction() {
            @Override
            public double evaluate(double x) {
                double y = exp(x) + x;
                return y;
            }
        };

        // define an IVP
        ODE1stOrder ivp = new ODE1stOrder(dy, y0, x0, x1);

        // using first order Runge-Kutta formula
        RungeKuttaStepper stepper1 = new RungeKutta1();
        ODESolver solver1 = new RungeKutta(stepper1, h);
        ODESolution soln1 = solver1.solve(ivp);

        // using second order Runge-Kutta formula
        RungeKuttaStepper stepper2 = new RungeKutta2();
        ODESolver solver2 = new RungeKutta(stepper2, h);
        ODESolution soln2 = solver2.solve(ivp);

        // using third order Runge-Kutta formula
        RungeKuttaStepper stepper3 = new RungeKutta3();
        ODESolver solver3 = new RungeKutta(stepper3, h);
        ODESolution soln3 = solver3.solve(ivp);

        double[] x = soln1.x();
        Vector[] y1 = soln1.y();
        Vector[] y2 = soln2.y();
        Vector[] y3 = soln3.y();
        for (int i = 0; i < x.length; ++i) {
            double yx = y.evaluate(x[i]); // the analytical solution
            double diff1 = yx - y1[i].get(1); // the first order error
            double diff2 = yx - y2[i].get(1); // the second order error
            double diff3 = yx - y3[i].get(1); // the third order error
            System.out.println(
                    String.format("y(%f) = %s (%.16f); = %s (%.16f); = %s (%.16f)",
                            x[i], y1[i], diff1,
                            y2[i], diff2,
                            y3[i], diff3
                    ));
        }
    }

    private void EulerMethod() {
        System.out.println("solve an ODE using Euler's method");

        // define the ODE to solve
        DerivativeFunction dy = new DerivativeFunction() {

            @Override
            public Vector evaluate(double x, Vector y) {

                Vector dy = y.scaled(-2. * x);
                return dy.add(1); // y' = 1 - 2xy
            }

            @Override
            public int dimension() {
                return 1;
            }
        };
        // initial condition, y0=0
        Vector y0 = new DenseVector(0.);

        double x0 = 0, x1 = 1.; // solution domain
        double h = 0.1; // step size

        // define an IVP
        ODE1stOrder ivp = new ODE1stOrder(dy, y0, x0, x1);
        // construt an ODE solver using Euler's method
        ODESolver solver = new EulerMethod(h);
        // solve the ODE
        ODESolution soln = solver.solve(ivp);
        // print out the solution function, y, at discrete points
        double[] x = soln.x();
        Vector[] y = soln.y();
        for (int i = 0; i < x.length; ++i) {
            System.out.println(String.format("y(%f) = %s", x[i], y[i]));
        }
    }

}
