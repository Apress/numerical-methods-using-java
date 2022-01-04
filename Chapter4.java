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
import dev.nm.analysis.function.rn2r1.AbstractTrivariateRealFunction;
import dev.nm.analysis.function.rn2r1.BivariateRealFunction;
import dev.nm.analysis.function.rn2r1.RealScalarFunction;
import dev.nm.analysis.function.rn2r1.TrivariateRealFunction;
import dev.nm.analysis.function.rn2rm.AbstractRealVectorFunction;
import dev.nm.analysis.function.rn2rm.RealVectorFunction;
import dev.nm.analysis.root.multivariate.NewtonSystemRoot;
import dev.nm.analysis.root.univariate.NoRootFoundException;
import java.util.Arrays;

/**
 * Numerical Methods Using Java: For Data Science, Analysis, and Engineering
 *
 * @author haksunli
 * @see
 * https://www.amazon.com/Numerical-Methods-Using-Java-Engineering/dp/1484267966
 * https://nm.dev/
 */
public class Chapter4 {

    public static void main(String[] args) throws Exception {
        System.out.println("Chapter 4 demos");

        Chapter4 chapter4 = new Chapter4();
        chapter4.define_multivariate_functions();
        chapter4.define_multivariate_vector_function();
        chapter4.solve_system_of_two_equations();
        chapter4.solve_system_of_equations();
    }

    public void define_multivariate_functions() {
        System.out.println("define multivariate functions");

        RealScalarFunction f1 = new AbstractRealScalarFunction(3) {
            @Override
            public Double evaluate(Vector x) {
                double x1 = x.get(1);
                double x2 = x.get(2);
                double x3 = x.get(3);
                return 2 * x1 * x1 + x2 * x2 - x3;
            }
        };
        System.out.println("f1(1,2,3) = " + f1.evaluate(new DenseVector(1, 2, 3)));

        TrivariateRealFunction f2 = new AbstractTrivariateRealFunction() {
            @Override
            public double evaluate(double x1, double x2, double x3) {
                return 2 * x1 * x1 + x2 * x2 - x3;
            }
        };
        System.out.println("f2(1,2,3) = " + f2.evaluate(new DenseVector(1, 2, 3)));
    }

    public void define_multivariate_vector_function() {
        System.out.println("define multivariate vector function");

        TrivariateRealFunction f1 = new AbstractTrivariateRealFunction() {
            @Override
            public double evaluate(double x, double y, double z) {
                return Math.pow(x, 2) + Math.pow(y, 3) - z - 6;
            }
        };
        TrivariateRealFunction f2 = new AbstractTrivariateRealFunction() {
            @Override
            public double evaluate(double x, double y, double z) {
                return 2 * x + 9 * y - z - 17;
            }
        };
        TrivariateRealFunction f3 = new AbstractTrivariateRealFunction() {
            @Override
            public double evaluate(double x, double y, double z) {
                return Math.pow(x, 4) + 5 * y + 6 * z - 29;
            }
        };
        RealScalarFunction[] F = new TrivariateRealFunction[]{f1, f2, f3};

        Vector x = new DenseVector(1.5, 2.5, 3.5);
        double f1_x = F[0].evaluate(x);
        double f2_x = F[1].evaluate(x);
        double f3_x = F[2].evaluate(x);
        double[] F_x = new double[]{f1_x, f2_x, f3_x};
        System.out.println("F(x) = " + Arrays.toString(F_x));

        RealVectorFunction G = new AbstractRealVectorFunction(3, 3) {
            @Override
            public Vector evaluate(Vector v) {
                double x = v.get(1);
                double y = v.get(2);
                double z = v.get(3);

                double g1 = Math.pow(x, 2) + Math.pow(y, 3) - z - 6;
                double g2 = 2 * x + 9 * y - z - 17;
                double g3 = Math.pow(x, 4) + 5 * y + 6 * z - 29;

                Vector g = new DenseVector(g1, g2, g3);
                return g;
            }
        };
        Vector Gx = G.evaluate(x);
        System.out.println("G(x) = " + Gx);
    }

    public void solve_system_of_two_equations() throws NoRootFoundException {
        System.out.println("solve a system of two equations");

        BivariateRealFunction f1 = new AbstractBivariateRealFunction() {
            @Override
            public double evaluate(double x, double y) {
                return 3 * x + y * y - 12;
            }
        };
        BivariateRealFunction f2 = new AbstractBivariateRealFunction() {
            @Override
            public double evaluate(double x, double y) {
                return x * x + y - 4;
            }
        };
        BivariateRealFunction[] F = new BivariateRealFunction[]{f1, f2};

        NewtonSystemRoot solver = new NewtonSystemRoot(1e-8, 10);
        Vector initial = new DenseVector(new double[]{0, 0}); // (0, 0)
        Vector root = solver.solve(F, initial);

        System.out.println(String.format("f(%s) = (%f, %f)", root.toString(), f1.evaluate(root), f2.evaluate(root)));
    }

    public void solve_system_of_equations() throws NoRootFoundException {
        System.out.println("solve a system of equations");

        RealVectorFunction G = new AbstractRealVectorFunction(3, 3) {
            @Override
            public Vector evaluate(Vector v) {
                double x = v.get(1);
                double y = v.get(2);
                double z = v.get(3);

                double g1 = Math.pow(x, 2) + Math.pow(y, 3) - z - 6;
                double g2 = 2 * x + 9 * y - z - 17;
                double g3 = Math.pow(x, 4) + 5 * y + 6 * z - 29;

                Vector g = new DenseVector(g1, g2, g3);
                return g;
            }
        };

        NewtonSystemRoot solver = new NewtonSystemRoot(1e-8, 15);
        Vector initial = new DenseVector(new double[]{0, 0, 0}); // (0, 0, 0)
        Vector root = solver.solve(G, initial);

        System.out.println(String.format("f(%s) = %s", root.toString(), G.evaluate(root).toString()));
    }
}
