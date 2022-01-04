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

import dev.nm.analysis.function.polynomial.Polynomial;
import dev.nm.analysis.function.polynomial.root.PolyRoot;
import dev.nm.analysis.function.rn2r1.univariate.AbstractUnivariateRealFunction;
import dev.nm.analysis.function.rn2r1.univariate.UnivariateRealFunction;
import dev.nm.analysis.root.univariate.BisectionRoot;
import dev.nm.analysis.root.univariate.BrentRoot;
import dev.nm.analysis.root.univariate.HalleyRoot;
import dev.nm.analysis.root.univariate.NewtonRoot;
import dev.nm.analysis.root.univariate.NoRootFoundException;
import dev.nm.number.complex.Complex;
import java.util.Arrays;
import java.util.List;

/**
 * Numerical Methods Using Java: For Data Science, Analysis, and Engineering
 *
 * @author haksunli
 * @see
 * https://www.amazon.com/Numerical-Methods-Using-Java-Engineering/dp/1484267966
 * https://nm.dev/
 */
public class Chapter3 {

    public static void main(String[] args) throws Exception {
        System.out.println("Chapter 3 demos");

        Chapter3 chapter3 = new Chapter3();
        chapter3.define_functions();
        chapter3.solve_root_for_polynomial_1();
        chapter3.solve_root_for_polynomial_2();
        chapter3.solve_root_using_bisection_method();
        chapter3.solve_root_using_Brent_method();
        chapter3.solve_root_using_Netwon_method();
        chapter3.solve_root_using_Hally_method();
    }

    public void define_functions() {
        System.out.println("defining functions");

        Polynomial p = new Polynomial(1, -10, 35, -50, 24);
        System.out.println("p(1) = " + p.evaluate(1.));

        UnivariateRealFunction f = new AbstractUnivariateRealFunction() {
            @Override
            public double evaluate(double x) {
                return Math.sin(x) * x - 3;
            }
        };
        System.out.println("f(1) = " + f.evaluate(1.));
    }

    public void solve_root_for_polynomial_1() {
        System.out.println("solve root for polynomial");

        Polynomial p = new Polynomial(1, -10, 35, -50, 24);
        PolyRoot solver = new PolyRoot();
        List<? extends Number> roots = solver.solve(p);
        System.out.println(Arrays.toString(roots.toArray()));
    }

    public void solve_root_for_polynomial_2() {
        System.out.println("solve root for polynomial with complex root");

        Polynomial p = new Polynomial(1, 0, 1); // x^2 + 1 = 0
        PolyRoot solver = new PolyRoot();
        List<? extends Number> roots0 = solver.solve(p);
        System.out.println(Arrays.toString(roots0.toArray()));
        List<Complex> roots1 = PolyRoot.getComplexRoots(roots0);
        System.out.println(Arrays.toString(roots1.toArray()));
    }

    public void solve_root_using_bisection_method() throws NoRootFoundException {
        System.out.println("solve root using bisection method");

        UnivariateRealFunction f = new AbstractUnivariateRealFunction() {
            @Override
            public double evaluate(double x) {
                return x * Math.sin(x) - 3; // x * six(x) - 3 = 0
            }
        };

        BisectionRoot solver = new BisectionRoot(1e-8, 30);
        double root = solver.solve(f, 12., 14.);
        double fx = f.evaluate(root);
        System.out.println(String.format("f(%f) = %f", root, fx));
    }

    public void solve_root_using_Brent_method() {
        System.out.println("solve root using Brent's method");

        UnivariateRealFunction f = new AbstractUnivariateRealFunction() {
            @Override
            public double evaluate(double x) {
                return x * x - 3; // x^2 - 3 = 0
            }
        };

        BrentRoot solver = new BrentRoot(1e-8, 10);
        double root = solver.solve(f, 0., 4.);
        double fx = f.evaluate(root);
        System.out.println(String.format("f(%f) = %f", root, fx));
    }

    public void solve_root_using_Netwon_method() throws NoRootFoundException {
        System.out.println("solve root using Newton's method using the first order derivatie");

        UnivariateRealFunction f = new AbstractUnivariateRealFunction() {
            @Override
            public double evaluate(double x) {
                return x * x + 4 * x - 5; // x^2 +4x - 5 = 0
            }
        };

        UnivariateRealFunction df = new AbstractUnivariateRealFunction() {
            @Override
            public double evaluate(double x) {
                return 2 * x + 4; // 2x + 4
            }
        };

        NewtonRoot solver = new NewtonRoot(1e-8, 5);
        double root = solver.solve(f, df, 5.);
        double fx = f.evaluate(root);
        System.out.println(String.format("f(%f) = %f", root, fx));
    }

    public void solve_root_using_Hally_method() throws NoRootFoundException {
        System.out.println("solve root using Hally's method using the first and second order derivaties");

        UnivariateRealFunction f = new AbstractUnivariateRealFunction() {
            @Override
            public double evaluate(double x) {
                return x * x + 4 * x - 5; // x^2 +4x - 5 = 0
            }
        };

        UnivariateRealFunction df = new AbstractUnivariateRealFunction() {
            @Override
            public double evaluate(double x) {
                return 2 * x + 4; // 2x + 4
            }
        };

        UnivariateRealFunction d2f = new AbstractUnivariateRealFunction() {
            @Override
            public double evaluate(double x) {
                return 2; // 2
            }
        };

        HalleyRoot solver = new HalleyRoot(1e-8, 3);
        double root = solver.solve(f, df, d2f, 5.);
        double fx = f.evaluate(root);

        System.out.println(String.format("f(%f) = %f", root, fx));
    }
}
