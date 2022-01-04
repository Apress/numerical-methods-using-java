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
import dev.nm.algebra.linear.matrix.doubles.operation.MatrixMeasure;
import dev.nm.algebra.linear.vector.doubles.Vector;
import dev.nm.algebra.linear.vector.doubles.dense.DenseVector;
import dev.nm.analysis.differentiation.Ridders;
import dev.nm.analysis.differentiation.multivariate.Gradient;
import dev.nm.analysis.differentiation.multivariate.GradientFunction;
import dev.nm.analysis.differentiation.multivariate.Hessian;
import dev.nm.analysis.differentiation.multivariate.HessianFunction;
import dev.nm.analysis.differentiation.multivariate.Jacobian;
import dev.nm.analysis.differentiation.multivariate.JacobianFunction;
import dev.nm.analysis.differentiation.multivariate.MultivariateFiniteDifference;
import dev.nm.analysis.function.special.gaussian.Gaussian;
import dev.nm.analysis.differentiation.univariate.DBeta;
import dev.nm.analysis.differentiation.univariate.DBetaRegularized;
import dev.nm.analysis.differentiation.univariate.DErf;
import dev.nm.analysis.differentiation.univariate.DGamma;
import dev.nm.analysis.differentiation.univariate.DGaussian;
import dev.nm.analysis.differentiation.univariate.DPolynomial;
import dev.nm.analysis.differentiation.univariate.FiniteDifference;
import dev.nm.analysis.function.matrix.RntoMatrix;
import dev.nm.analysis.function.polynomial.Polynomial;
import dev.nm.analysis.function.rn2r1.AbstractBivariateRealFunction;
import dev.nm.analysis.function.rn2r1.RealScalarFunction;
import dev.nm.analysis.function.rn2r1.univariate.AbstractUnivariateRealFunction;
import dev.nm.analysis.function.rn2r1.univariate.UnivariateRealFunction;
import dev.nm.analysis.function.rn2rm.RealVectorFunction;
import dev.nm.analysis.function.special.beta.Beta;
import dev.nm.analysis.function.special.beta.BetaRegularized;
import dev.nm.analysis.function.special.gamma.Gamma;
import dev.nm.analysis.function.special.gamma.GammaLanczosQuick;
import dev.nm.analysis.function.special.gaussian.Erf;
import java.io.IOException;
import static java.lang.Math.*;

/**
 * Numerical Methods Using Java: For Data Science, Analysis, and Engineering
 *
 * @author haksunli
 * @see
 * https://www.amazon.com/Numerical-Methods-Using-Java-Engineering/dp/1484267966
 * https://nm.dev/
 */
public class Chapter6Differentiation {

    public static void main(String[] args) throws IOException {
        System.out.println("Chapter 6 demos on differentiation");

        Chapter6Differentiation chapter6 = new Chapter6Differentiation();
        chapter6.df1dx1();
        chapter6.df2dx2();
        chapter6.dGaussian();
        chapter6.dPolynomial();
        chapter6.dError();
        chapter6.dBeta();
        chapter6.dBetaRegularized();
        chapter6.dGamma();
        chapter6.partial_deriatvies_0010();
        chapter6.partial_deriatvies_0020();
        chapter6.gradient_0010();
        chapter6.gradient_0020();
        chapter6.jacobian_0010();
        chapter6.jacobian_0020();
        chapter6.hessian_0010();
        chapter6.ridder_0010();
        chapter6.ridder_0020();
    }

    public void ridder_0020() {
        System.out.println("comparing Ridder's method to finite difference for multivariate function");

        // f = xy + 2xyz
        RealScalarFunction f = new RealScalarFunction() {

            @Override
            public Double evaluate(Vector v) {
                return v.get(1) * v.get(2) + 2 * v.get(1) * v.get(2) * v.get(3);
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

        Ridders dxy_ridder = new Ridders(f, new int[]{1, 2});
        MultivariateFiniteDifference dxy // 1 + 2z
                // differentiate the first variable and then the second one
                = new MultivariateFiniteDifference(f, new int[]{1, 2});
        Vector x0 = new DenseVector(1., 1., 1.);
        System.out.println(String.format("Dxy(%s) by Ridder = %.16f", x0, dxy_ridder.evaluate(x0)));
        System.out.println(String.format("Dxy(%s) by FD =  %.16f", x0, dxy.evaluate(x0)));
        Vector x1 = new DenseVector(-100., 0., -1.5);
        System.out.println(String.format("Dxy(%s) by FD = %.16f", x1, dxy_ridder.evaluate(x1)));
        System.out.println(String.format("Dxy(%s) by FD = %.16f", x1, dxy.evaluate(x1)));

        //continuous function allows switching the order of differentiation by Clairaut's theorem
        Ridders dyx_ridder = new Ridders(f, new int[]{2, 1});
        MultivariateFiniteDifference dyx // 1 + 2z
                // differentiate the second variable and then the first one
                = new MultivariateFiniteDifference(f, new int[]{2, 1});
        System.out.println(String.format("Dyx(%s) by Ridder = %.16f", x0, dyx_ridder.evaluate(x0)));
        System.out.println(String.format("Dyx(%s) by FD = %.16f", x0, dyx.evaluate(x0)));
        System.out.println(String.format("Dyx(%s) by Ridder = %.16f", x1, dyx_ridder.evaluate(x1)));
        System.out.println(String.format("Dyx(%s) by FD = %.16f", x1, dyx.evaluate(x1)));
    }

    public void ridder_0010() {
        System.out.println("comparing Ridder's method to finite difference for univariate function");

        UnivariateRealFunction f = new AbstractUnivariateRealFunction() {
            @Override
            public double evaluate(double x) {
                return log(x);
            }
        };

        double x = 0.5;
        for (int order = 1; order < 10; ++order) {
            FiniteDifference fd = new FiniteDifference(f, order, FiniteDifference.Type.CENTRAL);
            Ridders ridder = new Ridders(f, order);
            System.out.println(String.format(
                    "%d-nd order derivative by Rideer @ %f = %.16f", order, x, ridder.evaluate(x)));
            System.out.println(String.format(
                    "%d-nd order derivative by FD @ %f = %.16f", order, x, fd.evaluate(x)));
        }
    }

    private void hessian_0010() {
        System.out.println("compute the Hessian for a multivariate real-valued function");

        RealScalarFunction f = new AbstractBivariateRealFunction() {
            @Override
            public double evaluate(double x, double y) {
                return x * y; // f = xy
            }
        };

        Vector x1 = new DenseVector(1., 1.);
        Hessian H1 = new Hessian(f, x1);
        System.out.println(String.format(
                "the Hessian at %s = %s, the det = %f",
                x1,
                H1,
                MatrixMeasure.det(H1)));

        Vector x2 = new DenseVector(0., 0.);
        Hessian H2 = new Hessian(f, x2);
        System.out.println(String.format(
                "the Hessian at %s = %s, the det = %f",
                x2,
                H2,
                MatrixMeasure.det(H2)));

        RntoMatrix H = new HessianFunction(f);
        Matrix Hx1 = H.evaluate(x1);
        System.out.println(String.format(
                "the Hessian at %s = %s, the det = %f",
                x1,
                Hx1,
                MatrixMeasure.det(Hx1)));
        Matrix Hx2 = H.evaluate(x2);
        System.out.println(String.format(
                "the Hessian at %s = %s, the det = %f",
                x2,
                Hx2,
                MatrixMeasure.det(Hx2)));
    }

    private void jacobian_0020() {
        System.out.println("compute the Jacobian for a multivariate vector-valued function");

        RealVectorFunction F = new RealVectorFunction() {
            @Override
            public Vector evaluate(Vector v) {
                double x1 = v.get(1);
                double x2 = v.get(2);
                double x3 = v.get(3);

                double f1 = 5. * x2;
                double f2 = 4. * x1 * x1 - 2. * sin(x2 * x3);
                double f3 = x2 * x3;

                return new DenseVector(f1, f2, f3);
            }

            @Override
            public int dimensionOfDomain() {
                return 3;
            }

            @Override
            public int dimensionOfRange() {
                return 3;
            }
        };

        Vector x0 = new DenseVector(0., 0., 1.);
        RntoMatrix J = new JacobianFunction(F);
        Matrix J0 = J.evaluate(x0);
        System.out.println(String.format(
                "the Jacobian at %s = %s, the det = %f",
                x0,
                J0,
                MatrixMeasure.det(J0)));

        Vector x1 = new DenseVector(1., 2., 3.);
        Matrix J1 = J.evaluate(x1);
        System.out.println(String.format(
                "the Jacobian at %s = %s, the det = %f",
                x1,
                J1,
                MatrixMeasure.det(J1)));
    }

    private void jacobian_0010() {
        System.out.println("compute the Jacobian for a multivariate vector-valued function");

        RealVectorFunction F = new RealVectorFunction() {
            @Override
            public Vector evaluate(Vector v) {
                double x = v.get(1);
                double y = v.get(2);

                double f1 = x * x * y;
                double f2 = 5. * x + sin(y);

                return new DenseVector(f1, f2);
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

        Vector x0 = new DenseVector(0., 0.);
        Matrix J00 = new Jacobian(F, x0);
        System.out.println(String.format(
                "the Jacobian at %s = %s, the det = %f",
                x0,
                J00,
                MatrixMeasure.det(J00)));

        RntoMatrix J = new JacobianFunction(F); // [2xy, x^2], [5, cosy]
        Matrix J01 = J.evaluate(x0);
        System.out.println(String.format(
                "the Jacobian at %s = %s, the det = %f",
                x0,
                J01,
                MatrixMeasure.det(J01)));

        Vector x1 = new DenseVector(1., PI);
        Matrix J1 = J.evaluate(x1);
        System.out.println(String.format(
                "the Jacobian at %s = %s, the det = %f",
                x1,
                J1,
                MatrixMeasure.det(J1)));
    }

    private void gradient_0020() {
        System.out.println("compute the gradient for a multivariate real-valued function");

        // f = -((cos(x))^2 + (cos(y))^2)^2
        RealScalarFunction f = new AbstractBivariateRealFunction() {

            @Override
            public double evaluate(double x, double y) {
                double z = cos(x) * cos(x);
                z += cos(y) * cos(y);
                z = -z * z;
                return z;
            }
        };

        Vector x1 = new DenseVector(0., 0.);
        Vector g1_0 = new Gradient(f, x1);
        System.out.println(String.format("gradient at %s = %s", x1, g1_0));

        GradientFunction df = new GradientFunction(f);
        Vector g1_1 = df.evaluate(x1);
        System.out.println(String.format("gradient at %s = %s", x1, g1_1));

        Vector x2 = new DenseVector(-1., 0.);
        Vector g2 = df.evaluate(x2);
        System.out.println(String.format("gradient at %s = %s", x2, g2));

        Vector x3 = new DenseVector(1., 0.);
        Vector g3 = df.evaluate(x3);
        System.out.println(String.format("gradient at %s = %s", x3, g3));
    }

    private void gradient_0010() {
        System.out.println("compute the gradient for a multivariate real-valued function");

        // f = x * exp(-(x^2 + y^2))
        RealScalarFunction f = new AbstractBivariateRealFunction() {

            @Override
            public double evaluate(double x, double y) {
                return x * exp(-(x * x + y * y));
            }
        };

        Vector x1 = new DenseVector(0., 0.);
        Vector g1_0 = new Gradient(f, x1);
        System.out.println(String.format("gradient at %s = %s", x1, g1_0));

        GradientFunction df = new GradientFunction(f);
        Vector g1_1 = df.evaluate(x1);
        System.out.println(String.format("gradient at %s = %s", x1, g1_1));

        Vector x2 = new DenseVector(-1., 0.);
        Vector g2 = df.evaluate(x2);
        System.out.println(String.format("gradient at %s = %s", x2, g2));

        Vector x3 = new DenseVector(1., 0.);
        Vector g3 = df.evaluate(x3);
        System.out.println(String.format("gradient at %s = %s", x3, g3));
    }

    private void partial_deriatvies_0010() {
        System.out.println("compute the partial derivatives for a multivariate real-valued function");

        // f = x^2 + xy + y^2
        RealScalarFunction f = new AbstractBivariateRealFunction() {

            @Override
            public double evaluate(double x, double y) {
                return x * x + x * y + y * y;
            }
        };

        // df/dx = 2x + y
        MultivariateFiniteDifference dx
                = new MultivariateFiniteDifference(f, new int[]{1});
        System.out.println(String.format("Dxy(1.,1.) %f", dx.evaluate(new DenseVector(1., 1.))));
    }

    private void partial_deriatvies_0020() {
        System.out.println("compute the partial derivatives for a multivariate real-valued function");

        // f = xy + 2xyz
        RealScalarFunction f = new RealScalarFunction() {

            @Override
            public Double evaluate(Vector v) {
                return v.get(1) * v.get(2) + 2 * v.get(1) * v.get(2) * v.get(3);
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

        MultivariateFiniteDifference dxy // 1 + 2z
                // differentiate the first variable and then the second one
                = new MultivariateFiniteDifference(f, new int[]{1, 2});
        System.out.println(String.format("Dxy(1.,1.,1.) %f", dxy.evaluate(new DenseVector(1., 1., 1.))));
        System.out.println(String.format("Dxy(-100.,0.,-1.5) %f", dxy.evaluate(new DenseVector(-100., 0., -1.5))));

        //continuous function allows switching the order of differentiation by Clairaut's theorem
        MultivariateFiniteDifference dyx // 1 + 2z
                // differentiate the second variable and then the first one
                = new MultivariateFiniteDifference(f, new int[]{2, 1});
        System.out.println(String.format("Dyx(1.,1.,1.) %f", dyx.evaluate(new DenseVector(1., 1., 1.))));
        System.out.println(String.format("Dyx(-100.,0.,-1.5) %f", dyx.evaluate(new DenseVector(-100., 0., -1.5))));
    }

    private void dGamma() {
        System.out.println("compute the first order derivative for the Gamma function");

        double z = 0.5;
        // <a href="http://en.wikipedia.org/wiki/Lanczos_approximation">Wikipedia: Lanczos approximation</a>
        Gamma G = new GammaLanczosQuick();
        DGamma dG = new DGamma();
        System.out.println(String.format("Gamma(%f) = %f", z, G.evaluate(z)));
        System.out.println(String.format("dGamma/dz(%f) = %f", z, dG.evaluate(z)));
    }

    private void dBetaRegularized() {
        System.out.println("compute the first order derivative for the regularized Beta function");

        double p = 0.5;
        double q = 2.5;
        BetaRegularized I = new BetaRegularized(p, q);
        DBetaRegularized dI = new DBetaRegularized(p, q);

        double x = 1.;
        System.out.println(String.format("BetaRegularized(%f) = %f", x, I.evaluate(x)));
        System.out.println(String.format("dBetaRegularized/dz(%f) = %f", x, dI.evaluate(x)));
    }

    private void dBeta() {
        System.out.println("compute the first order derivative for the Beta function");

        double x = 1.5;
        double y = 2.5;
        Beta B = new Beta();
        DBeta dB = new DBeta();
        System.out.println(String.format("Beta(%f) = %f", x, B.evaluate(x, y)));
        System.out.println(String.format("dBeta/dz(%f) = %f", x, dB.evaluate(x, y)));
    }

    private void dError() {
        System.out.println("compute the first order derivative for the Error function");

        double z = 0.5;
        Erf E = new Erf();
        DErf dE = new DErf();
        System.out.println(String.format("erf(%f) = %f", z, E.evaluate(z)));
        System.out.println(String.format("dErf/dz(%f) = %f", z, dE.evaluate(z)));
    }

    private void dPolynomial() {
        System.out.println("compute the first order derivative for a polynomial");

        Polynomial p = new Polynomial(1, 2, 1); // x^2 + 2x + 1
        Polynomial dp = new DPolynomial(p); // 2x + 2
        double x = 1.;
        System.out.println(String.format("dp/dx(%f) = %f", x, dp.evaluate(x)));
    }

    private void dGaussian() {
        System.out.println("compute the first order derivative for the Gaussian function");

        Gaussian G = new Gaussian(1., 0., 1.); // standard Gaussian
        DGaussian dG = new DGaussian(G);
        double x = -0.5;
        System.out.println(String.format("dG/dx(%f) = %f", x, dG.evaluate(x)));
        x = 0;
        System.out.println(String.format("dG/dx(%f) = %f", x, dG.evaluate(x)));
        x = 0.5;
        System.out.println(String.format("dG/dx(%f) = %f", x, dG.evaluate(x)));
    }

    private void df1dx1() {
        System.out.println("differentiate univariate functions");

        final UnivariateRealFunction f = new AbstractUnivariateRealFunction() {

            @Override
            public double evaluate(double x) {
                return -(x * x - 4 * x + 6); // -(x^2 - 4x + 6)
            }
        };
        double x = 2.;

        UnivariateRealFunction df1_forward
                = new FiniteDifference(f, 1, FiniteDifference.Type.FORWARD);
        double dfdx = df1_forward.evaluate(x); // evaluate at x
        System.out.println(String.format("df/dx(x=%f) = %.16f using forward difference", x, dfdx));

        UnivariateRealFunction df1_backward
                = new FiniteDifference(f, 1, FiniteDifference.Type.BACKWARD);
        dfdx = df1_backward.evaluate(x); // evaluate at x
        System.out.println(String.format("df/dx(x=%f) = %.16f using backward difference", x, dfdx));

        UnivariateRealFunction df1_central
                = new FiniteDifference(f, 1, FiniteDifference.Type.CENTRAL);
        dfdx = df1_central.evaluate(x); // evaluate at x
        System.out.println(String.format("df/dx(x=%f) = %.16f using central difference", x, dfdx));
    }

    private void df2dx2() {
        System.out.println("compute the second order derivative of univariate functions");

        final UnivariateRealFunction f = new AbstractUnivariateRealFunction() {

            @Override
            public double evaluate(double x) {
                return -(x * x - 4 * x + 6); // -(x^2 - 4x + 6)
            }
        };
        double x = 2.;

        System.out.println("differentiate univariate functions");

        UnivariateRealFunction df1_forward
                = new FiniteDifference(f, 2, FiniteDifference.Type.FORWARD);
        double dfdx = df1_forward.evaluate(x); // evaluate at x
        System.out.println(String.format("d2f/dx2(x=%f) = %.16f using forward difference", x, dfdx));

        UnivariateRealFunction df1_backward
                = new FiniteDifference(f, 2, FiniteDifference.Type.BACKWARD);
        dfdx = df1_backward.evaluate(x); // evaluate at x
        System.out.println(String.format("d2f/dx2(x=%f) = %.16f using backward difference", x, dfdx));

        UnivariateRealFunction df1_central
                = new FiniteDifference(f, 2, FiniteDifference.Type.CENTRAL);
        dfdx = df1_central.evaluate(x); // evaluate at x
        System.out.println(String.format("d2f/d2x(x=%f) = %.16f using central difference", x, dfdx));
    }

}
