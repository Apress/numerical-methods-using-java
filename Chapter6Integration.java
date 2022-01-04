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
import dev.nm.analysis.function.rn2r1.univariate.AbstractUnivariateRealFunction;
import dev.nm.analysis.function.rn2r1.univariate.UnivariateRealFunction;
import dev.nm.analysis.integration.univariate.riemann.ChangeOfVariable;
import dev.nm.analysis.integration.univariate.riemann.Integrator;
import dev.nm.analysis.integration.univariate.riemann.IterativeIntegrator;
import dev.nm.analysis.integration.univariate.riemann.gaussian.GaussChebyshevQuadrature;
import dev.nm.analysis.integration.univariate.riemann.gaussian.GaussHermiteQuadrature;
import dev.nm.analysis.integration.univariate.riemann.gaussian.GaussLaguerreQuadrature;
import dev.nm.analysis.integration.univariate.riemann.gaussian.GaussLegendreQuadrature;
import dev.nm.analysis.integration.univariate.riemann.newtoncotes.NewtonCotes;
import dev.nm.analysis.integration.univariate.riemann.newtoncotes.NewtonCotes.Type;
import dev.nm.analysis.integration.univariate.riemann.newtoncotes.Romberg;
import dev.nm.analysis.integration.univariate.riemann.newtoncotes.Simpson;
import dev.nm.analysis.integration.univariate.riemann.newtoncotes.Trapezoidal;
import dev.nm.analysis.integration.univariate.riemann.substitution.DoubleExponential;
import dev.nm.analysis.integration.univariate.riemann.substitution.DoubleExponential4HalfRealLine;
import dev.nm.analysis.integration.univariate.riemann.substitution.DoubleExponential4RealLine;
import dev.nm.analysis.integration.univariate.riemann.substitution.Exponential;
import dev.nm.analysis.integration.univariate.riemann.substitution.InvertingVariable;
import dev.nm.analysis.integration.univariate.riemann.substitution.MixedRule;
import dev.nm.analysis.integration.univariate.riemann.substitution.PowerLawSingularity;
import dev.nm.analysis.integration.univariate.riemann.substitution.StandardInterval;
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
public class Chapter6Integration {

    public static void main(String[] args) throws IOException {
        System.out.println("Chapter 6 demos on integration");

        Chapter6Integration chapter6 = new Chapter6Integration();
        chapter6.trapezoidalRule();
        chapter6.SimpsonRule();
        chapter6.NewtonCotes();
        chapter6.Romberg();
        chapter6.Gauss_Legendre_quadrature();
        chapter6.Gauss_Laguerre_quadrature();
        chapter6.Gauss_Hermite_quadrature();
        chapter6.Gauss_Chebyshev_quadrature();
        chapter6.standard_interval_substitution();
        chapter6.inverting_variable_substitution();
        chapter6.exponential_substitution();
        chapter6.mixed_rule_substitution();
        chapter6.double_exponential_substitution();
        chapter6.double_exponential_real_line_substitution();
        chapter6.power_law_singularity_substitution();
    }

    public void power_law_singularity_substitution() {
        System.out.println("integration by substitution using DoubleExponential4HalfRealLine");

        double a = 1, b = 2; // the limits
        NewtonCotes integrator1
                = new NewtonCotes(3, NewtonCotes.Type.OPEN, 1e-15, 15);
        ChangeOfVariable integrator2
                = new ChangeOfVariable(
                        new PowerLawSingularity(
                                PowerLawSingularity.PowerLawSingularityType.LOWER,
                                0.5, // gamma = 0.5
                                a, b),
                        integrator1);

        double I = integrator2.integrate( // I = 2
                new AbstractUnivariateRealFunction() {
            @Override
            public double evaluate(double x) {
                return 1 / sqrt(x - 1);
            }
        },
                a, b
        );

        System.out.println(String.format("S_[%.0f,%.0f] f(x) dx = %f", a, b, I));
    }

    public void double_exponential_half_real_line_substitution() {
        System.out.println("integration by substitution using DoubleExponential4HalfRealLine");

        UnivariateRealFunction f = new AbstractUnivariateRealFunction() {
            @Override
            public double evaluate(double x) {
                return x / (exp(x) - 1);
            }
        };

        double a = Double.NEGATIVE_INFINITY, b = Double.POSITIVE_INFINITY; // the limits
        NewtonCotes integrator
                = new NewtonCotes(3, NewtonCotes.Type.OPEN, 1e-15, 15);
        ChangeOfVariable instance
                = new ChangeOfVariable(new DoubleExponential4HalfRealLine(f, a, b, 1), integrator);
        double I = instance.integrate(f, a, b); // PI * PI / 6

        System.out.println(String.format("S_[%.0f,%.0f] f(x) dx = %f", a, b, I));
    }

    public void double_exponential_real_line_substitution() {
        System.out.println("integration by substitution using DoubleExponential4RealLine");

        UnivariateRealFunction f = new AbstractUnivariateRealFunction() {
            @Override
            public double evaluate(double x) {
                return exp(-x * x);
            }
        };

        double a = Double.NEGATIVE_INFINITY, b = Double.POSITIVE_INFINITY; // the limits
        NewtonCotes integrator1
                = new NewtonCotes(3, NewtonCotes.Type.CLOSED, 1e-15, 6);//only 6 iterations!
        ChangeOfVariable integrator2
                = new ChangeOfVariable(new DoubleExponential4RealLine(f, a, b, 1), integrator1);
        double I = integrator2.integrate(f, a, b); // sqrt(PI)

        System.out.println(String.format("S_[%.0f,%.0f] f(x) dx = %f", a, b, I));
    }

    public void double_exponential_substitution() {
        System.out.println("integration by substitution using DoubleExponential");

        UnivariateRealFunction f = new AbstractUnivariateRealFunction() {
            @Override
            public double evaluate(double x) {
                return log(x) * log(1 - x);
            }
        };

        double a = 0., b = 1.; // the limits
        NewtonCotes integrator1
                = new NewtonCotes(2, NewtonCotes.Type.CLOSED, 1e-15, 6); // only 6 iterations!
        ChangeOfVariable integrator2
                = new ChangeOfVariable(new DoubleExponential(f, a, b, 1), integrator1);
        double I = integrator2.integrate(f, a, b); // I = 2 - PI * PI / 6

        System.out.println(String.format("S_[%.0f,%.0f] f(x) dx = %f", a, b, I));
    }

    private void mixed_rule_substitution() {
        System.out.println("integration by substitution using MixedRule");

        UnivariateRealFunction f = new AbstractUnivariateRealFunction() {
            @Override
            public double evaluate(double x) {
                return exp(-x) * pow(x, -1.5) * sin(x / 2);
            }
        };

        double a = 0., b = Double.POSITIVE_INFINITY; // the limits
        NewtonCotes integrator1
                = new NewtonCotes(2, NewtonCotes.Type.CLOSED, 1e-15, 7); // only 7 iteration!
        ChangeOfVariable integrator2
                = new ChangeOfVariable(new MixedRule(f, a, b, 1), integrator1);
        double I = integrator2.integrate(f, a, b); // I = sqrt(PI * (sqrt(5) - 2))

        System.out.println(String.format("S_[%.0f,%.0f] f(x) dx = %f", a, b, I));
    }

    private void exponential_substitution() {
        System.out.println("integration by substitution using Exponential");

        double a = 0., b = Double.POSITIVE_INFINITY; // the limits
        NewtonCotes integrator1
                = new NewtonCotes(3, NewtonCotes.Type.OPEN, 1e-15, 15);
        ChangeOfVariable integrator2
                = new ChangeOfVariable(new Exponential(a), integrator1);

        double I = integrator2.integrate( // I = sqrt(PI)/2
                new AbstractUnivariateRealFunction() {
            @Override
            public double evaluate(double x) {
                return sqrt(x) * exp(-x); // the original integrand
            }
        },
                a, b // the original limits
        );

        System.out.println(String.format("S_[%.0f,%.0f] f(x) dx = %f", a, b, I));
    }

    private void inverting_variable_substitution() {
        System.out.println("integration by substitution using InvertingVariable");

        double a = 1., b = Double.POSITIVE_INFINITY; // the limits
        NewtonCotes integrator1
                = new NewtonCotes(3, NewtonCotes.Type.OPEN, 1e-15, 10);
        ChangeOfVariable integrator2
                = new ChangeOfVariable(new InvertingVariable(a, b), integrator1);

        double I = integrator2.integrate( // I = 1
                new AbstractUnivariateRealFunction() {
            @Override
            public double evaluate(double x) {
                return 1 / x / x; // the original integrand
            }
        },
                a, b // the original limits
        );

        System.out.println(String.format("S_[%.0f,%.0f] f(x) dx = %f", a, b, I));
    }

    private void standard_interval_substitution() {
        System.out.println("integration by substitution using StandardInterval");

        double a = 0., b = 10.; // the limits
        Integrator integrator1
                = new NewtonCotes(3, NewtonCotes.Type.OPEN, 1e-8, 10);
        Integrator integrator2
                = new ChangeOfVariable(new StandardInterval(a, b), integrator1);

        double I = integrator2.integrate(
                new AbstractUnivariateRealFunction() {
            @Override
            public double evaluate(double t) {
                return t; // the original integrand
            }
        },
                a, b // the original limits
        );

        System.out.println(String.format("S_[%.0f,%.0f] f(x) dx = %f", a, b, I));
    }

    private void Gauss_Chebyshev_quadrature() {
        System.out.println("integrate using the Gauss Chebyshev quadrature");

        final Polynomial poly = new Polynomial(1, 2, 1); // x^2 + 2x + 1
        UnivariateRealFunction f = new AbstractUnivariateRealFunction() {
            @Override
            public double evaluate(double x) {
                // second order polynomial divided by weighting can be reproduced exactly
                return poly.evaluate(x) / sqrt(1 - x * x);
            }
        };

        // the integrators
        Integrator integrator1 = new Trapezoidal(1e-8, 20); // using the trapezoidal rule
        Integrator integrator2 = new Simpson(1e-8, 20); // using the Simpson rule        
        Integrator integrator3 = new GaussChebyshevQuadrature(2);

        // the limits
        double a = -1., b = 1.;

        // the integrations
        double I1 = integrator1.integrate(f, a, b);
        System.out.println(String.format("S_[%.0f,%.0f] f(x) dx = %.16f, using the trapezoidal rule", a, b, I1));
        double I2 = integrator2.integrate(f, a, b);
        System.out.println(String.format("S_[%.0f,%.0f] f(x) dx = %.16f, using the Sampson rule", a, b, I2));
        double I3 = integrator3.integrate(f, a, b);
        System.out.println(String.format("S_[%.0f,%.0f] f(x) dx = %.16f, using the Gauss Hermite quadrature", a, b, I3));
    }

    public void Gauss_Hermite_quadrature() {
        System.out.println("integrate using the Gauss Hermite quadrature");

        final Polynomial poly = new Polynomial(1, 2, 1); // x^2 + 2x + 1
        UnivariateRealFunction f = new AbstractUnivariateRealFunction() {
            @Override
            public double evaluate(double x) {
                return exp(-(x * x)) * poly.evaluate(x); // e^(-x^2) * (x^2 + 2x + 1)
            }
        };

        // the integrators
        Integrator integrator1 = new Trapezoidal(1e-8, 20); // using the trapezoidal rule
        Integrator integrator2 = new Simpson(1e-8, 20); // using the Simpson rule        
        Integrator integrator3 = new GaussHermiteQuadrature(2);

        // the limits
        double a = Double.NEGATIVE_INFINITY, b = Double.POSITIVE_INFINITY;

        // the integrations
        double I1 = integrator1.integrate(f, a, b);
        System.out.println(String.format("S_[%.0f,%.0f] f(x) dx = %.16f, using the trapezoidal rule", a, b, I1));
        double I2 = integrator2.integrate(f, a, b);
        System.out.println(String.format("S_[%.0f,%.0f] f(x) dx = %.16f, using the Sampson rule", a, b, I2));
        double I3 = integrator3.integrate(f, a, b);
        System.out.println(String.format("S_[%.0f,%.0f] f(x) dx = %.16f, using the Gauss Hermite quadrature", a, b, I3));
    }

    private void Gauss_Laguerre_quadrature() {
        System.out.println("integrate using the Gauss Laguerre quadrature");

        final Polynomial poly = new Polynomial(1, 2, 1); // x^2 + 2x + 1
        UnivariateRealFunction f = new AbstractUnivariateRealFunction() {
            @Override
            public double evaluate(double x) {
                return exp(-x) * poly.evaluate(x); // e^-x * (x^2 + 2x + 1)
            }
        };

        // the integrators
        Integrator integrator1 = new Trapezoidal(1e-8, 20); // using the trapezoidal rule
        Integrator integrator2 = new Simpson(1e-8, 20); // using the Simpson rule        
        Integrator integrator3 = new GaussLaguerreQuadrature(2, 1e-8);

        // the limits
        double a = 0., b = Double.POSITIVE_INFINITY;

        // the integrations
        double I1 = integrator1.integrate(f, a, b);
        System.out.println(String.format("S_[%.0f,%.0f] f(x) dx = %.16f, using the trapezoidal rule", a, b, I1));
        double I2 = integrator2.integrate(f, a, b);
        System.out.println(String.format("S_[%.0f,%.0f] f(x) dx = %.16f, using the Sampson rule", a, b, I2));
        double I3 = integrator3.integrate(f, a, b);
        System.out.println(String.format("S_[%.0f,%.0f] f(x) dx = %.16f, using the Gauss Laguerre quadrature", a, b, I3));
    }

    private void Gauss_Legendre_quadrature() {
        System.out.println("integrate using the Gauss Legendre quadrature");

        UnivariateRealFunction f = new AbstractUnivariateRealFunction() {
            @Override
            public double evaluate(double x) {
                return 4 * x * x * x + 2 * x + 1; // x^4 + x^2 + x
            }
        };

        // the integrators
        Integrator integrator1 = new Trapezoidal(1e-8, 20); // using the trapezoidal rule
        Integrator integrator2 = new Simpson(1e-8, 20); // using the Simpson rule        
        Integrator integrator3 = new GaussLegendreQuadrature(2);
        // the limits
        double a = -1., b = 1.;

        // the integrations
        double I1 = integrator1.integrate(f, a, b);
        System.out.println(String.format("S_[%.0f,%.0f] f(x) dx = %.16f, using the trapezoidal rule", a, b, I1));
        double I2 = integrator2.integrate(f, a, b);
        System.out.println(String.format("S_[%.0f,%.0f] f(x) dx = %.16f, using the Sampson rule", a, b, I2));
        double I3 = integrator3.integrate(f, a, b);
        System.out.println(String.format("S_[%.0f,%.0f] f(x) dx = %.16f, using the Gauss Legendre quadrature", a, b, I3));
    }

    private void Romberg() {
        System.out.println("integrate using the Romberg formulas");

        final UnivariateRealFunction f = new AbstractUnivariateRealFunction() {

            @Override
            public double evaluate(double x) {
                return exp(2. * x) - 4. * x - 7.;
            }
        };
        IterativeIntegrator integrator1 = new Trapezoidal(1e-8, 20); // using the trapezoidal rule
        Integrator integrator2 = new Simpson(1e-8, 20); // using the Simpson rule
        Integrator integrator3 = new Romberg(integrator1);
        // the limit
        double a = 0., b = 1.;
        // the integrations
        double I1 = integrator1.integrate(f, a, b);
        System.out.println(String.format("S_[%.0f,%.0f] f(x) dx = %.16f, using the trapezoidal rule", a, b, I1));
        double I2 = integrator2.integrate(f, a, b);
        System.out.println(String.format("S_[%.0f,%.0f] f(x) dx = %.16f, using the Simpson rule", a, b, I2));
        double I3 = integrator3.integrate(f, a, b);
        System.out.println(String.format("S_[%.0f,%.0f] f(x) dx = %.16f, using the Romberg formula", a, b, I3));
    }

    private void NewtonCotes() {
        System.out.println("integrate using the Newton-Cotes formulas");

        final UnivariateRealFunction f = new AbstractUnivariateRealFunction() {

            @Override
            public double evaluate(double x) {
                return 4. / (1. + x * x); // 4/(1+x^2)
            }
        };

        // the limit
        double a = 0., b = 1.;
        Integrator integrator1 = new Trapezoidal(1e-8, 20); // using the trapezoidal rule
        Integrator integrator2 = new Simpson(1e-8, 20); // using the Simpson rule
        Integrator integrator3 = new NewtonCotes(3, Type.CLOSED, 1e-8, 20); // using the Newton-Cotes rule
        Integrator integrator4 = new NewtonCotes(3, Type.OPEN, 1e-8, 20); // using the Newton-Cotes rule

        // the integrations
        double I1 = integrator1.integrate(f, a, b);
        System.out.println(String.format("S_[%.0f,%.0f] f(x) dx = %.16f, using the trapezoidal rule", a, b, I1));
        double I2 = integrator2.integrate(f, a, b);
        System.out.println(String.format("S_[%.0f,%.0f] f(x) dx = %.16f, using the Simpson rule", a, b, I2));
        double I3 = integrator3.integrate(f, a, b);
        System.out.println(String.format("S_[%.0f,%.0f] f(x) dx = %.16f, using the Newton-Cotes closed rule", a, b, I3));
        double I4 = integrator4.integrate(f, a, b);
        System.out.println(String.format("S_[%.0f,%.0f] f(x) dx = %.16f, using using the Newton-Cotes open rule", a, b, I4));
    }

    private void SimpsonRule() {
        System.out.println("integrate using the Simpson rule");

        final UnivariateRealFunction f = new AbstractUnivariateRealFunction() {

            @Override
            public double evaluate(double x) {
                return -(x * x - 4 * x + 6); // -(x^2 - 4x + 6)
            }
        };

        // the limit
        double a = 0., b = 4.;
        // an integrator using the Simpson rule
        Integrator integrator = new Simpson(1e-8, 20); // precision, max number of iterations
        // the integration
        double I = integrator.integrate(f, a, b);
        System.out.println(String.format("S_[%.0f,%.0f] f(x) dx = %f", a, b, I));
    }

    private void trapezoidalRule() {
        System.out.println("integrate using the trapezoidal rule");

        final UnivariateRealFunction f = new AbstractUnivariateRealFunction() {

            @Override
            public double evaluate(double x) {
                return -(x * x - 4 * x + 6); // -(x^2 - 4x + 6)
            }
        };

        // the limit
        double a = 0., b = 4.;
        // an integrator using the trapezoidal rule
        Integrator integrator = new Trapezoidal(1e-8, 20); // precision, max number of iterations
        // the integration
        double I = integrator.integrate(f, a, b);
        System.out.println(String.format("S_[%.0f,%.0f] f(x) dx = %f", a, b, I));
    }

}
