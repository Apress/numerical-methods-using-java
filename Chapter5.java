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

import java.io.BufferedWriter;
import java.io.FileWriter;
import java.io.IOException;

import dev.nm.algebra.linear.vector.doubles.dense.DenseVector;
import dev.nm.analysis.curvefit.LeastSquares;
import dev.nm.analysis.curvefit.interpolation.bivariate.BicubicInterpolation;
import dev.nm.analysis.curvefit.interpolation.bivariate.BicubicSpline;
import dev.nm.analysis.curvefit.interpolation.bivariate.BilinearInterpolation;
import dev.nm.analysis.curvefit.interpolation.bivariate.BivariateArrayGrid;
import dev.nm.analysis.curvefit.interpolation.bivariate.BivariateGrid;
import dev.nm.analysis.curvefit.interpolation.bivariate.BivariateGridInterpolation;
import dev.nm.analysis.curvefit.interpolation.bivariate.BivariateRegularGrid;
import dev.nm.analysis.curvefit.interpolation.multivariate.MultivariateArrayGrid;
import dev.nm.analysis.curvefit.interpolation.multivariate.RecursiveGridInterpolation;
import dev.nm.analysis.curvefit.interpolation.univariate.CubicHermite;
import dev.nm.analysis.curvefit.interpolation.univariate.CubicSpline;
import dev.nm.analysis.curvefit.interpolation.univariate.Interpolation;
import dev.nm.analysis.curvefit.interpolation.univariate.LinearInterpolation;
import dev.nm.analysis.curvefit.interpolation.univariate.NewtonPolynomial;
import dev.nm.analysis.function.rn2r1.RealScalarFunction;
import dev.nm.analysis.function.rn2r1.univariate.UnivariateRealFunction;
import dev.nm.analysis.function.tuple.OrderedPairs;
import dev.nm.analysis.function.tuple.SortedOrderedPairs;
import dev.nm.misc.datastructure.MultiDimensionalArray;
import java.io.File;
import static java.lang.Math.log;

/**
 * Numerical Methods Using Java: For Data Science, Analysis, and Engineering
 *
 * @author haksunli
 * @see
 * https://www.amazon.com/Numerical-Methods-Using-Java-Engineering/dp/1484267966
 * https://nm.dev/
 */
public class Chapter5 {

    public static void main(String[] args) throws IOException {
        System.out.println("Chapter 5 demos");

        Chapter5 chapter5 = new Chapter5();
        chapter5.least_square_curve_fitting();
        chapter5.linear_interpolation();
        chapter5.cubic_Hermite_interpolation();
        chapter5.cubic_spline_interpolation();
        chapter5.newton_polynomial_interpolation();
        chapter5.bivariate_interpolation();
        chapter5.bivariate_interpolation_using_derivatives();
        chapter5.multivariate_interpolation();
    }

    public void least_square_curve_fitting() {
        System.out.println("least square curve fitting");

        // the data set
        OrderedPairs data = new SortedOrderedPairs(
                new double[]{0., 1., 2., 3., 4., 5.},
                new double[]{0., 1., 1.414, 1.732, 2., 2.236}
        );

        LeastSquares ls = new LeastSquares(2);
        UnivariateRealFunction f = ls.fit(data);
        System.out.println(String.format("f(%.0f)=%f", 0., f.evaluate(0.))); // f(0) = 0.09
        System.out.println(String.format("f(%.0f)=%f", 1., f.evaluate(1.))); // f(1) = 0.82
        System.out.println(String.format("f(%.0f)=%f", 2., f.evaluate(2.))); // f(2) = 1.39
        System.out.println(String.format("f(%.0f)=%f", 3., f.evaluate(3.))); // f(3) = 1.81
        System.out.println(String.format("f(%.0f)=%f", 4., f.evaluate(4.))); // f(4) = 2.07
        System.out.println(String.format("f(%.0f)=%f", 5., f.evaluate(5.))); // f(5) = 2.17
    }

    public void linear_interpolation() throws IOException {
        System.out.println("linear interpolation");

        // the data set
        OrderedPairs data = new SortedOrderedPairs(
                new double[]{0., 0.7, 1.4, 2.1, 2.8, 3.5, 4.2, 4.9, 5.6, 6.3},
                new double[]{0., 0.644218, 0.98545, 0.863209, 0.334988, -0.350783, -0.871576, -0.982453, -0.631267, 0.0168139}
        );
        LinearInterpolation li = new LinearInterpolation();
        UnivariateRealFunction f = li.fit(data);
        System.out.println(f.evaluate(2)); // f(2) = 0.880672
        System.out.println(f.evaluate(3)); // f(3) = 0.139053

        plot(f, 100, 0, 6.5, "./plots/chapter5/figure_5_6/f_sample.txt");
    }

    public void cubic_Hermite_interpolation() throws IOException {
        System.out.println("cubic_Hermite interpolation");

        // the data set
        OrderedPairs data = new SortedOrderedPairs(
                new double[]{0., 0.7, 1.4, 2.1, 2.8, 3.5, 4.2, 4.9, 5.6, 6.3},
                new double[]{0., 0.644218, 0.98545, 0.863209, 0.334988, -0.350783, -0.871576, -0.982453, -0.631267, 0.0168139}
        );
        CubicHermite spline = new CubicHermite(CubicHermite.Tangents.CATMULL_ROM);
        // CubicHermite spline = new CubicHermite(CubicHermite.Tangents.FINITE_DIFFERENCE);
        UnivariateRealFunction f = spline.fit(data);
        System.out.println(f.evaluate(2)); // f(2) = 0.906030
        System.out.println(f.evaluate(3)); // f(3) = 0.145727

        plot(f, 100, 0, 6.3, "./plots/chapter5/figure_5_7/f_sample.txt");
        plot(f, 100, 0.7, 2.1, "./plots/chapter5/figure_5_8/f_sample.txt");
    }

    public void cubic_spline_interpolation() throws IOException {
        System.out.println("cubic spline interpolation");

        // the data set
        OrderedPairs data = new SortedOrderedPairs(
                new double[]{0., 1., 2., 3., 4., 5.},
                new double[]{0., 3.5, 5., 3., 1., 4.}
        );
        CubicSpline cs1 = CubicSpline.natural();
        UnivariateRealFunction f1 = cs1.fit(data);
        plot(f1, 100, 0, 5, "./plots/chapter5/figure_5_9/natural_cspline_sample.txt");

        CubicSpline cs2 = CubicSpline.clamped();
        UnivariateRealFunction f2 = cs2.fit(data);
        plot(f2, 100, 0, 5, "./plots/chapter5/figure_5_9/clamped_cspline_sample.txt");

        CubicSpline cs3 = CubicSpline.notAKnot();
        UnivariateRealFunction f3 = cs3.fit(data);
        plot(f3, 100, 0, 5, "./plots/chapter5/figure_5_9/notaknot_cspline_sample.txt");
    }

    public void newton_polynomial_interpolation() throws IOException {
        System.out.println("Newton polynomial interpolation");

        // 2 data points, linear form
        OrderedPairs data1 = new SortedOrderedPairs(
                new double[]{1., 3.},
                new double[]{log(1.), log(3.)}
        );
        Interpolation np1 = new NewtonPolynomial();
        UnivariateRealFunction f1 = np1.fit(data1);
        plot(f1, 100, 1, 3, "./plots/chapter5/figure_5_10/linear_newton_sample.txt");

        // 3 data points, quadratic form
        OrderedPairs data2 = new SortedOrderedPairs(
                new double[]{1., 2., 3.},
                new double[]{log(1.), log(2.), log(3.)}
        );
        Interpolation np2 = new NewtonPolynomial();
        UnivariateRealFunction f2 = np2.fit(data2);
        plot(f2, 100, 1, 3, "./plots/chapter5/figure_5_11/quadratic_newton_sample.txt");

        // comparison between Newton polynomial and cubic spline
        OrderedPairs data3 = new SortedOrderedPairs(
                new double[]{1., 2., 3., 4., 5., 6., 7.},
                new double[]{3., 4., 2., 5., 4., 3., 6.}
        );
        Interpolation np3 = new NewtonPolynomial();
        UnivariateRealFunction f3_1 = np3.fit(data3);
        plot(f3_1, 500, 1, 7, "./plots/chapter5/figure_5_12/newton_sample.txt");

        Interpolation cs = CubicSpline.natural();
        UnivariateRealFunction f3_2 = cs.fit(data3);
        plot(f3_2, 500, 1, 7, "./plots/chapter5/figure_5_12/cspline_sample.txt");
    }

    public void bivariate_interpolation() throws IOException {
        System.out.println("bivariate interpolation");

        BivariateGrid grids = new BivariateArrayGrid(
                new double[][]{
                    {1, 1, 1}, // z(1, 1) = 1, z(1, 2) = 1, z(1, 3) = 1
                    {2, 4, 8}, // z(2, 1) = 2, z(2, 2) = 4, z(2, 3) = 8
                    {3, 9, 27} // z(3, 1) = 3, z(3, 2) = 9, z(3, 3) = 27
                },
                new double[]{1, 2, 3}, // x
                new double[]{1, 2, 3} // y
        );

        BivariateGridInterpolation bl = new BilinearInterpolation();
        RealScalarFunction f1 = bl.interpolate(grids); // f3(1.5, 1.5) = 2.0
        System.out.println(f1.evaluate(new DenseVector(new double[]{1.5, 1.5})));

        BivariateGridInterpolation bs = new BicubicSpline();
        RealScalarFunction f2 = bs.interpolate(grids); // f2(1.5, 1.5) = 1.8828125
        System.out.println(f2.evaluate(new DenseVector(new double[]{1.5, 1.5})));

        BivariateGridInterpolation bi = new BicubicInterpolation();
        RealScalarFunction f3 = bi.interpolate(grids); // f1(1.5, 1.5) = 1.90625
        System.out.println(f3.evaluate(new DenseVector(new double[]{1.5, 1.5})));

        plot(f3, 30, new double[]{1, 3}, new double[]{1, 3}, "./plots/chapter5/figure_5_13/bicspline_sample.txt");
    }

    public void bivariate_interpolation_using_derivatives() throws IOException {
        System.out.println("bivariate interpolation using derivatives");

        // derivatives and answers from Michael Flanagan's library
        double[][] z = new double[][]{
            {1.0, 3.0, 5.0},
            {2.0, 4.0, 8.0},
            {9.0, 10.0, 11.0}
        };

        final double[][] dx = new double[][]{
            {6.0, 2.0, 2.0},
            {6.0, 7.0, 8.0},
            {6.0, 12.0, 14.0}
        };
        final double[][] dy = new double[][]{
            {8.0, 8.0, 8.0},
            {16.0, 12.0, 8.0},
            {4.0, 4.0, 4.0}
        };
        final double[][] dxdy = new double[][]{
            {16.0, 8.0, 0.0},
            {-4.0, -4.0, -4.0},
            {-24.0, -16.0, -8.0}
        };

        BicubicInterpolation.PartialDerivatives deriv
                = new BicubicInterpolation.PartialDerivatives() {
            @Override
            public double dx(BivariateGrid grid, int i, int j) {
                return getDeriv(dx, i, j); // for some reason the y-axis is written in reverse...
            }

            @Override
            public double dy(BivariateGrid grid, int i, int j) {
                return getDeriv(dy, i, j);
            }

            @Override
            public double dxdy(BivariateGrid grid, int i, int j) {
                return getDeriv(dxdy, i, j);
            }

            private double getDeriv(double[][] dx, int i, int j) {
                return dx[i][2 - j];
            }
        };

        BivariateGridInterpolation interpolation = new BicubicInterpolation(deriv);
        BivariateGrid grid = new BivariateRegularGrid(z, 0.0, 0.0, 0.5, 0.25);
        RealScalarFunction f = interpolation.interpolate(grid);

        System.out.println(f.evaluate(new DenseVector(0.0, 0.0))); // 1.0

        System.out.println(f.evaluate(new DenseVector(0.0, 0.125))); // 2.0
        System.out.println(f.evaluate(new DenseVector(0.0, 0.25))); // 3.0
        System.out.println(f.evaluate(new DenseVector(0.0, 0.375))); // 4.0
        System.out.println(f.evaluate(new DenseVector(0.0, 0.5))); // 5.0

        System.out.println(f.evaluate(new DenseVector(0.25, 0.0))); // 1.125
        System.out.println(f.evaluate(new DenseVector(0.25, 0.125))); // 2.078125
        System.out.println(f.evaluate(new DenseVector(0.25, 0.25))); // 3.1875
        System.out.println(f.evaluate(new DenseVector(0.25, 0.375))); // 4.765625
        System.out.println(f.evaluate(new DenseVector(0.25, 0.5))); // 6.5

        System.out.println(f.evaluate(new DenseVector(0.5, 0.0))); // 2.0
        System.out.println(f.evaluate(new DenseVector(0.5, 0.125))); // 2.875
        System.out.println(f.evaluate(new DenseVector(0.5, 0.25))); // 4.0
        System.out.println(f.evaluate(new DenseVector(0.5, 0.375))); // 5.875
        System.out.println(f.evaluate(new DenseVector(0.5, 0.5))); // 8.0

        System.out.println(f.evaluate(new DenseVector(0.75, 0.0))); // 5.125
        System.out.println(f.evaluate(new DenseVector(0.75, 0.125))); // 5.828125
        System.out.println(f.evaluate(new DenseVector(0.75, 0.25))); // 6.6875
        System.out.println(f.evaluate(new DenseVector(0.75, 0.375))); // 8.015625
        System.out.println(f.evaluate(new DenseVector(0.75, 0.5))); // 9.5

        System.out.println(f.evaluate(new DenseVector(1.0, 0.0))); // 9.0
        System.out.println(f.evaluate(new DenseVector(1.0, 0.125))); // 9.5
        System.out.println(f.evaluate(new DenseVector(1.0, 0.25))); // 10.0
        System.out.println(f.evaluate(new DenseVector(1.0, 0.375))); // 10.5
        System.out.println(f.evaluate(new DenseVector(1.0, 0.5))); // 11.0
    }

    public void multivariate_interpolation() {
        // the data set
        MultiDimensionalArray<Double> mda
                = new MultiDimensionalArray<>(2, 2, 2);
        mda.set(1., 0, 0, 0); // mda[0][0][0] = 1.
        mda.set(2., 1, 0, 0);
        mda.set(3., 0, 1, 0);
        mda.set(4., 0, 0, 1);
        mda.set(5., 1, 1, 0);
        mda.set(6., 1, 0, 1);
        mda.set(7., 0, 1, 1);
        mda.set(8., 1, 1, 1);

        MultivariateArrayGrid mvGrid = new MultivariateArrayGrid(
                mda,
                new double[]{1, 2},
                new double[]{1, 2},
                new double[]{1, 2}
        );
        RecursiveGridInterpolation rgi
                = new RecursiveGridInterpolation(new LinearInterpolation());
        RealScalarFunction f = rgi.interpolate(mvGrid);
        System.out.println(f.evaluate(new DenseVector(new double[]{1.5, 1.5, 1.5}))); // f(1.5, 1.5, 1.5) = 4.5
    }

    /**
     * Export sample points to plot a function curve.
     *
     * @param f the function to plot
     * @param nSamples number of sample points
     * @param rangeStart start of the plotting range
     * @param rangeEnd end of the plotting range
     * @param filename export file name
     * @throws IOException
     */
    private void plot(
            UnivariateRealFunction f,
            int nSamples,
            double rangeStart,
            double rangeEnd,
            String filename
    ) throws IOException {
        File file = new File(filename);
        if (!file.getParentFile().exists()) {
            file.getParentFile().mkdirs();
        }
        file.createNewFile();
        try (BufferedWriter writer = new BufferedWriter(new FileWriter(file))) {
            double gridSize = (rangeEnd - rangeStart) / (nSamples - 1);
            double x = rangeStart;
            for (int i = 0; i < nSamples; i++) {
                writer.write(String.format("%f %f\n", x, f.evaluate(x)));
//                writer.write(new StringBuilder().append(x).append(" ").append(f.evaluate(x)).toString());
//                writer.newLine();
                x += gridSize;
            }
        }
    }

    /**
     * Export sample points to plot a bivariate function surface.
     *
     * @param f the function to plot
     * @param nSamples number of sample points
     * @param rangeX plotting range of x
     * @param rangeY plotting range of y
     * @param filename export file name
     * @throws IOException
     */
    private void plot(
            RealScalarFunction f,
            int nSamples,
            double[] rangeX,
            double[] rangeY,
            String filename
    ) throws IOException {
        File file = new File(filename);
        if (!file.getParentFile().exists()) {
            file.getParentFile().mkdirs();
        }
        file.createNewFile();
        try (BufferedWriter writer = new BufferedWriter(new FileWriter(file))) {
            double gridSizeX = (rangeX[1] - rangeX[0]) / (nSamples - 1);
            double gridSizeY = (rangeY[1] - rangeY[0]) / (nSamples - 1);
            double x = rangeX[0];
            for (int i = 0; i < nSamples; i++) {
                double y = rangeY[0];
                for (int j = 0; j < nSamples; j++) {
                    writer.write(String.format("%f %f %s\n", x, y, f.evaluate(new DenseVector(x, y)).toString()));
//                    writer.write(new StringBuilder().append(x).append(" ").append(y).append(" ").append(f.evaluate(new DenseVector(x, y))).toString());
//                    writer.newLine();
                    y += gridSizeY;
                }
                x += gridSizeX;
            }
        }
    }

}
