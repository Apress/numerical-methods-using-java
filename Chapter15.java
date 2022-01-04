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
import dev.nm.algebra.linear.matrix.doubles.operation.Inverse;
import dev.nm.algebra.linear.matrix.doubles.operation.MatrixFactory;
import static dev.nm.algebra.linear.matrix.doubles.operation.MatrixFactory.cbind;
import dev.nm.algebra.linear.vector.doubles.Vector;
import dev.nm.algebra.linear.vector.doubles.dense.DenseVector;
import dev.nm.number.DoubleUtils;
import dev.nm.stat.cointegration.CointegrationMLE;
import dev.nm.stat.cointegration.JohansenAsymptoticDistribution.Test;
import dev.nm.stat.cointegration.JohansenTest;
import dev.nm.stat.descriptive.covariance.SampleCovariance;
import dev.nm.stat.descriptive.moment.Mean;
import dev.nm.stat.random.rng.RNGUtils;
import dev.nm.stat.random.rng.univariate.RandomNumberGenerator;
import dev.nm.stat.random.rng.univariate.normal.StandardNormalRNG;
import dev.nm.stat.test.timeseries.adf.AugmentedDickeyFuller;
import dev.nm.stat.test.timeseries.adf.TrendType;
import dev.nm.stat.test.timeseries.portmanteau.LjungBox;
import dev.nm.stat.timeseries.datastructure.multivariate.realtime.inttime.MultivariateIntTimeTimeSeries;
import dev.nm.stat.timeseries.datastructure.multivariate.realtime.inttime.MultivariateSimpleTimeSeries;
import dev.nm.stat.timeseries.datastructure.univariate.realtime.inttime.SimpleTimeSeries;
import dev.nm.stat.timeseries.linear.multivariate.MultivariateAutoCovarianceFunction;
import dev.nm.stat.timeseries.linear.multivariate.arima.VARIMAModel;
import dev.nm.stat.timeseries.linear.multivariate.arima.VARIMASim;
import dev.nm.stat.timeseries.linear.multivariate.stationaryprocess.MultivariateForecastOneStep;
import dev.nm.stat.timeseries.linear.multivariate.stationaryprocess.arma.VARFit;
import dev.nm.stat.timeseries.linear.multivariate.stationaryprocess.arma.VARMAAutoCovariance;
import dev.nm.stat.timeseries.linear.multivariate.stationaryprocess.arma.VARMAModel;
import dev.nm.stat.timeseries.linear.multivariate.stationaryprocess.arma.VARModel;
import dev.nm.stat.timeseries.linear.multivariate.stationaryprocess.arma.VARXModel;
import dev.nm.stat.timeseries.linear.multivariate.stationaryprocess.arma.VECMTransitory;
import dev.nm.stat.timeseries.linear.multivariate.stationaryprocess.arma.VMAModel;
import dev.nm.stat.timeseries.linear.univariate.arima.ARIMAForecast;
import dev.nm.stat.timeseries.linear.univariate.arima.ARIMAModel;
import dev.nm.stat.timeseries.linear.univariate.arima.ARIMASim;
import dev.nm.stat.timeseries.linear.univariate.arima.AutoARIMAFit;
import dev.nm.stat.timeseries.linear.univariate.sample.SampleAutoCorrelation;
import dev.nm.stat.timeseries.linear.univariate.sample.SampleAutoCovariance;
import dev.nm.stat.timeseries.linear.univariate.sample.SamplePartialAutoCorrelation;
import dev.nm.stat.timeseries.linear.univariate.stationaryprocess.AdditiveModel;
import dev.nm.stat.timeseries.linear.univariate.stationaryprocess.MADecomposition;
import dev.nm.stat.timeseries.linear.univariate.stationaryprocess.MultiplicativeModel;
import dev.nm.stat.timeseries.linear.univariate.stationaryprocess.arma.ARMAForecast;
import dev.nm.stat.timeseries.linear.univariate.stationaryprocess.arma.ARMAForecastOneStep;
import dev.nm.stat.timeseries.linear.univariate.stationaryprocess.arma.ARMAModel;
import dev.nm.stat.timeseries.linear.univariate.stationaryprocess.arma.ARModel;
import dev.nm.stat.timeseries.linear.univariate.stationaryprocess.arma.AutoCorrelation;
import dev.nm.stat.timeseries.linear.univariate.stationaryprocess.arma.AutoCovariance;
import dev.nm.stat.timeseries.linear.univariate.stationaryprocess.arma.ConditionalSumOfSquares;
import dev.nm.stat.timeseries.linear.univariate.stationaryprocess.arma.LinearRepresentation;
import dev.nm.stat.timeseries.linear.univariate.stationaryprocess.arma.MAModel;
import dev.nm.stat.timeseries.linear.univariate.stationaryprocess.armagarch.ARMAGARCHFit;
import dev.nm.stat.timeseries.linear.univariate.stationaryprocess.garch.GARCHFit;
import dev.nm.stat.timeseries.linear.univariate.stationaryprocess.garch.GARCHModel;
import dev.nm.stat.timeseries.linear.univariate.stationaryprocess.garch.GARCHSim;
import java.io.IOException;
import static java.lang.Math.log;
import java.util.Arrays;

/**
 * Numerical Methods Using Java: For Data Science, Analysis, and Engineering
 *
 * @author haksunli
 * @see
 * https://www.amazon.com/Numerical-Methods-Using-Java-Engineering/dp/1484267966
 * https://nm.dev/
 */
public class Chapter15 {

    public static void main(String[] args) throws Exception {
        System.out.println("Chapter 15 demos");

        Chapter15 chapter15 = new Chapter15();
        chapter15.univariate_ts();
        chapter15.sp500_daily();
        chapter15.sp500_monthly();
        chapter15.decomposition();
        chapter15.ar();
        chapter15.ma();
        chapter15.arma11();
        chapter15.arma23();
        chapter15.arma_forecast();
        chapter15.goodness_of_fit();
        chapter15.adf();
        chapter15.garch_models();
        chapter15.mts();
        chapter15.spx_aapl();
        chapter15.VARIMA();
        chapter15.VAR();
        chapter15.VARMA_forecast();
        chapter15.VARMA_autocovariance();
        chapter15.varim_simulation();
        chapter15.cointegration();
        chapter15.vecm();
    }

    public void vecm() {
        System.out.println("convert VAR to VECM");

        // define a VAR(2) model
        VARXModel var = new VARXModel(
                new Matrix[]{
                    new DenseMatrix(
                            new double[][]{
                                {-0.210, 0.167},
                                {0.512, 0.220}
                            }),
                    new DenseMatrix(
                            new double[][]{
                                {0.743, 0.746},
                                {-0.405, 0.572}
                            })
                },
                null);

        // construct a VECM fron a VAR
        VECMTransitory vecm = new VECMTransitory(var);

        System.out.println("dimension = " + vecm.dimension());
        System.out.println("PI, the impact matrix = ");
        System.out.println(vecm.pi());
        System.out.println("GAMMA = ");
        System.out.println(vecm.gamma(1));
    }

    public void cointegration() throws Exception {
        System.out.println("cointegration between SPX and AAPL");

        // read the monthly S&P 500 data from a csv file
        double[][] spx_arr
                = DoubleUtils.readCSV2d(
                        this.getClass().getClassLoader().getResourceAsStream("sp500_monthly.csv"),
                        true,
                        true
                );
        // convert the csv file into a matrix for manipulation
        Matrix spx_M1 = new DenseMatrix(spx_arr);
        // remove the data before 2008
        Matrix spx_M2 = MatrixFactory.subMatrix(spx_M1, 325, spx_M1.nRows(), 1, spx_M1.nCols());
        // extract the column of adjusted close prices
        Vector spx_v = spx_M2.getColumn(5); // adjusted closes

        // read the monthly AAPL data from a csv file
        double[][] appl_arr
                = DoubleUtils.readCSV2d(
                        this.getClass().getClassLoader().getResourceAsStream("AAPL_monthly.csv"),
                        true,
                        true
                );
        // convert the csv file into a matrix for manipulation
        Matrix aapl_M1 = new DenseMatrix(appl_arr);
        // remove the data before 2008
        Matrix aapl_M2 = MatrixFactory.subMatrix(aapl_M1, 325, aapl_M1.nRows(), 1, aapl_M1.nCols());
        // extract the column of adjusted close prices
        Vector aapl_v = aapl_M2.getColumn(5); // adjusted closes

        // combine SPX and AAPL to form a bivariate time series
        MultivariateSimpleTimeSeries mts
                = new MultivariateSimpleTimeSeries(cbind(spx_v, aapl_v));
//        System.out.println("(spx, aapl) prices: \n" + mts);

        // run cointegration on all combinations of available test and trend types
        for (Test test : Test.values()) {
            for (dev.nm.stat.cointegration.JohansenAsymptoticDistribution.TrendType trend
                    : dev.nm.stat.cointegration.JohansenAsymptoticDistribution.TrendType.values()) {
                CointegrationMLE coint = new CointegrationMLE(mts, true);
                JohansenTest johansen
                        = new JohansenTest(test, trend, coint.getEigenvalues().size());
                System.out.println("alpha:");
                System.out.println(coint.alpha());
                System.out.println("beta:");
                System.out.println(coint.beta());
                System.out.println("Johansen test: "
                        + test.toString()
                        + "\t" + trend.toString()
                        + "\t eigenvalues: " + coint.getEigenvalues()
                        + "\t statistics: " + johansen.getStats(coint)
                );
            }
        }

        // run ADF test to check if the cointegrated series is indeed stationary
        double[] betas = new double[]{-12.673296, -6.995392};
        for (double beta : betas) {
            System.out.printf("testing for beta = %f%n", beta);
            Vector ci = spx_v.add(aapl_v.scaled(beta));
            AugmentedDickeyFuller adf
                    = new AugmentedDickeyFuller(
                            ci.toArray(),
                            TrendType.CONSTANT, // constant drift term
                            4, // the lag order
                            null
                    );

            System.out.println("H0: " + adf.getNullHypothesis());
            System.out.println("H1: " + adf.getAlternativeHypothesis());
            System.out.printf("the p-value for the test = %f%n", adf.pValue());
            System.out.printf("the statistics for the test = %f%n", adf.statistics());
        }
    }

    public void varim_simulation() {
        // number of random numbers to generate
        int T = 100000;

        // the mean
        Vector MU = new DenseVector(new double[]{1., 1.});
        // the AR coefficients
        Matrix[] PHI = new Matrix[]{
            new DenseMatrix(
            new double[][]{
                {0.5, 0.5},
                {0., 0.5}
            })};
        // the MA coefficients
        Matrix[] THETA = new Matrix[]{PHI[0]};
        // the white noise covariance structure
        Matrix SIGMA
                = new DenseMatrix(
                        new double[][]{
                            {1, 0.2},
                            {0.2, 1}
                        });

        // construct a VARMA model
        VARMAModel VARMA = new VARMAModel(MU, PHI, THETA, SIGMA);

        // construct a random number generator from a VARMA model
        VARIMASim SIM = new VARIMASim(VARMA);
        SIM.seed(1234567890L);

        // produce the random vectors
        Vector[] data = new Vector[T];
        for (int i = 0; i < T; ++i) {
            data[i] = new DenseVector(SIM.nextVector());
        }

        // statistics about the simulation
        System.out.printf("each vector size = %d%n", data[0].size());
        System.out.printf("sample size = %d%n", data.length);

        // compute the theoretical mean of the data
        double[] theoretical_mean = new Inverse(new DenseMatrix(2, 2).ONE().minus(VARMA.AR(1))).multiply(new DenseVector(MU)).toArray();
        System.out.println("theoretical mean =");
        System.out.println(Arrays.toString(theoretical_mean));

        // cast the random data in matrix form for easy manipulation
        Matrix dataM = MatrixFactory.rbind(data);
        // compute the sample mean of the data
        double sample_mean1 = new Mean(dataM.getColumn(1).toArray()).value();
        double sample_mean2 = new Mean(dataM.getColumn(2).toArray()).value();
        System.out.printf("sample mean of the first variable = %f%n", sample_mean1);
        System.out.printf("sample mean of the second variable = %f%n", sample_mean2);

        // compute the theoretical covariance of the data
        Matrix cov_theoretical = new DenseMatrix(
                new double[][]{
                    {811. / 135., 101. / 45.},
                    {101. / 45., 7. / 3.}
                });
        System.out.println("theoretical covariance =");
        System.out.println(cov_theoretical);

        // compute the sample covariance of the data
        SampleCovariance sample_cov = new SampleCovariance(dataM);
        System.out.println("sample covariance =");
        System.out.println(sample_cov);
    }

    public void VARMA_autocovariance() {
        System.out.println("autocovariance function for a causal VARMA");

        // the AR coefficients
        Matrix[] AR = new Matrix[1];
        AR[0] = new DenseMatrix(
                new double[][]{
                    {0.5, 0.5},
                    {0, 0.5}});

        // the MA coefficients
        Matrix[] MA = new Matrix[1];
        MA[0] = AR[0].t();

        // SIGMA
        Matrix SIGMA
                = new DenseMatrix(
                        new double[][]{
                            {1, 0.2},
                            {0.2, 1}
                        });

        // define a VARIMA(1,1) model
        VARMAModel varma11 = new VARMAModel(AR, MA, SIGMA);

        // comptue the autocovariance function for a VARIMA process up to a certian number of lags
        VARMAAutoCovariance GAMMA
                = new VARMAAutoCovariance(
                        varma11,
                        10 // number of lags
                );

        // print out the autocovariance function 
        for (int i = 0; i <= 5; ++i) {
            System.out.printf("GAMMA(%d) = %n", i);
            System.out.println(GAMMA.evaluate(1));
            System.out.println();
        }
    }

    /**
     * P. J. Brockwell and R. A. Davis, "Example 5.2.1. Chapter 5. Multivariate
     * Time Series," in <i>Time Series: Theory and Methods</i>, Springer, 2006.
     */
    public void VARMA_forecast() {
        System.out.println("forecast using the innovation algorithm");

        // MA(1) model parameters
        final double theta = -0.9;
        final double sigma = 1;

        // a multivariate time series (although the dimension is just 1)
        MultivariateIntTimeTimeSeries Xt
                = new MultivariateSimpleTimeSeries(
                        new double[][]{
                            {-2.58},
                            {1.62},
                            {-0.96},
                            {2.62},
                            {-1.36}
                        });

        // the autocovariance function
        MultivariateAutoCovarianceFunction K = new MultivariateAutoCovarianceFunction() {

            @Override
            public Matrix evaluate(double x1, double x2) {
                int i = (int) x1;
                int j = (int) x2;

                double k = 0;

                if (i == j) {
                    k = sigma * sigma;
                    k *= 1 + theta * theta;
                }

                if (Math.abs(j - i) == 1) {
                    k = theta;
                    k *= sigma * sigma;
                }

                // γ = 0 otherwise
                DenseMatrix result = new DenseMatrix(new double[][]{{k}});
                return result;
            }
        };

        // run the innovation algorithm
        MultivariateForecastOneStep forecast
                = new MultivariateForecastOneStep(Xt, K);

        // making forecasts
        for (int i = 0; i <= 5; ++i) {
            System.out.println(Arrays.toString(forecast.xHat(i).toArray()));
        }
    }

    public void VAR() {
        System.out.println("VAR models");

        // construct a VAR(2) model
        Vector MU = new DenseVector(new double[]{1., 2.});
        Matrix[] PHI = new Matrix[]{
            new DenseMatrix(new double[][]{
                {0.2, 0.3},
                {0., 0.4}}),
            new DenseMatrix(new double[][]{
                {0.1, 0.2},
                {0.3, 0.1}})
        };
        VARModel model0 = new VARModel(MU, PHI);

        // construct a RNG from the model
        VARIMASim sim = new VARIMASim(model0);
        sim.seed(1234567891L);

        // generate a random multivariate time series
        final int N = 5000;
        double[][] ts = new double[N][];
        for (int i = 0; i < N; ++i) {
            ts[i] = sim.nextVector();
        }
        MultivariateIntTimeTimeSeries mts
                = new MultivariateSimpleTimeSeries(ts);

        // fit the data to a VAR(2) model
        VARModel model1 = new VARFit(mts, 2);
        System.out.println("μ = ");
        System.out.println(model1.mu());
        System.out.println("ϕ_1 =");
        System.out.println(model1.AR(1));
        System.out.println("ϕ_2 =");
        System.out.println(model1.AR(2));
        System.out.println("sigma =");
        System.out.println(model1.sigma());
    }

    public void VARIMA() {
        System.out.println("VAR, VMA, VARMA, VARIMA and all those");

        // construct a VAR(1) model
        Matrix[] PHI = new Matrix[1];
        PHI[0] = new DenseMatrix(
                new double[][]{
                    {0.7, 0.12},
                    {0.31, 0.6}
                });
        VARModel var1 = new VARModel(PHI);
        System.out.println("unconditional mean = " + var1.unconditionalMean());

        // construct a VAR(1) model
        Matrix[] THETA = new Matrix[1];
        THETA[0] = new DenseMatrix(
                new double[][]{
                    {0.5, 0.16},
                    {-0.7, 0.28}
                });
        VMAModel vma1 = new VMAModel(THETA);
        System.out.println("unconditional mean = " + vma1.unconditionalMean());

        // construct a VARIMA(1,1) model
        Matrix PHI1 = new DenseMatrix(
                new double[][]{
                    {0.7, 0},
                    {0, 0.6}
                });
        Matrix THETA1 = new DenseMatrix(
                new double[][]{
                    {0.5, 0.6},
                    {-0.7, 0.8}
                });
        VARMAModel varma11 = new VARMAModel(
                new Matrix[]{PHI1},
                new Matrix[]{THETA1}
        );
        System.out.println("unconditional mean = " + varma11.unconditionalMean());

        // the AR coefficients
        Matrix[] PHI_1 = new Matrix[4];
        PHI_1[0] = new DenseMatrix(new DenseVector(new double[]{0.3}));
        PHI_1[1] = new DenseMatrix(new DenseVector(new double[]{-0.2}));
        PHI_1[2] = new DenseMatrix(new DenseVector(new double[]{0.05}));
        PHI_1[3] = new DenseMatrix(new DenseVector(new double[]{0.04}));

        // the MA coefficients
        Matrix[] THETA_1 = new Matrix[2];
        THETA_1[0] = new DenseMatrix(new DenseVector(new double[]{0.2}));
        THETA_1[1] = new DenseMatrix(new DenseVector(new double[]{0.5}));

        // the order of integration
        int d = 1;

        // construct a VARIMA(1,1,1) model
        VARIMAModel varima111 = new VARIMAModel(PHI_1, d, THETA_1);
    }

    public void spx_aapl() throws IOException {
        System.out.println("SPX and APPL");

        // read the monthly S&P 500 data from a csv file
        double[][] spx_arr
                = DoubleUtils.readCSV2d(
                        this.getClass().getClassLoader().getResourceAsStream("sp500_monthly.csv"),
                        true,
                        true
                );
        // convert the csv file into a matrix for manipulation
        Matrix spx_M = new DenseMatrix(spx_arr);
        // extract the column of adjusted close prices
        Vector spx_v = spx_M.getColumn(5); // adjusted closes

        // read the monthly AAPL data from a csv file
        double[][] appl_arr
                = DoubleUtils.readCSV2d(
                        this.getClass().getClassLoader().getResourceAsStream("AAPL_monthly.csv"),
                        true,
                        true
                );
        // convert the csv file into a matrix for manipulation
        Matrix aapl_M = new DenseMatrix(appl_arr);
        // extract the column of adjusted close prices
        Vector aapl_v = aapl_M.getColumn(5); // adjusted closes

        // combine SPX and AAPL to form a bivariate time series
        MultivariateSimpleTimeSeries mts
                = new MultivariateSimpleTimeSeries(cbind(spx_v, aapl_v));
        System.out.println("(spx, aapl) prices: \n" + mts);

        // compute the bivariate time series of log returns
        Matrix p1 = mts.drop(1).toMatrix();
        Matrix p = mts.toMatrix();
        Matrix log_returns_M = p1.ZERO(); // allocate space for the new matrix
        for (int i = 1; i <= p1.nRows(); i++) { // matrix and vector index starts from 1
            double r = log(p1.get(i, 1)) - log(p.get(i, 1));
            log_returns_M.set(i, 1, r);
            r = log(p1.get(i, 2)) - log(p.get(i, 2));
            log_returns_M.set(i, 2, r);
        }
        // convert a matrix to a multivariate time series
        MultivariateSimpleTimeSeries log_returns
                = new MultivariateSimpleTimeSeries(log_returns_M);
        System.out.println("(spx, aapl) log_returns: \n" + log_returns);

        // fit the log returns to an VAR(1) model
        VARFit fit = new VARFit(log_returns, 1);
        System.out.println("the estimated phi_1 for var(1) is");
        System.out.println(fit.AR(1));
        VARMAModel log_returns2 = fit.getDemeanedModel(); // demeaned version

        // predict the future values using the innovation algorithm
        MultivariateAutoCovarianceFunction K
                = new VARMAAutoCovariance(log_returns2, log_returns.size());
        MultivariateForecastOneStep forecast
                = new MultivariateForecastOneStep(log_returns, K);
        for (int i = 0; i <= 5; ++i) {
            System.out.println(Arrays.toString(forecast.xHat(i).toArray()));
        }

    }

    public void mts() {
        System.out.println("multivariate time series");

        // construct a bivariate time series
        MultivariateSimpleTimeSeries X_T0
                = new MultivariateSimpleTimeSeries(
                        new double[][]{
                            {-1.875, 1.693},
                            {-2.518, -0.03},
                            {-3.002, -1.057},
                            {-2.454, -1.038},
                            {-1.119, -1.086},
                            {-0.72, -0.455},
                            {-2.738, 0.962},
                            {-2.565, 1.992},
                            {-4.603, 2.434},
                            {-2.689, 2.118}
                        });
        System.out.println("X_T0: " + X_T0);

        // first difference of the bivariate time series
        MultivariateSimpleTimeSeries X_T1 = X_T0.diff(1);
        System.out.println("diff(1): " + X_T1);

        // first order lagged time series
        MultivariateSimpleTimeSeries X_T2 = X_T0.lag(1);
        System.out.println("lag(1): " + X_T2);

        // drop the first two items
        MultivariateSimpleTimeSeries X_T3 = X_T0.drop(1);
        System.out.println("drop(1): " + X_T3);
    }

    public void garch_models() {
        System.out.println("ARCH and GARCH models");

        // σ_t^2 = 0.2 + 0.212 * ε_(t-1)^2
        GARCHModel arch1
                = new GARCHModel(
                        0.2, // constant
                        new double[]{0.212}, // ARCH terms
                        new double[]{} // no GARCH terms
                );
        System.out.println(arch1);
        System.out.printf("conditional variance = %f%n", arch1.var());

        // σ_t^2= 0.2+0.212ε_(t-1)^2+0.106σ_(t-1)^2
        GARCHModel garch11
                = new GARCHModel(
                        0.2, // constant
                        new double[]{0.212}, // ARCH terms
                        new double[]{0.106} // GARCH terms
                );
        System.out.println(garch11);
        System.out.printf("conditional variance = %f%n", garch11.var());

        // simulate the GARCH process
        GARCHSim sim = new GARCHSim(garch11);
        sim.seed(1234567890L);
        double[] series = RNGUtils.nextN(sim, 200);
        System.out.println(Arrays.toString(series));
    }

    /**
     *
     * x<-c(0.2,0.3,-0.1,0.4,-0.5,0.6,0.1,0.2) adf.test(c(x,x),alternative =
     * c("stationary"),4)
     *
     * The p-value from R is obtained form the interpolation
     * (0.9-0.1)/(3.24-1.14)=(x-0.1)/(3.24-2.642)
     * "x=0.8/(3.24-1.14)*(3.24-2.642)+0.1=0.3278"
     *
     * The values -3.24 and -1.14 are from table 4.2c in A. Banerjee, J. J.
     * Dolado, J. W. Galbraith, and D. F. Hendry (1993): Cointegration, Error
     * Correction, and the Econometric Analysis of Non-Stationary Data, Oxford
     * University Press, Oxford.
     */
    public void adf() {
        System.out.println("unit root test using the augmented Dickey-Fuller test");

        double[] sample = new double[]{0.2, 0.3, -0.1, 0.4, -0.5, 0.6, 0.1, 0.2, 0.2, 0.3, -0.1, 0.4, -0.5, 0.6, 0.1, 0.2};
        AugmentedDickeyFuller adf
                = new AugmentedDickeyFuller(
                        sample,
                        TrendType.CONSTANT, // constant drift term
                        4, // the lag order
                        null
                );

        System.out.println("H0: " + adf.getNullHypothesis());
        System.out.println("H1: " + adf.getAlternativeHypothesis());
        System.out.printf("the p-value for the test = %f%n", adf.pValue());
        System.out.printf("the statistics for the test = %f%n", adf.statistics());
    }

    public void goodness_of_fit() {
        System.out.println("check the goodness of fit using the Ljung-Box test");

        // define an ARMA(1, 1)
        ARMAModel arma11 = new ARMAModel(
                new double[]{0.2}, // the AR coefficients
                new double[]{1.1} // the MA coefficients
        );

        // create a random number generator from the ARMA model
        ARIMASim sim = new ARIMASim(arma11);
        sim.seed(1234567890L);

        // generate a random time series
        final int T = 50; // length of the time series
        double[] x = new double[T];
        for (int i = 0; i < T; ++i) {
            // call the RNG to generate random numbers according to the specification
            x[i] = sim.nextDouble();
        }

        // fit an ARMA(1,1) model
        ConditionalSumOfSquares css
                = new ConditionalSumOfSquares(x, 1, 0, 1);
        ARMAModel arma11_css = css.getARMAModel();
        System.out.printf("The fitted ARMA(1,1) model is %s%n", arma11_css);
        System.out.printf("The fitted ARMA(1,1) model, demeanded, is %s%n", arma11_css.getDemeanedModel());
        System.out.printf("AIC = %f%n", css.AIC());

        // compute the fitted values using the fitted model
        ARMAForecastOneStep forecaset = new ARMAForecastOneStep(x, arma11_css);
        double[] x_hat = new double[T];
        double[] residuals = new double[T];
        for (int i = 0; i < T; ++i) {
            // the fitted value
            x_hat[i] = forecaset.xHat(i);
            // the residuals
            residuals[i] = x[i] - x_hat[i];
        }

        System.out.println(Arrays.toString(x));
        System.out.println(Arrays.toString(x_hat));
        System.out.println(Arrays.toString(residuals));

        // run the goodness-of-fit test
        LjungBox test = new LjungBox(
                residuals,
                4, // check up to the 4-th lag
                2 // number of parameters
        );
        System.out.printf("The test statistic = %f%n", test.statistics());
        System.out.printf("The p-value = %f%n", test.pValue());
    }

    public void arma_forecast() {
        System.out.println("making forecasts for an ARMA model");

        // define an ARMA(1, 1)
        ARMAModel arma11 = new ARMAModel(
                new double[]{0.2}, // the AR coefficients
                new double[]{1.1} // the MA coefficients
        );

        // create a random number generator from the ARMA model
        ARIMASim sim = new ARIMASim(arma11);
        sim.seed(1234567890L);

        // generate a random time series
        final int T = 50; // length of the time series
        double[] x = new double[T];
        for (int i = 0; i < T; ++i) {
            // call the RNG to generate random numbers according to the specification
            x[i] = sim.nextDouble();
        }

        // compute the one-step conditional expectations for the model
        ARMAForecastOneStep forecaset = new ARMAForecastOneStep(x, arma11);
        double[] x_hat = new double[T];
        double[] residuals = new double[T];
        for (int i = 0; i < T; ++i) {
            // the one-step conditional expectations
            x_hat[i] = forecaset.xHat(i);
            // the errors
            residuals[i] = x[i] - x_hat[i];
        }

        System.out.println(Arrays.toString(x));
        System.out.println(Arrays.toString(x_hat));
        System.out.println(Arrays.toString(residuals));
    }

    public void arma23() throws IOException {
        System.out.println("ARMA(2,3) model");

        // build an ARMA(2, 3)
        ARMAModel arma23 = new ARMAModel(
                new double[]{0.6, -0.23}, // the AR coefficients
                new double[]{0.1, 0.2, 0.4} // the MA coefficients
        );

        // compute the linear representation
        LinearRepresentation ma = new LinearRepresentation(arma23);
        for (int i = 1; i <= 20; i++) {
            System.out.printf("the coefficients of the linear representation at lag %d = %f%n", i, ma.AR(i));
        }

        // compute the autocovariance function for the model
        AutoCovariance acvf = new AutoCovariance(arma23);
        for (int i = 1; i < 10; i++) {
            System.out.printf("the acvf of the ARMA(2,3) model at lag%d: %f%n", i, acvf.evaluate(i));
        }

        // compute the autocorrelation function for the model
        AutoCorrelation acf = new AutoCorrelation(arma23, 10);
        for (int i = 0; i < 10; i++) {
            System.out.printf("the acf of the ARMA(2,3) model at lag%d: %f%n", i, acf.evaluate(i));
        }
    }

    public void arma11() throws IOException {
        System.out.println("ARMA(1,1) model");

        // build an ARMA(1, 1)
        ARMAModel arma11 = new ARMAModel(
                new double[]{0.2}, // the AR coefficients
                new double[]{1.1} // the MA coefficients
        );

        // compute the linear representation
        LinearRepresentation ma = new LinearRepresentation(arma11);
        for (int i = 1; i <= 20; i++) {
            System.out.printf("the coefficients of the linear representation at lag %d = %f%n", i, ma.AR(i));
        }

        // compute the autocovariance function for the model
        AutoCovariance acvf = new AutoCovariance(arma11);
        for (int i = 1; i < 10; i++) {
            System.out.printf("the acvf of the ARMA(1,1) model at lag%d: %f%n", i, acvf.evaluate(i));
        }

        // compute the autocorrelation function for the model
        AutoCorrelation acf = new AutoCorrelation(arma11, 10);
        for (int i = 0; i < 10; i++) {
            System.out.printf("the acf of the ARMA(1,1) model at lag%d: %f%n", i, acf.evaluate(i));
        }

    }

    public void ma() throws IOException {
        System.out.println("MA models");

        // define an MA(1) model
        MAModel ma1 = new MAModel(
                new double[]{0.8} // theta_1 = 0.8
        );

        final int nLags = 10;
        // compute the autocovariance function for the model
        AutoCovariance acvf1 = new AutoCovariance(ma1);
        for (int i = 0; i < nLags; i++) {
            System.out.printf("the acvf of the MA(1) model at lag%d = %f%n", i, acvf1.evaluate(i));
        }
        // compute the autocorrelation function for the model
        AutoCorrelation acf1 = new AutoCorrelation(ma1, nLags);
        for (int i = 0; i < nLags; i++) {
            System.out.printf("the acf of the MA(1) model at lag%d = %f%n", i, acf1.evaluate(i));
        }

        // define an MA(2) model
        MAModel ma2 = new MAModel(
                new double[]{-0.2, 0.01} // the moving-average coefficients
        );
        AutoCovariance acvf2 = new AutoCovariance(ma2);
        for (int i = 1; i < 10; i++) {
            System.out.printf("the acvf of the MA(2) model at lag%d: %f%n", i, acvf2.evaluate(i));
        }
        AutoCorrelation acf2 = new AutoCorrelation(ma2, 10);
        for (int i = 0; i < 10; i++) {
            System.out.printf("the acf of the MA(2) model at lag%d: %f%n", i, acf2.evaluate(i));
        }

        // define an MA(2) model
        MAModel ma3 = new MAModel(
                0.1, // the mean
                new double[]{-0.5, 0.01}, // the moving-average coefficients
                10. // the standard deviation
        );

        // create a random number generator from the MA model
        ARIMASim sim = new ARIMASim(ma3);
        sim.seed(1234567890L);

        // generate a random time series
        final int T = 500; // length of the time series
        double[] x = new double[T];
        for (int i = 0; i < T; ++i) {
            // call the RNG to generate random numbers according to the specification
            x[i] = sim.nextDouble();
        }

        // determine the cutoff lag using autocorrelation
        SampleAutoCorrelation acf3_hat
                = new SampleAutoCorrelation(new SimpleTimeSeries(x));
        for (int i = 0; i < 5; i++) {
            System.out.printf("the empirical acf of a time series at lag%d: %f%n", i, acf3_hat.evaluate(i));
        }

        // fit an MA(2) model using the data
        ConditionalSumOfSquares fit
                = new ConditionalSumOfSquares(
                        x,
                        0,
                        0,
                        2); // the MA order
        System.out.printf("theta: %s%n", Arrays.toString(fit.getARMAModel().theta()));
        System.out.printf("var of error term: %s%n", fit.var());

        // make forecast
        ARMAForecast forecast_ma3
                = new ARMAForecast(new SimpleTimeSeries(x), ma3);
        System.out.println("The forecasts for the MA(2) model are");
        for (int j = 0; j < 10; ++j) {
            System.out.println(forecast_ma3.next());
        }
    }

    public void ar() throws IOException {
        System.out.println("AR models");

        // define an AR(1) model
        ARModel ar1 = new ARModel(
                new double[]{0.6} // phi_1 = 0.6
        );
        final int nLags = 10;
        // compute the autocovariance function for the model
        AutoCovariance acvf1 = new AutoCovariance(ar1);
        for (int i = 0; i < nLags; i++) {
            System.out.printf("the acvf of the AR(1) model at lag%d = %f%n", i, acvf1.evaluate(i));
        }
        // compute the autocorrelation function for the model
        AutoCorrelation acf1 = new AutoCorrelation(ar1, nLags);
        for (int i = 0; i < nLags; i++) {
            System.out.printf("the acf of the AR(1) model at lag%d = %f%n", i, acf1.evaluate(i));
        }

        // define an AR(2) model
        ARModel ar2 = new ARModel(
                new double[]{1.2, -0.35}
        );
        // compute the autocovariance function for the model
        AutoCovariance acvf2 = new AutoCovariance(ar2);
        for (int i = 1; i < nLags; i++) {
            System.out.printf("the acvf of the AR(2) model at lag%d = %f%n", i, acvf2.evaluate(i));
        }
        // compute the autocorrelation function for the model
        AutoCorrelation acf2 = new AutoCorrelation(ar2, 10);
        for (int i = 0; i < nLags; i++) {
            System.out.printf("the acf of the AR(2) model at lag%d = %f%n", i, acf2.evaluate(i));
        }

        // create a random number generator from the AR model
        ARIMASim sim = new ARIMASim(ar2);
        sim.seed(1234567890L);

        // generate a random time series
        final int T = 500; // length of the time series
        double[] x = new double[T];
        for (int i = 0; i < T; ++i) {
            // call the RNG to generate random numbers according to the specification
            x[i] = sim.nextDouble();
        }

        // determine the cutoff lag using partial autocorrelation
        SamplePartialAutoCorrelation acf2_hat
                = new SamplePartialAutoCorrelation(new SimpleTimeSeries(x));
        for (int i = 1; i < 5; i++) {
            System.out.printf("the empirical pacf of a time series at lag%d: %f%n", i, acf2_hat.evaluate(i));
        }

        // fit an AR(2) model using the data
        ConditionalSumOfSquares fit
                = new ConditionalSumOfSquares(
                        x,
                        2, // the AR order
                        0,
                        0);
        System.out.printf("phi: %s%n", Arrays.toString(fit.getARMAModel().phi()));
        System.out.printf("var of error term: %s%n", fit.var());

        // make forecast
        ARMAForecast forecast_ar2
                = new ARMAForecast(new SimpleTimeSeries(x), ar2);
        System.out.println("The forecasts for the AR(2) model are");
        for (int j = 0; j < 10; ++j) {
            System.out.println(forecast_ar2.next());
        }
    }

    public void decomposition() throws IOException {
        System.out.println("classical time series decomposition");

        RandomNumberGenerator rng = new StandardNormalRNG();

        // construct an additive model
        AdditiveModel additive_model = new AdditiveModel(
                // the trend
                new double[]{
                    1, 3, 5, 7, 9, 11, 13, 15, 17, 19
                },
                // the seasonal component
                new double[]{
                    0, 1
                },
                // the source of randomness
                rng
        );
        System.out.println(additive_model);

        // construct a multiplicative model
        MultiplicativeModel multiplicative_model = new MultiplicativeModel(
                // the trend
                new double[]{
                    1, 3, 5, 7, 9, 11, 13, 15, 17, 19
                },
                // the seasonal component
                new double[]{
                    -1, 1
                },
                // the source of randomness
                rng
        );
        System.out.println(multiplicative_model);

    }

    public void sp500_monthly() throws IOException {
        System.out.println("univariate time series of S&P 500 monthly close");

        // read the monthly S&P 500 data from a csv file
        double[][] spx_arr
                = DoubleUtils.readCSV2d(
                        this.getClass().getClassLoader().getResourceAsStream("sp500_monthly.csv"),
                        true,
                        true
                );
        // convert the csv file into a matrix for manipulation
        Matrix spx_M = new DenseMatrix(spx_arr);
        // extract the column of adjusted close prices
        Vector spx_v = spx_M.getColumn(5); // adjusted closes
        // construct a univariate time series from the data
        SimpleTimeSeries spx = new SimpleTimeSeries(spx_v.toArray());
//        System.out.println(spx);

        // ACVF for lags 0, 1, 2, ..., 24
        SampleAutoCovariance acvf = new SampleAutoCovariance(spx);
        for (int i = 0; i < 25; i++) {
            System.out.printf("acvf(%d) = %f%n", i, acvf.evaluate(i));
        }

        // ACF for lags 0, 1, 2, ..., 24
        SampleAutoCorrelation acf = new SampleAutoCorrelation(spx);
        for (int i = 0; i < 25; i++) {
            System.out.printf("acf(%d) = %f%n", i, acf.evaluate(i));
        }

        // PACF for lags 1, 2, ..., 24
        SamplePartialAutoCorrelation pacf = new SamplePartialAutoCorrelation(spx);
        for (int i = 1; i < 25; i++) {
            System.out.printf("pacf(%d) = %f%n", i, pacf.evaluate(i));
        }

        // decompose the S&P 500 monthly returns into a classical additive model
        MADecomposition spx_model
                = new MADecomposition(
                        spx.toArray(), // the SP 500 monthly returns
                        12 // the lenght of seasonality
                );
        System.out.printf("the trend component: %s%n", Arrays.toString(spx_model.getTrend()));
        System.out.printf("the seasonal component: %s%n", Arrays.toString(spx_model.getSeasonal()));
        System.out.printf("the random component: %s%n", Arrays.toString(spx_model.getRandom()));

        // auto fit an ARIMA model to S&P 500 prices
        AutoARIMAFit fit_sp500_1 = new AutoARIMAFit(spx.toArray(), 5, 1, 5, 0, 0, 1000);
        ARIMAModel arima1 = fit_sp500_1.optimalModelByAIC();
        ARIMAModel arima2 = fit_sp500_1.optimalModelByAICC();
        System.out.println("The optimal model for S&P500 prices by AIC is:");
        System.out.println(arima1);
        System.out.println("The optimal model for S&P500 prices by AICc is:");
        System.out.println(arima2);

        // compute log returns from prices
        double[] p1 = spx.drop(1).toArray();
        double[] p = spx.toArray();
        double[] log_returns = new double[p1.length];
        for (int i = 0; i < p1.length; i++) {
            log_returns[i] = log(p1[i]) - log(p[i]);
        }
        System.out.println(Arrays.toString(log_returns));

        // auto fit an ARIMA model to S&P 500 log returns
        AutoARIMAFit fit_sp500_2 = new AutoARIMAFit(log_returns, 5, 1, 5, 0, 0, 1000);
        ARIMAModel arima3 = fit_sp500_2.optimalModelByAIC();
        ARIMAModel arima4 = fit_sp500_2.optimalModelByAICC();
        System.out.println("The optimal model for S&P500 log returns by AIC is:");
        System.out.println(arima3);
        System.out.println("The optimal model for S&P500 log returns by AICc is:");
        System.out.println(arima4);

        // make price forecasts based on the fitted model
        ARIMAForecast forecast_prices = new ARIMAForecast(spx, arima1);
        System.out.println("The next 10 price forecasts for sp500 using the ARIMA(2,1,1) model is:");
        for (int i = 0; i < 10; i++) {
            System.out.println(forecast_prices.next());
        }

        // make log return forecasts based on the fitted model
        ARIMAForecast forecast_returns = new ARIMAForecast(new SimpleTimeSeries(log_returns), arima4);
        System.out.println("The next 10 log return forecasts for sp500 using the ARMA(2, 3) model is:");
        for (int i = 0; i < 10; i++) {
            System.out.println(forecast_returns.next());
        }
    }

    public void sp500_daily() throws IOException {
        System.out.println("univariate time series of S&P 500 daily close");

        // read the daily S&P 500 data from a csv file
        double[][] spx_arr
                = DoubleUtils.readCSV2d(
                        this.getClass().getClassLoader().getResourceAsStream("sp500_daily.csv"),
                        true,
                        true
                );
        // convert the csv file into a matrix for manipulation
        Matrix spx_M = new DenseMatrix(spx_arr);
        // extract the column of adjusted close prices
        Vector spx_v = spx_M.getColumn(5); // adjusted closes
        // construct a univariate time series from the data
        SimpleTimeSeries spx = new SimpleTimeSeries(spx_v.toArray());
        System.out.println(spx);

        // compute simple returns from prices
        double[] diff = spx.diff(1).toArray();
        double[] p0 = spx.lag(1).toArray();
        double[] simple_returns = new double[diff.length];
        for (int i = 0; i < diff.length; ++i) {
            simple_returns[i] = diff[i] / p0[i];
        }
        System.out.println(Arrays.toString(simple_returns));

        // compute log returns from prices
        double[] p1 = spx.drop(1).toArray();
        double[] p = spx.toArray();
        double[] log_returns = new double[p1.length];
        for (int i = 0; i < p0.length; i++) {
            log_returns[i] = log(p1[i]) - log(p[i]);
        }
        System.out.println(Arrays.toString(log_returns));

        // run the Ljung-Box test
        LjungBox lb_test = new LjungBox(log_returns, 14, 0);
        System.out.printf("The null hypothesis is: %s%n", lb_test.getNullHypothesis());
        System.out.printf("The alternative hypothesis is: %s%n", lb_test.getAlternativeHypothesis());
        System.out.printf("The test statistic = %f%n", lb_test.statistics());
        System.out.printf("The p-value = %f%n", lb_test.pValue());

        AugmentedDickeyFuller adf
                = new AugmentedDickeyFuller(spx.toArray());
        System.out.println(adf.getNullHypothesis());
        System.out.println(adf.getAlternativeHypothesis());
        System.out.printf("the p-value for the test = %f%n", adf.pValue());
        System.out.printf("the statistics for the test = %f%n", adf.statistics());

        // auto fit an ARIMA model to S&P 500 prices
        AutoARIMAFit fit_sp500_1 = new AutoARIMAFit(spx.toArray(), 5, 1, 5, 0, 0, 1000);
        ARIMAModel arima1 = fit_sp500_1.optimalModelByAIC();
        ARIMAModel arima2 = fit_sp500_1.optimalModelByAICC();
        System.out.println("The optimal model for S&P500 prices by AIC is:");
        System.out.println(arima1);
        System.out.println("The optimal model for S&P500 prices by AICc is:");
        System.out.println(arima2);

        // auto fit an ARIMA model to S&P 500 log returns
        AutoARIMAFit fit_sp500_2 = new AutoARIMAFit(log_returns, 5, 1, 5, 0, 0, 1000);
        ARIMAModel arima3 = fit_sp500_2.optimalModelByAIC();
        ARIMAModel arima4 = fit_sp500_2.optimalModelByAICC();
        System.out.println("The optimal model for S&P500 log returns by AIC is:");
        System.out.println(arima3);
        System.out.println("The optimal model for S&P500 log returns by AICc is:");
        System.out.println(arima4);

        // make price forecasts based on the fitted model
        ARIMAForecast forecast_prices = new ARIMAForecast(spx, arima1);
        System.out.println("The next 10 price forecasts for sp500 using the ARIMA(2,1,1) model is:");
        for (int i = 0; i < 10; i++) {
            System.out.println(forecast_prices.next());
        }

        // make log return forecasts based on the fitted model
        ARIMAForecast forecast_returns
                = new ARIMAForecast(new SimpleTimeSeries(log_returns), arima4);
        System.out.println("The next 10 log return forecasts for sp500 using the ARMA(2, 3) model is:");
        for (int i = 0; i < 10; i++) {
            System.out.println(forecast_returns.next());
        }

        // compute the conditional means of the model at each time
        ARMAForecastOneStep log_returns_hat
                = new ARMAForecastOneStep(log_returns, arima4.getARMA());

        // compute the residuals = observations - fitted values
        double[] residuals = new double[log_returns.length];
        for (int t = 0; t < log_returns.length; ++t) {
            residuals[t] = log_returns[t] - log_returns_hat.xHat(t);
        }
        System.out.println("residuals:");
        System.out.println(Arrays.toString(residuals));

        // fit the residuals to a GARCH(1,1) model
        GARCHFit garch_fit = new GARCHFit(residuals, 1, 1);
        GARCHModel garch = garch_fit.getModel();
        System.out.printf("the residual GARCH(1,1) model is: %s%n", garch);

        // fit both the ARMA and GARCH models in one go
        ARMAGARCHFit arma_garch_fit
                = new ARMAGARCHFit(
                        log_returns, // the input series
                        2, // the order for AR in ARMA model
                        3, // the order for MA in ARMA model
                        1, // the order for GARCH in GARCH model
                        1 // the order for ARCH in GARCH model
                );
        System.out.println("the ARMA model is:");
        System.out.println(arma_garch_fit.getARMAGARCHModel().getARMAModel());
        System.out.println("the GARCH model is:");
        System.out.println(arma_garch_fit.getARMAGARCHModel().getGARCHModel());
    }

    public void univariate_ts() throws IOException {
        System.out.println("univariate time series");

        // construct a time series
        SimpleTimeSeries ts1 = new SimpleTimeSeries(new double[]{1, 2, 3, 4, 5, 6, 7, 8, 9, 10});
        System.out.println("ts1:" + ts1);

        // construct a time series by taking the first difference of another one
        // each term is the difference between two successive terms in the original time series
        SimpleTimeSeries ts2 = ts1.diff(1);
        System.out.println("ts2:" + ts2);

        // construct a time series by dropping the first two terms of another one
        SimpleTimeSeries ts3 = ts1.drop(2);
        System.out.println("ts3:" + ts3);

        // construct a time series by lagging another one
        // This operation makes sense only for equi-distant data points.
        SimpleTimeSeries ts4 = ts1.lag(2);
        System.out.println("ts4:" + ts4);
    }
}
