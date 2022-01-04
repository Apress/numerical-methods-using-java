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
import dev.nm.analysis.function.rn2r1.univariate.AbstractUnivariateRealFunction;
import dev.nm.analysis.function.rn2r1.univariate.UnivariateRealFunction;
import dev.nm.analysis.function.special.gaussian.CumulativeNormalMarsaglia;
import dev.nm.analysis.function.special.gaussian.Gaussian;
import dev.nm.analysis.function.special.gaussian.StandardCumulativeNormal;
import dev.nm.interval.RealInterval;
import dev.nm.number.DoubleUtils;
import dev.nm.stat.descriptive.covariance.SampleCovariance;
import dev.nm.stat.descriptive.moment.Kurtosis;
import dev.nm.stat.descriptive.moment.Mean;
import dev.nm.stat.descriptive.moment.Skewness;
import dev.nm.stat.descriptive.moment.Variance;
import dev.nm.stat.distribution.univariate.BetaDistribution;
import dev.nm.stat.distribution.univariate.EmpiricalDistribution;
import dev.nm.stat.distribution.univariate.ExponentialDistribution;
import dev.nm.stat.distribution.univariate.GammaDistribution;
import dev.nm.stat.distribution.univariate.PoissonDistribution;
import dev.nm.stat.distribution.univariate.ProbabilityDistribution;
import dev.nm.stat.random.Estimator;
import dev.nm.stat.random.rng.multivariate.MultinomialRVG;
import dev.nm.stat.random.rng.multivariate.NormalRVG;
import dev.nm.stat.random.rng.multivariate.RandomVectorGenerator;
import dev.nm.stat.random.rng.multivariate.UniformDistributionOverBox;
import dev.nm.stat.random.rng.univariate.InverseTransformSampling;
import dev.nm.stat.random.rng.univariate.RandomLongGenerator;
import dev.nm.stat.random.rng.univariate.RandomNumberGenerator;
import dev.nm.stat.random.rng.univariate.beta.Cheng1978;
import dev.nm.stat.random.rng.univariate.beta.RandomBetaGenerator;
import dev.nm.stat.random.rng.univariate.exp.RandomExpGenerator;
import dev.nm.stat.random.rng.univariate.exp.Ziggurat2000Exp;
import dev.nm.stat.random.rng.univariate.gamma.KunduGupta2007;
import dev.nm.stat.random.rng.univariate.normal.NormalRNG;
import dev.nm.stat.random.rng.univariate.normal.RandomStandardNormalGenerator;
import dev.nm.stat.random.rng.univariate.normal.Zignor2005;
import dev.nm.stat.random.rng.univariate.poisson.Knuth1969;
import dev.nm.stat.random.rng.univariate.uniform.UniformRNG;
import dev.nm.stat.random.rng.univariate.uniform.linear.CompositeLinearCongruentialGenerator;
import dev.nm.stat.random.rng.univariate.uniform.linear.LEcuyer;
import dev.nm.stat.random.rng.univariate.uniform.linear.Lehmer;
import dev.nm.stat.random.rng.univariate.uniform.linear.LinearCongruentialGenerator;
import dev.nm.stat.random.rng.univariate.uniform.mersennetwister.MersenneTwister;
import dev.nm.stat.random.sampler.resampler.BootstrapEstimator;
import dev.nm.stat.random.sampler.resampler.bootstrap.CaseResamplingReplacement;
import dev.nm.stat.random.sampler.resampler.bootstrap.block.PattonPolitisWhite2009;
import dev.nm.stat.random.sampler.resampler.bootstrap.block.PattonPolitisWhite2009ForObject;
import dev.nm.stat.random.variancereduction.AntitheticVariates;
import dev.nm.stat.random.variancereduction.CommonRandomNumbers;
import dev.nm.stat.random.variancereduction.ControlVariates;
import dev.nm.stat.random.variancereduction.ImportanceSampling;
import dev.nm.stat.test.distribution.normality.ShapiroWilk;
import static java.lang.Math.PI;
import static java.lang.Math.sin;
import static java.lang.Math.sqrt;
import java.util.Arrays;

/**
 * Numerical Methods Using Java: For Data Science, Analysis, and Engineering
 *
 * @author haksunli
 * @see
 * https://www.amazon.com/Numerical-Methods-Using-Java-Engineering/dp/1484267966
 * https://nm.dev/
 */
public class Chapter13 {

    public static void main(String[] args) {
        System.out.println("Chapter 13 demos");

        Chapter13 chapter13 = new Chapter13();
        chapter13.lcgs();
        chapter13.MT19937();
        chapter13.normal_rng();
        chapter13.beta_rng();
        chapter13.gamma_rng();
        chapter13.Poisson_rng();
        chapter13.exponential_rng();
        chapter13.compute_Pi();
        chapter13.normal_rvg();
        chapter13.multinomial_rvg();
        chapter13.empirical_rng();
        chapter13.case_resampling_1();
        chapter13.case_resampling_2();
        chapter13.bootstrapping_methods();
        chapter13.crn();
        chapter13.antithetic_variates();
        chapter13.control_variates();
        chapter13.importance_sampling_1();
        chapter13.importance_sampling_2();
    }

    // from
    // https://www.scratchapixel.com/lessons/mathematics-physics-for-computer-graphics/monte-carlo-methods-in-practice/variance-reduction-methods
    private void importance_sampling_2() {
        RandomNumberGenerator rng = new UniformRNG();
        rng.seed(1234567892L);

        int N = 16;
        for (int n = 0; n < 10; ++n) {
            float sumUniform = 0, sumImportance = 0;
            for (int i = 0; i < N; ++i) {
                double r = rng.nextDouble();
                sumUniform += sin(r * PI * 0.5);
                double xi = sqrt(r) * PI * 0.5;
                sumImportance += sin(xi) / ((8 * xi) / (PI * PI));
            }
            sumUniform *= (PI * 0.5) / N;
            sumImportance *= 1.f / N;
            System.out.println(String.format("%f %f\n", sumUniform, sumImportance));
        }
    }

    private void importance_sampling_1() {
        UnivariateRealFunction h = new AbstractUnivariateRealFunction() {

            @Override
            public double evaluate(double x) {
                return x; // the identity function
            }
        };

        UnivariateRealFunction w = new AbstractUnivariateRealFunction() {
            private final Gaussian phi = new Gaussian();
            private final StandardCumulativeNormal N = new CumulativeNormalMarsaglia();
            private final double I = N.evaluate(1) - N.evaluate(0);

            @Override
            public double evaluate(double x) {
                double w = phi.evaluate(x) / I; // the weight
                return w;
            }
        };

        RandomNumberGenerator rng = new UniformRNG();
        rng.seed(1234567892L);

        ImportanceSampling is = new ImportanceSampling(h, w, rng);
        Estimator estimator = is.estimate(100000);
        System.out.println(
                String.format(
                        "mean = %f, variance = %f",
                        estimator.mean(),
                        estimator.variance()));
    }

    private void control_variates() {
        UnivariateRealFunction f
                = new AbstractUnivariateRealFunction() {

            @Override
            public double evaluate(double x) {
                double fx = 1. / (1. + x);
                return fx;
            }
        };

        UnivariateRealFunction g
                = new AbstractUnivariateRealFunction() {

            @Override
            public double evaluate(double x) {
                double gx = 1. + x;
                return gx;
            }
        };

        RandomLongGenerator uniform = new UniformRNG();
        uniform.seed(1234567891L);

        ControlVariates cv
                = new ControlVariates(f, g, 1.5, -0.4773, uniform);
        ControlVariates.Estimator estimator = cv.estimate(1500);
        System.out.println(
                String.format(
                        "mean = %f, variance = %f, b = %f",
                        estimator.mean(),
                        estimator.variance(),
                        estimator.b()));
    }

    private void antithetic_variates() {
        UnivariateRealFunction f
                = new AbstractUnivariateRealFunction() {

            @Override
            public double evaluate(double x) {
                double fx = 1. / (1. + x);
                return fx;
            }
        };

        RandomLongGenerator uniform = new UniformRNG();
        uniform.seed(1234567894L);

        AntitheticVariates av
                = new AntitheticVariates(
                        f,
                        uniform,
                        AntitheticVariates.REFLECTION);
        Estimator estimator = av.estimate(1500);
        System.out.println(
                String.format(
                        "mean = %f, variance = %f",
                        estimator.mean(),
                        estimator.variance()));
    }

    private void crn() {
        final UnivariateRealFunction f
                = new AbstractUnivariateRealFunction() {

            @Override
            public double evaluate(double x) {
                double fx = 2. - Math.sin(x) / x;
                return fx;
            }
        };

        final UnivariateRealFunction g
                = new AbstractUnivariateRealFunction() {

            @Override
            public double evaluate(double x) {
                double gx = Math.exp(x * x) - 0.5;
                return gx;
            }
        };

        RandomLongGenerator X1 = new UniformRNG();
        X1.seed(1234567890L);

        CommonRandomNumbers crn0
                = new CommonRandomNumbers(
                        f,
                        g,
                        X1,
                        new AbstractUnivariateRealFunction() { // another independent uniform RNG
                    final RandomLongGenerator X2 = new UniformRNG();

                    {
                        X2.seed(246890123L);
                    }

                    @Override
                    public double evaluate(double x) {
                        return X2.nextDouble();
                    }
                });
        Estimator estimator0 = crn0.estimate(100_000);
        System.out.println(
                String.format("d = %f, variance = %f",
                        estimator0.mean(),
                        estimator0.variance()));

        CommonRandomNumbers crn1
                = new CommonRandomNumbers(f, g, X1); // use X1 for both f and g
        Estimator estimator1 = crn1.estimate(100_000);
        System.out.println(
                String.format("d = %f, variance = %f",
                        estimator1.mean(),
                        estimator1.variance()));
    }

    /**
     * Constructs a dependent sequence (consisting of 0 and 1) by retaining the
     * last value with probability <i>q</i> while changing the last value with
     * probability <i>1-q</i>.
     * <p/>
     * The simple bootstrapping method {@linkplain CaseResamplingReplacement}
     * will severely overestimate the occurrences of certain pattern, while
     * block bootstrapping method {@linkplain BlockBootstrap} gives a good
     * estimation of the occurrences in the original sample. All estimators over
     * estimate.
     */
    private void bootstrapping_methods() {
        final int N = 10000;
        final double q = 0.70; // the probability of retaining last value

        UniformRNG uniformRNG = new UniformRNG();
        uniformRNG.seed(1234567890L);

        // generate a randome series of 0s and 1s with serial correlation
        final double[] sample = new double[N];
        sample[0] = uniformRNG.nextDouble() > 0.5 ? 1 : 0;
        for (int i = 1; i < N; ++i) {
            sample[i] = uniformRNG.nextDouble() < q ? sample[i - 1] : 1 - sample[i - 1];
        }

        // simple case resampling with replacement method
        CaseResamplingReplacement simpleBoot
                = new CaseResamplingReplacement(sample, uniformRNG);
        Mean countInSimpleBootstrap = new Mean();

        RandomNumberGenerator rlg = new Ziggurat2000Exp();
        rlg.seed(1234567890L);

        // Patton-Politis-White method using stationary blocks
        PattonPolitisWhite2009 stationaryBlock
                = new PattonPolitisWhite2009(
                        sample,
                        PattonPolitisWhite2009ForObject.Type.STATIONARY,
                        uniformRNG,
                        rlg);
        Mean countInStationaryBlockBootstrap = new Mean();

        // Patton-Politis-White method using circular blocks
        PattonPolitisWhite2009 circularBlock
                = new PattonPolitisWhite2009(
                        sample,
                        PattonPolitisWhite2009ForObject.Type.CIRCULAR,
                        uniformRNG,
                        rlg);
        Mean countInCircularBlockBootstrap = new Mean();

        // change this line to use a different pattern
        final double[] pattern = new double[]{1, 0, 1, 0, 1};

        final int B = 10000;
        for (int i = 0; i < B; ++i) {
            // count the number of occurrences for the pattern in the series
            int numberOfMatches = match(simpleBoot.newResample(), pattern);
            countInSimpleBootstrap.addData(numberOfMatches);

            // count the number of occurrences for the pattern in the series
            numberOfMatches = match(stationaryBlock.newResample(), pattern);
            countInStationaryBlockBootstrap.addData(numberOfMatches);

            // count the number of occurrences for the pattern in the series
            numberOfMatches = match(circularBlock.newResample(), pattern);
            countInCircularBlockBootstrap.addData(numberOfMatches);
        }

        // compare the numbers of occurrences of the pattern using different bootstrap methods
        int countInSample = match(sample, pattern);
        System.out.println("matched patterns in sample: " + countInSample);
        System.out.println("matched patterns in simple bootstrap: " + countInSimpleBootstrap.value());
        System.out.println("matched patterns in stationary block bootstrap: " + countInStationaryBlockBootstrap.value());
        System.out.println("matched patterns in circular block bootstrap: " + countInCircularBlockBootstrap.value());
    }

    private static int match(double[] seq, double[] pattern) {
        int count = 0;
        for (int i = 0; i < seq.length - pattern.length; ++i) {
            if (seq[i] == pattern[0]) {
                double[] trunc = Arrays.copyOfRange(seq, i, i + pattern.length);
                if (DoubleUtils.equal(trunc, pattern, 1e-7)) {
                    count++;
                }
            }
        }
        return count;
    }

    private void case_resampling_1() {
        // sample from true population
        double[] sample = new double[]{150., 155., 160., 165., 170.};
        CaseResamplingReplacement boot = new CaseResamplingReplacement(sample);
        boot.seed(1234567890L);

        int B = 1000;
        double[] means = new double[B];
        for (int i = 0; i < B; ++i) {
            double[] resample = boot.newResample();
            means[i] = new Mean(resample).value();
        }

        // estimator of population mean
        double mean = new Mean(means).value();
        // variance of estimator; limited by sample size (regardless of how big B is)
        double var = new Variance(means).value();

        System.out.println(
                String.format("mean = %f, variance of the estimated mean = %f",
                        mean,
                        var));
    }

    private void case_resampling_2() {
        // sample from true population
        double[] sample = new double[]{150., 155., 160., 165., 170.};
        CaseResamplingReplacement boot = new CaseResamplingReplacement(sample);
        boot.seed(1234567890L);

        int B = 1000;
        BootstrapEstimator estimator
                = new BootstrapEstimator(boot, () -> new Mean(), B);

        System.out.println(
                String.format("mean = %f, variance of the estimated mean = %f",
                        estimator.value(),
                        estimator.variance()));
    }

    private void empirical_rng() {
        // we first generate some samples from standard normal distribution
        RandomLongGenerator uniform = new MersenneTwister();
        uniform.seed(1234567890L);

        RandomStandardNormalGenerator rng1 = new Zignor2005(uniform); // mean = 0, stdev = 1
        int N = 1000;
        double[] x1 = new double[N];
        for (int i = 0; i < N; ++i) {
            x1[i] = rng1.nextDouble();
        }

        // compute the empirical distribution function from the sample data
        EmpiricalDistribution dist2 = new EmpiricalDistribution(x1);

        // construct an RNG using inverse transform sampling method
        InverseTransformSampling rng2 = new InverseTransformSampling(dist2);

        // generate some random variates from the RNG
        double[] x2 = new double[N];
        for (int i = 0; i < N; ++i) {
            x2[i] = rng2.nextDouble();
        }

        // check the properties of the random variates
        Variance var = new Variance(x2);
        double mean = var.mean();
        double stdev = var.standardDeviation();
        System.out.println(String.format("mean = %f, standard deviation = %f", mean, stdev));

        // check if the samples are normally distributed
        ShapiroWilk test = new ShapiroWilk(x2);
        System.out.println(String.format("ShapiroWilk statistics = %f, pValue = %f", test.statistics(), test.pValue()));
    }

    private void multinomial_rvg() {
        MultinomialRVG rvg
                = new MultinomialRVG(100_000, new double[]{0.7, 0.3}); // bin0 is 70% chance, bin1 30% chance
        double[] bin = rvg.nextVector();

        double total = 0;
        for (int i = 0; i < bin.length; ++i) {
            total += bin[i];
        }

        double bin0 = bin[0] / total; // bin0 percentage
        double bin1 = bin[1] / total; // bin0 percentage

        System.out.println(String.format("bin0 %% = %f, bin1 %% = %f", bin0, bin1));
    }

    private void normal_rvg() {
        // mean
        Vector mu = new DenseVector(new double[]{-2., 2.});
        // covariance matrix
        Matrix sigma = new DenseMatrix(new double[][]{
            {1., 0.5},
            {0.5, 1.}
        });
        NormalRVG rvg = new NormalRVG(mu, sigma);
        rvg.seed(1234567890L);

        final int size = 10_000;
        double[][] x = new double[size][];

        Mean mean1 = new Mean();
        Mean mean2 = new Mean();
        for (int i = 0; i < size; ++i) {
            double[] v = rvg.nextVector();
            mean1.addData(v[0]);
            mean2.addData(v[1]);

            x[i] = v;
        }

        System.out.println(String.format("mean of X_1 = %f", mean1.value()));
        System.out.println(String.format("mean of X_2 = %f", mean2.value()));

        Matrix X = new DenseMatrix(x);
        SampleCovariance cov = new SampleCovariance(X);
        System.out.println(String.format("sample covariance = %s", cov.toString()));
    }

    private void compute_Pi() {
        final int N = 1_000_000;

        RandomVectorGenerator rvg
                = new UniformDistributionOverBox(
                        new RealInterval(-1., 1.), // a unit square box
                        new RealInterval(-1., 1.));

        int N0 = 0;
        for (int i = 0; i < N; i++) {
            double[] xy = rvg.nextVector();
            double x = xy[0], y = xy[1];
            if (x * x + y * y <= 1.) { // check if the dot is inside a circle
                N0++;
            }
        }
        double pi = 4. * N0 / N;
        System.out.println("pi = " + pi);
    }

    private void exponential_rng() {
        int size = 500_000;

        RandomExpGenerator rng = new Ziggurat2000Exp();
        rng.seed(634641070L);

        double[] x = new double[size];
        for (int i = 0; i < size; ++i) {
            x[i] = rng.nextDouble();
        }

        // compute the sample statistics
        Mean mean = new Mean(x);
        Variance var = new Variance(x);
        Skewness skew = new Skewness(x);
        Kurtosis kurtosis = new Kurtosis(x);

        // compute the theoretial statistics
        ProbabilityDistribution dist = new ExponentialDistribution();

        // compute the theoretial statistics
        printStats(dist, mean, var, skew, kurtosis);
    }

    private void Poisson_rng() {
        final int N = 10_000;
        double lambda = 1;

        RandomNumberGenerator rng = new Knuth1969(lambda);
        rng.seed(123456789L);

        double[] x = new double[N];
        for (int i = 0; i < N; ++i) {
            x[i] = rng.nextDouble();
        }

        // compute the sample statistics
        Mean mean = new Mean(x);
        Variance var = new Variance(x);
        Skewness skew = new Skewness(x);
        Kurtosis kurtosis = new Kurtosis(x);

        // compute the theoretial statistics
        PoissonDistribution dist = new PoissonDistribution(lambda);

        // compute the theoretial statistics
        printStats(dist, mean, var, skew, kurtosis);
    }

    private void gamma_rng() {
        final int size = 1_000_000;

        final double k = 0.1;
        final double theta = 1;

        KunduGupta2007 rng = new KunduGupta2007(k, theta, new UniformRNG());
        rng.seed(1234567895L);

        double[] x = new double[size];
        for (int i = 0; i < size; ++i) {
            x[i] = rng.nextDouble();
        }

        // compute the sample statistics
        Mean mean = new Mean(x);
        Variance var = new Variance(x);
        Skewness skew = new Skewness(x);
        Kurtosis kurtosis = new Kurtosis(x);

        // compute the theoretial statistics
        ProbabilityDistribution dist = new GammaDistribution(k, theta);

        // compute the theoretial statistics
        printStats(dist, mean, var, skew, kurtosis);
    }

    private void beta_rng() {
        final int size = 1_000_000;

        final double alpha = 0.1;
        final double beta = 0.2;

        RandomBetaGenerator rng = new Cheng1978(alpha, beta, new UniformRNG());
        rng.seed(1234567890L);

        double[] x = new double[size];
        for (int i = 0; i < size; ++i) {
            x[i] = rng.nextDouble();
        }

        // compute the sample statistics
        Mean mean = new Mean(x);
        Variance var = new Variance(x);
        Skewness skew = new Skewness(x);
        Kurtosis kurtosis = new Kurtosis(x);

        // compute the theoretial statistics
        ProbabilityDistribution dist = new BetaDistribution(alpha, beta);

        // compare sample vs theoretical statistics
        printStats(dist, mean, var, skew, kurtosis);
    }

    private void normal_rng() {
        RandomLongGenerator uniform = new MersenneTwister();
        uniform.seed(1234567890L);

        RandomStandardNormalGenerator rng1 = new Zignor2005(uniform); // mean = 0, stdev = 1
        int N = 1000;
        double[] arr1 = new double[N];
        for (int i = 0; i < N; ++i) {
            arr1[i] = rng1.nextDouble();
        }

        // check the statistics of the random samples
        Variance var1 = new Variance(arr1);
        System.out.println(
                String.format(
                        "mean = %f, stdev = %f",
                        var1.mean(),
                        var1.standardDeviation()));

        NormalRNG rng2 = new NormalRNG(1., 2., rng1); // mean = 1, stdev = 2
        double[] arr2 = new double[N];
        for (int i = 0; i < N; ++i) {
            arr2[i] = rng2.nextDouble();
        }

        // check the statistics of the random samples
        Variance var2 = new Variance(arr2);
        System.out.println(
                String.format(
                        "mean = %f, stdev = %f",
                        var2.mean(),
                        var2.standardDeviation()));
    }

    private void MT19937() {
        RandomLongGenerator rng = new MersenneTwister();

        long startTime = System.nanoTime();
        int N = 1_000_000;
        for (int i = 0; i < N; ++i) {
            rng.nextDouble();
        }
        long endTime = System.nanoTime();

        long duration = (endTime - startTime);
        double ms = (double) duration / 1_000_000.; // divide by 1000000 to get milliseconds
        System.out.println(String.format("took MT19937 %f milliseconds to generate %d random numbers", ms, N));
    }

    private void lcgs() {
        System.out.println("generate randome numbers using an Lehmer RNG:");
        RandomLongGenerator rng1 = new Lehmer();
        rng1.seed(1234567890L);
        generateIntAndPrint(rng1, 10);
        double[] arr = generate(rng1, 10);
        print(arr);

        System.out.println("generate randome numbers using an LEcuyer RNG:");
        RandomLongGenerator rng2 = new LEcuyer();
        rng2.seed(1234567890L);
        generateIntAndPrint(rng2, 10);
        arr = generate(rng2, 10);
        print(arr);

        System.out.println("generate randome numbers using a composite LCG:");
        RandomLongGenerator rng3
                = new CompositeLinearCongruentialGenerator(
                        new LinearCongruentialGenerator[]{
                            (LinearCongruentialGenerator) rng1,
                            (LinearCongruentialGenerator) rng2
                        }
                );
        rng3.seed(1234567890L);
        generateIntAndPrint(rng3, 10);
        arr = generate(rng3, 10);
        print(arr);
    }

    private static double[] generate(RandomNumberGenerator rng, int n) {
        double[] arr = new double[n];
        for (int i = 0; i < n; i++) {
            arr[i] = rng.nextDouble();
        }
        return arr;
    }

    private static void print(double[] arr) {
        System.out.println(Arrays.toString(arr));
    }

    private static void generateIntAndPrint(RandomNumberGenerator rng, int n) {
        double[] randomNumbers = new double[n];
        for (int i = 0; i < n; i++) {
            randomNumbers[i] = rng.nextDouble();
        }
        System.out.println(Arrays.toString(randomNumbers));
    }

    private void printStats(
            ProbabilityDistribution dist,
            Mean mean,
            Variance var,
            Skewness skew,
            Kurtosis kurtosis
    ) {
        System.out.println(
                String.format("theoretical mean = %f, sample mean = %f",
                        dist.mean(),
                        mean.value()));
        System.out.println(
                String.format("theoretical var = %f, sample var = %f",
                        dist.variance(),
                        var.value()));
        System.out.println(
                String.format("theoretical skew = %f, sample skew = %f",
                        dist.skew(),
                        skew.value()));
        System.out.println(
                String.format("theoretical kurtosis = %f, sample kurtosis = %f",
                        dist.kurtosis(),
                        kurtosis.value()));
    }
}
