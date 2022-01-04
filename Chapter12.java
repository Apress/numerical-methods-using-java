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

import java.util.List;
import java.util.ArrayList;
import java.util.Arrays;
import dev.nm.algebra.linear.matrix.doubles.Matrix;
import dev.nm.algebra.linear.matrix.doubles.matrixtype.dense.DenseMatrix;
import dev.nm.algebra.linear.matrix.doubles.matrixtype.dense.triangle.SymmetricMatrix;
import dev.nm.algebra.linear.matrix.doubles.operation.Inverse;
import dev.nm.algebra.linear.vector.doubles.Vector;
import dev.nm.algebra.linear.vector.doubles.dense.DenseVector;
import dev.nm.number.DoubleUtils;
import dev.nm.stat.covariance.*;
import dev.nm.stat.covariance.covarianceselection.CovarianceSelectionProblem;
import dev.nm.stat.covariance.covarianceselection.lasso.CovarianceSelectionGLASSOFAST;
import dev.nm.stat.covariance.covarianceselection.lasso.CovarianceSelectionLASSO;
import dev.nm.stat.covariance.nlshrink.LedoitWolf2016;
import dev.nm.stat.descriptive.correlation.CorrelationMatrix;
import dev.nm.stat.descriptive.correlation.SpearmanRankCorrelation;
import dev.nm.stat.descriptive.covariance.Covariance;
import dev.nm.stat.descriptive.covariance.SampleCovariance;
import dev.nm.stat.descriptive.moment.Kurtosis;
import dev.nm.stat.descriptive.moment.Mean;
import dev.nm.stat.descriptive.moment.Moments;
import dev.nm.stat.descriptive.moment.Skewness;
import dev.nm.stat.descriptive.moment.Variance;
import dev.nm.stat.descriptive.moment.weighted.WeightedMean;
import dev.nm.stat.descriptive.moment.weighted.WeightedVariance;
import dev.nm.stat.descriptive.rank.Max;
import dev.nm.stat.descriptive.rank.Min;
import dev.nm.stat.descriptive.rank.Quantile;
import dev.nm.stat.descriptive.rank.Rank;
import dev.nm.stat.distribution.discrete.ProbabilityMassQuantile;
import dev.nm.stat.distribution.univariate.WeibullDistribution;
import dev.nm.stat.distribution.discrete.ProbabilityMassFunction.Mass;
import dev.nm.stat.distribution.multivariate.DirichletDistribution;
import dev.nm.stat.distribution.multivariate.MultinomialDistribution;
import dev.nm.stat.distribution.multivariate.MultivariateNormalDistribution;
import dev.nm.stat.distribution.multivariate.MultivariateProbabilityDistribution;
import dev.nm.stat.distribution.multivariate.MultivariateTDistribution;
import dev.nm.stat.distribution.univariate.BetaDistribution;
import dev.nm.stat.distribution.univariate.BinomialDistribution;
import dev.nm.stat.distribution.univariate.ChiSquareDistribution;
import dev.nm.stat.distribution.univariate.EmpiricalDistribution;
import dev.nm.stat.distribution.univariate.ExponentialDistribution;
import dev.nm.stat.distribution.univariate.FDistribution;
import dev.nm.stat.distribution.univariate.GammaDistribution;
import dev.nm.stat.distribution.univariate.LogNormalDistribution;
import dev.nm.stat.distribution.univariate.NormalDistribution;
import dev.nm.stat.distribution.univariate.PoissonDistribution;
import dev.nm.stat.distribution.univariate.RayleighDistribution;
import dev.nm.stat.distribution.univariate.TDistribution;
import dev.nm.stat.factor.factoranalysis.FAEstimator;
import dev.nm.stat.factor.factoranalysis.FactorAnalysis;
import dev.nm.stat.factor.pca.PCA;
import dev.nm.stat.factor.pca.PCAbyEigen;
import dev.nm.stat.factor.pca.PCAbySVD;
import dev.nm.stat.hmm.ForwardBackwardProcedure;
import dev.nm.stat.hmm.HmmInnovation;
import dev.nm.stat.hmm.Viterbi;
import dev.nm.stat.hmm.discrete.BaumWelch;
import dev.nm.stat.hmm.discrete.DiscreteHMM;
import dev.nm.stat.hmm.mixture.MixtureHMM;
import dev.nm.stat.hmm.mixture.MixtureHMMEM;
import dev.nm.stat.hmm.mixture.distribution.NormalMixtureDistribution;
import dev.nm.stat.markovchain.SimpleMC;
import dev.nm.stat.random.rng.univariate.normal.StandardNormalRNG;
import dev.nm.stat.test.distribution.AndersonDarling;
import dev.nm.stat.test.distribution.CramerVonMises2Samples;
import dev.nm.stat.test.distribution.kolmogorov.KolmogorovSmirnov;
import dev.nm.stat.test.distribution.kolmogorov.KolmogorovSmirnov1Sample;
import dev.nm.stat.test.distribution.kolmogorov.KolmogorovSmirnov2Samples;
import dev.nm.stat.test.distribution.normality.DAgostino;
import dev.nm.stat.test.distribution.normality.JarqueBera;
import dev.nm.stat.test.distribution.normality.Lilliefors;
import dev.nm.stat.test.distribution.normality.ShapiroWilk;
import dev.nm.stat.test.distribution.pearson.ChiSquareIndependenceTest;
import dev.nm.stat.test.mean.OneWayANOVA;
import dev.nm.stat.test.mean.T;
import dev.nm.stat.test.rank.KruskalWallis;
import dev.nm.stat.test.rank.SiegelTukey;
import dev.nm.stat.test.rank.VanDerWaerden;
import dev.nm.stat.test.rank.wilcoxon.WilcoxonSignedRank;
import static java.lang.Math.abs;
import static java.lang.Math.sqrt;

/**
 * Numerical Methods Using Java: For Data Science, Analysis, and Engineering
 *
 * @author haksunli
 * @see
 * https://www.amazon.com/Numerical-Methods-Using-Java-Engineering/dp/1484267966
 * https://nm.dev/
 */
public class Chapter12 {

    public static void main(String[] args) throws Exception {
        System.out.println("Chapter 12 demos");

        Chapter12 chapter12 = new Chapter12();
        chapter12.sample_statistics();
        chapter12.rank();
        chapter12.quantile();
        chapter12.LedoitWolf2004();
        chapter12.LedoitWolf2016();
        chapter12.normal_distributions();
        chapter12.lognormal_distributions();
        chapter12.exponential_distribution();
        chapter12.Poisson_distribution();
        chapter12.binomial_distribution();
        chapter12.t_distribution();
        chapter12.F_distribution();
        chapter12.chi_square_distribution();
        chapter12.Rayleigh_distribution();
        chapter12.gamma_distribution();
        chapter12.beta_distribution();
        chapter12.Weibull_distribution();
        chapter12.empirical_distribution();
        chapter12.multivariate_normal_distribution();
        chapter12.multivariate_t_distribution();
        chapter12.Dirichlet_distribution();
        chapter12.multinomial_distribution();
        chapter12.hypothesis_testing();
        chapter12.Shapiro_Wilk_test();
        chapter12.Jarque_Bera_test();
        chapter12.DAgostino_test();
        chapter12.Lilliefors_test();
        chapter12.Kolmogorov_Smirnov_test();
        chapter12.Anderson_Darling_test();
        chapter12.Cramer_Von_Mises_test();
        chapter12.Chi_square_independence_test();
        chapter12.t_test();
        chapter12.one_way_ANOVA();
        chapter12.Kruskal_Wallis_test();
        chapter12.Wilcoxon_signed_rank_test();
        chapter12.Siegel_Tukey_test();
        chapter12.Van_Der_Waerden_test();
        chapter12.DTMC();
        chapter12.HMM1();
        chapter12.HMM2();
        chapter12.HMM3();
        chapter12.PCA_engen();
        chapter12.PCA_svd();
        chapter12.factor_analysis();
        chapter12.covariance_selection_LASSO();
    }

    public void covariance_selection_LASSO() {
        System.out.println("covariance selection using LASSO");

        // generate random samples from standard normal distribution
        StandardNormalRNG rnorm = new StandardNormalRNG();
        rnorm.seed(1234567890L);
        int nRows = 50;
        int nCols = 10;
        Matrix X = new DenseMatrix(nRows, nCols);
        for (int i = 1; i <= nRows; ++i) {
            for (int j = 1; j <= nCols; ++j) {
                X.set(i, j, rnorm.nextDouble());
            }
        }

        // sample covariance matrix
        Matrix S = new SampleCovariance(X);
        System.out.println("sample covariance:");
        System.out.println(S);
        Matrix S_inv = new Inverse(S);
        System.out.println("inverse sample covariance:");
        System.out.println(S_inv);

        // the penalty parameter
        double rho = 0.03;
        CovarianceSelectionProblem problem
                = new CovarianceSelectionProblem(S, rho);

        long time1 = System.currentTimeMillis();
        CovarianceSelectionLASSO lasso
                = new CovarianceSelectionLASSO(problem, 1e-5);
        Matrix sigma = lasso.covariance();
        time1 = System.currentTimeMillis() - time1;

        System.out.println("estimated sigma:");
        System.out.println(sigma);
        System.out.println("inverse sigma:");
        Matrix sigma_inv = lasso.inverseCovariance();
        System.out.println(sigma_inv);

        long time2 = System.currentTimeMillis();
        CovarianceSelectionGLASSOFAST lasso2 = new CovarianceSelectionGLASSOFAST(problem);
        Matrix sigma2 = lasso2.covariance();
        time2 = System.currentTimeMillis() - time2;

        System.out.println("CovarianceSelectionLASSO took " + time1 + " millisecs");
        System.out.println("CovarianceSelectionGLASSOFAST took " + time2 + " millisecs");
    }

    // data set from R
    private static final Matrix R_data
            = new DenseMatrix(new double[][]{
        {1., 1., 3., 3., 1., 1.},
        {1., 2., 3., 3., 1., 1.},
        {1., 1., 3., 4., 1., 1.},
        {1., 1., 3., 3., 1., 2.},
        {1., 1., 3., 3., 1., 1.},
        {1., 1., 1., 1., 3., 3.},
        {1., 2., 1., 1., 3., 3.},
        {1., 1., 1., 2., 3., 3.},
        {1., 2., 1., 1., 3., 4.},
        {1., 1., 1., 1., 3., 3.},
        {3., 3., 1., 1., 1., 1.},
        {3., 4., 1., 1., 1., 1.},
        {3., 3., 1., 2., 1., 1.},
        {3., 3., 1., 1., 1., 2.},
        {3., 3., 1., 1., 1., 1.},
        {4., 4., 5., 5., 6., 6.},
        {5., 6., 4., 6., 4., 5.},
        {6., 5., 6., 4., 5., 4.}
    });

    public void factor_analysis() {
        System.out.println("factor analysis");

        // number of hidden factors
        int nFactors = 3;
        FactorAnalysis factor_analysis
                = new FactorAnalysis(
                        R_data,
                        nFactors,
                        FactorAnalysis.ScoringRule.THOMSON // specify the scoring rule
                );

        System.out.println("number of observations = " + factor_analysis.nObs());
        System.out.println("number of variables = " + factor_analysis.nVariables());
        System.out.println("number of factors = " + factor_analysis.nFactors());

        // covariance matrix
        Matrix S = factor_analysis.S();
        System.out.println("covariance matrix:");
        System.out.println(S);

        FAEstimator estimators = factor_analysis.getEstimators(700);
        double fitted = estimators.logLikelihood();
        System.out.println("log-likelihood of the fitting = " + fitted);

        Vector uniqueness = estimators.psi();
        System.out.println("uniqueness = " + uniqueness);

        int dof = estimators.dof();
        System.out.println("degree of freedom = " + dof);

        // the factor loadings
        Matrix loadings = estimators.loadings();
        System.out.println("factor loadings:");
        System.out.println(loadings);

        double testStats = estimators.statistics();
        System.out.println("test statistics = " + testStats);

        double pValue = estimators.pValue();
        System.out.println("p-value = " + pValue);

        Matrix scores = estimators.scores();
        System.out.println("scores:");
        System.out.println(scores);
    }

    // this is the US arrest data from R
    private static final DenseMatrix USArrests
            = new DenseMatrix(new double[][]{
        {13.2, 236, 58, 21.2},
        {10.0, 263, 48, 44.5},
        {8.1, 294, 80, 31.0},
        {8.8, 190, 50, 19.5},
        {9.0, 276, 91, 40.6},
        {7.9, 204, 78, 38.7},
        {3.3, 110, 77, 11.1},
        {5.9, 238, 72, 15.8},
        {15.4, 335, 80, 31.9},
        {17.4, 211, 60, 25.8},
        {5.3, 46, 83, 20.2},
        {2.6, 120, 54, 14.2},
        {10.4, 249, 83, 24.0},
        {7.2, 113, 65, 21.0},
        {2.2, 56, 57, 11.3},
        {6.0, 115, 66, 18.0},
        {9.7, 109, 52, 16.3},
        {15.4, 249, 66, 22.2},
        {2.1, 83, 51, 7.8},
        {11.3, 300, 67, 27.8},
        {4.4, 149, 85, 16.3},
        {12.1, 255, 74, 35.1},
        {2.7, 72, 66, 14.9},
        {16.1, 259, 44, 17.1},
        {9.0, 178, 70, 28.2},
        {6.0, 109, 53, 16.4},
        {4.3, 102, 62, 16.5},
        {12.2, 252, 81, 46.0},
        {2.1, 57, 56, 9.5},
        {7.4, 159, 89, 18.8},
        {11.4, 285, 70, 32.1},
        {11.1, 254, 86, 26.1},
        {13.0, 337, 45, 16.1},
        {0.8, 45, 44, 7.3},
        {7.3, 120, 75, 21.4},
        {6.6, 151, 68, 20.0},
        {4.9, 159, 67, 29.3},
        {6.3, 106, 72, 14.9},
        {3.4, 174, 87, 8.3},
        {14.4, 279, 48, 22.5},
        {3.8, 86, 45, 12.8},
        {13.2, 188, 59, 26.9},
        {12.7, 201, 80, 25.5},
        {3.2, 120, 80, 22.9},
        {2.2, 48, 32, 11.2},
        {8.5, 156, 63, 20.7},
        {4.0, 145, 73, 26.2},
        {5.7, 81, 39, 9.3},
        {2.6, 53, 66, 10.8},
        {6.8, 161, 60, 15.6}
    });

    public void PCA_svd() {
        System.out.println("PCA by SVD");

        // run PCA on the data using SVD
        PCA pca = new PCAbySVD(
                USArrests // the data set in matrix form
        );

        // number of factors
        int p = pca.nFactors();
        // number of observations
        int n = pca.nObs();

        Vector mean = pca.mean();
        Vector scale = pca.scale();
        Vector sdev = pca.sdPrincipalComponents();
        Matrix loadings = pca.loadings();
        Vector proportion = pca.proportionVar();
        Vector cumprop = pca.cumulativeProportionVar();

        System.out.println("number of factors = " + p);
        System.out.println("number of observations = " + n);

        System.out.println("mean: " + mean);
        System.out.println("scale: " + scale);
        // The standard deviations differ by a factor of sqrt(50 / 49),
        // since we use divisor (nObs - 1) for the sample covariance matrix
        System.out.println("standard deviation: " + sdev);

        // The signs of the columns of the loading are arbitrary.
        System.out.println("loading: ");
        System.out.println(loadings);

        // the proportion of variance in each dimension
        System.out.println("proportion of variance: " + proportion);
        System.out.println("cumulative proportion of variance: " + cumprop);
    }

    public void PCA_engen() {
        System.out.println("PCA by eigen decomposition");

        // run PCA on the data using eigen decomposition
        PCA pca = new PCAbyEigen(
                USArrests, // the data set in matrix form
                false // use covariance matrix instead of correlation matrix
        );

        // number of factors
        int p = pca.nFactors();
        // number of observations
        int n = pca.nObs();

        Vector mean = pca.mean();
        Vector scale = pca.scale();
        Vector sdev = pca.sdPrincipalComponents();
        Matrix loadings = pca.loadings();
        Vector proportion = pca.proportionVar();
        Vector cumprop = pca.cumulativeProportionVar();
        Matrix scores = pca.scores();

        System.out.println("number of factors = " + p);
        System.out.println("number of observations = " + n);

        System.out.println("mean: " + mean);
        System.out.println("scale: " + scale);
        // The standard deviations differ by a factor of sqrt(50 / 49),
        // since we use divisor (nObs - 1) for the sample covariance matrix
        System.out.println("standard deviation: " + sdev);

        // The signs of the columns of the loading are arbitrary.
        System.out.println("loading: ");
        System.out.println(loadings);

        // the proportion of variance in each dimension
        System.out.println("proportion of variance: " + proportion);
        System.out.println("cumulative proportion of variance: " + cumprop);

//        System.out.println("score: ");
//        System.out.println(scores);
    }

    public void HMM3() {
        System.out.println("learning hidden Markov model with normal distribution");

        // the initial probabilities
        Vector PI0 = new DenseVector(new double[]{0., 0., 1.});
        // the transition probabilities
        Matrix A0 = new DenseMatrix(new double[][]{
            {0.4, 0.2, 0.4},
            {0.3, 0.2, 0.5},
            {0.25, 0.25, 0.5}
        });
        // the conditional normal distributions
        NormalMixtureDistribution.Lambda[] lambda0 = new NormalMixtureDistribution.Lambda[]{
            new NormalMixtureDistribution.Lambda(0., .5), // (mu, sigma)
            new NormalMixtureDistribution.Lambda(0., 1.), // medium volatility
            new NormalMixtureDistribution.Lambda(0., 2.5) // high volatility
        };
        // the original HMM: a model of daily stock returns in 3 regimes
        MixtureHMM model0 = new MixtureHMM(PI0, A0, new NormalMixtureDistribution(lambda0));
        model0.seed(1234567890L);

        // generate a sequence of observations from the HMM
        int T = 10000;
        HmmInnovation[] innovations = new HmmInnovation[T];
        double[] observations = new double[T];
        for (int t = 0; t < T; ++t) {
            innovations[t] = model0.next();
            observations[t] = innovations[t].observation();
        }
        System.out.println("observations: ");
        for (int t = 1; t <= 100; ++t) {
            System.out.print(observations[t] + ", ");
            if (t % 20 == 0) {
                System.out.println("");
            }
        }
        System.out.println("");

        // learn an HMM from the observations
        MixtureHMM model1
                = new MixtureHMMEM(observations, model0, 1e-5, 20); // using true parameters as initial estimates
        Matrix A1 = model1.A();
        NormalMixtureDistribution.Lambda[] lambda1
                = ((NormalMixtureDistribution) model1.getDistribution()).getParams();

        System.out.println("original transition probabilities");
        System.out.println(A0);
        System.out.println("learned transition probabilities");
        System.out.println(A1);

        for (int i = 0; i < lambda0.length; ++i) {
            System.out.println(String.format("compare mu: %f vs %f", lambda0[i].mu, lambda1[i].mu));
            System.out.println(String.format("compare sigma: %f vs %f", lambda0[i].sigma, lambda1[i].sigma));
        }
    }

    public void HMM2() {
        System.out.println("learning hidden Markov model");

        // generate a sequence of observations from a HMM
        // the initial probabilities for 2 states
        DenseVector PI = new DenseVector(
                new double[]{0.6, 0.4}
        );
        // the transition probabilities
        DenseMatrix A = new DenseMatrix(new double[][]{
            {0.7, 0.3},
            {0.4, 0.6}
        });
        // the observation probabilities; 3 possible outcomes for 2 states
        DenseMatrix B = new DenseMatrix(new double[][]{
            {0.5, 0.4, 0.1},
            {0.1, 0.3, 0.6}
        });
        // construct an HMM1
        DiscreteHMM model = new DiscreteHMM(PI, A, B);
        model.seed(1234507890L, 1234507891L);

        // generate the observations
        int T = 10000;
        HmmInnovation[] innovations = new HmmInnovation[T];
        int[] states = new int[T];
        int[] observations = new int[T];
        for (int t = 0; t < T; ++t) {
            innovations[t] = model.next();
            states[t] = innovations[t].state();
            observations[t] = (int) innovations[t].observation();
        }
        System.out.println("observations: ");
        for (int t = 1; t <= 100; ++t) {
            System.out.print(observations[t] + ", ");
            if (t % 20 == 0) {
                System.out.println("");
            }
        }
        System.out.println("");

        // learn the HMM from observations
        DenseVector PI_0 = new DenseVector(new double[]{0.5, 0.5}); // initial guesses
        DenseMatrix A_0 = new DenseMatrix(new double[][]{ // initial guesses
            {0.5, 0.5},
            {0.5, 0.5}
        });
        DenseMatrix B_0 = new DenseMatrix(new double[][]{ // initial guesses
            {0.60, 0.20, 0.20},
            {0.20, 0.20, 0.60}
        });
        DiscreteHMM model_0 = new DiscreteHMM(PI_0, A_0, B_0);  // initial guesses

        // training
        int nIterations = 40;
        for (int i = 1; i <= nIterations; ++i) {
            model_0 = BaumWelch.train(observations, model_0);
        }

        // training results
        System.out.println("estimated transition probabilities: ");
        System.out.println(model_0.A()); // should be close to A
        System.out.println("(observation) conditional probabilities: ");
        System.out.println(model_0.B());  // should be close to B
    }

    public void HMM1() {
        System.out.println("hidden Markov model");

        // the initial probabilities for 2 states
        DenseVector PI = new DenseVector(
                new double[]{0.6, 0.4}
        );
        // the transition probabilities
        DenseMatrix A = new DenseMatrix(new double[][]{
            {0.7, 0.3},
            {0.4, 0.6}
        });
        // the observation probabilities; 3 possible outcomes for 2 states
        DenseMatrix B = new DenseMatrix(new double[][]{
            {0.5, 0.4, 0.1},
            {0.1, 0.3, 0.6}
        });
        // construct an HMM1
        DiscreteHMM hmm = new DiscreteHMM(PI, A, B);

        // the realized observations
        double[] observations = new double[]{1, 2, 3};

        // run the forward-backward algorithm
        ForwardBackwardProcedure fb = new ForwardBackwardProcedure(hmm, observations);
        for (int t = 1; t <= observations.length; ++t) {
            System.out.println(String.format(
                    "the *scaled* forward probability, alpha, in each state at time %d: %s",
                    t,
                    fb.scaledAlpha(t)
            ));
        }

        // run the Viterbi algorithm to find the most likely sequence of hidden states
        Viterbi viterbi = new Viterbi(hmm);
        int[] viterbi_states = viterbi.getViterbiStates(observations);
        System.out.println("the Viterbi states: " + Arrays.toString(viterbi_states));
    }

    public void DTMC() {
        System.out.println("discrete time Markov chain");

        // the stochastic matrix of transition probabilities
        Matrix A = new DenseMatrix(new double[][]{
            {0.4, 0.2, 0.4},
            {0.3, 0.2, 0.5},
            {0.25, 0.25, 0.5}
        });
        // start in state 3
        Vector I = new DenseVector(0., 0., 1.);

        SimpleMC MC = new SimpleMC(I, A);
        Vector PI = SimpleMC.getStationaryProbabilities(A);
        System.out.println("the stationary distribution = " + PI);

        // simulate the next 9 steps
        System.out.println("time 0 = " + 3);
        for (int i = 1; i < 10; ++i) {
            int state = MC.nextState();
            System.out.println(String.format("time %d = %d", i, state));
        }
    }

    public void Van_Der_Waerden_test() {
        System.out.println("Van der Waerden test");

        double[][] samples = new double[3][];
        samples[0] = new double[]{8, 10, 9, 10, 9};
        samples[1] = new double[]{7, 8, 5, 8, 5};
        samples[2] = new double[]{4, 8, 7, 5, 7};

        VanDerWaerden test = new VanDerWaerden(samples);
        System.out.println("H0: " + test.getNullHypothesis());
        System.out.println("H1: " + test.getAlternativeHypothesis());
        System.out.println("test statistics = " + test.statistics());
        System.out.println("p-value = " + test.pValue());
        System.out.println("is null rejected at 5% = " + test.isNullRejected(0.05));
    }

    public void Siegel_Tukey_test() {
        System.out.println("Siegel Tukey test");

        double[] sample1 = new double[]{4, 16, 48, 51, 66, 98};
        double[] sample2 = new double[]{33, 62, 84, 85, 88, 93, 97};

        SiegelTukey test = new SiegelTukey(
                sample1,
                sample2,
                0, // the hypothetical mean difference
                true // use the exact Wilcoxon Rank Sum distribution rather than normal distribution
        );
        System.out.println("H0: " + test.getNullHypothesis());
        System.out.println("H1: " + test.getAlternativeHypothesis());
        System.out.println("test statistics = " + test.statistics());
        System.out.println("p-value = " + test.pValue());
        System.out.println("p-value, right sided = " + test.rightOneSidedPvalue());
        System.out.println("p-value, left sided = " + test.leftOneSidedPvalue());
        System.out.println("is null rejected at 5% = " + test.isNullRejected(0.05));
    }

    public void Wilcoxon_signed_rank_test() {
        System.out.println("Wilcoxon signed rank test");

        double[] sample1 = new double[]{1.3, 5.4, 7.6, 7.2, 3.5};
        double[] sample2 = new double[]{2.7, 5.2, 6.3, 4.4, 9.8};

        WilcoxonSignedRank test = new WilcoxonSignedRank(
                sample1, sample2,
                2, // the hypothetical median that the distribution is symmetric about
                true // use the exact Wilcoxon rank sum distribution rather than normal distribution
        );
        System.out.println("H0: " + test.getNullHypothesis());
        System.out.println("H1: " + test.getAlternativeHypothesis());
        System.out.println("test statistics = " + test.statistics());
        System.out.println("p-value = " + test.pValue());
        System.out.println("p-value, right sided = " + test.rightOneSidedPvalue());
        System.out.println("p-value, left sided = " + test.leftOneSidedPvalue());
        System.out.println("is null rejected at 5% = " + test.isNullRejected(0.05));
    }

    public void Kruskal_Wallis_test() {
        System.out.println("Kruskal Wallis test");

        double[][] samples = new double[4][];
        samples[0] = new double[]{1, 1, 7.6, 7.2, 3.5};
        samples[1] = new double[]{2, 2, 6.3, 4.4, 9.8, 10.24};
        samples[2] = new double[]{-9, -9, -4.33, -5.4};
        samples[3] = new double[]{0.21, 0.21, 0.21, 0.86, 0.902, 0.663};

        KruskalWallis test = new KruskalWallis(samples);
        System.out.println("H0: " + test.getNullHypothesis());
        System.out.println("H1: " + test.getAlternativeHypothesis());
        System.out.println("test statistics = " + test.statistics());
        System.out.println("p-value = " + test.pValue());
        System.out.println("is null rejected at 5% = " + test.isNullRejected(0.05));
    }

    public void one_way_ANOVA() {
        System.out.println("One-way ANOVA");

        double[][] samples = new double[4][];
        samples[0] = new double[]{1.3, 5.4, 7.6, 7.2, 3.5};
        samples[1] = new double[]{2.7, 5.21, 6.3, 4.4, 9.8, 10.24};
        samples[2] = new double[]{-2.3, -5.3, -4.33, -5.4};
        samples[3] = new double[]{0.21, 0.34, 0.27, 0.86, 0.902, 0.663};

        OneWayANOVA test = new OneWayANOVA(samples);
        System.out.println("H0: " + test.getNullHypothesis());
        System.out.println("H1: " + test.getAlternativeHypothesis());
        System.out.println("test statistics = " + test.statistics());
        System.out.println("p-value = " + test.pValue());
        System.out.println("is null rejected at 5% = " + test.isNullRejected(0.05));
    }

    public void t_test() {
        System.out.println("t test");

        // the t-test
        T test1 = new T(
                new double[]{1, 3, 5, 2, 3, 5},
                new double[]{2, 5, 6, 4, 9, 8},
                true, // assume variances are equal
                4 // the hypothetical mean-difference = 4 in the null hypothesis
        );
        System.out.println("H0: " + test1.getNullHypothesis());
        System.out.println("H1: " + test1.getAlternativeHypothesis());
        System.out.println("test statistics = " + test1.statistics());
        System.out.println("1st mean = " + test1.mean1());
        System.out.println("2nd mean = " + test1.mean2());
        System.out.println("p-value = " + test1.pValue());
        System.out.println("p-value, right sided = " + test1.rightOneSidedPvalue());
        System.out.println("p-value, left sided = " + test1.leftOneSidedPvalue());
        System.out.println(String.format("95%% confidence interval = (%f, %f)", test1.leftConfidenceInterval(0.95), test1.rightConfidenceInterval(0.95)));
        System.out.println("97.5%% confidence interval = " + Arrays.toString(test1.confidenceInterval(0.975)));
        System.out.println("is null rejected at 5% = " + test1.isNullRejected(0.05));

        // Welch's t-test
        T test2 = new T(
                new double[]{1, 3, 5, 2, 3, 5},
                new double[]{2, 5, 6, 4, 9, 8},
                false, // assume variances are different
                4 // the hypothetical mean-difference = 4 in the null hypothesis
        );
        System.out.println("test statistics = " + test2.statistics());
        System.out.println("p-value = " + test2.pValue());
        System.out.println("p-value, right sided = " + test2.rightOneSidedPvalue());
        System.out.println("p-value, left sided = " + test2.leftOneSidedPvalue());
        System.out.println(String.format("95%% confidence interval = (%f, %f)", test2.leftConfidenceInterval(0.95), test2.rightConfidenceInterval(0.95)));
        System.out.println("97.5%% confidence interval = " + Arrays.toString(test2.confidenceInterval(0.975)));
        System.out.println("is null rejected at 5% = " + test2.isNullRejected(0.05));
    }

    public void Chi_square_independence_test() {
        System.out.println("Chi-square independence test");

        // the attendance/absence vs. pass/fail counts
        Matrix counts = new DenseMatrix(new double[][]{
            {25, 6},
            {8, 15}
        });
        ChiSquareIndependenceTest test1
                = new ChiSquareIndependenceTest(
                        counts,
                        0,
                        // the asymptotic distribution is the Chi-square distribution
                        ChiSquareIndependenceTest.Type.ASYMPTOTIC
                );

        Matrix expected = ChiSquareIndependenceTest.getExpectedContingencyTable(
                new int[]{31, 23}, // row sums
                new int[]{33, 21} // column sums
        );
        System.out.println("the expected frequencies:");
        System.out.println(expected);

        System.out.println("H0: " + test1.getNullHypothesis());
        System.out.println("H1: " + test1.getAlternativeHypothesis());
        System.out.println("test statistics = " + test1.statistics());
        System.out.println("p-value = " + test1.pValue());
        System.out.println("is null rejected at 5% = " + test1.isNullRejected(0.05));

        ChiSquareIndependenceTest test2
                = new ChiSquareIndependenceTest(
                        counts,
                        100000,// number of simulation to compute the Fisher exact distribution
                        ChiSquareIndependenceTest.Type.EXACT // use the Fisher exact distribution
                );
        System.out.println("p-value = " + test2.pValue());
        System.out.println("is null rejected at 5% = " + test2.isNullRejected(0.05));
    }

    public void Cramer_Von_Mises_test() {
        System.out.println("Cramer Von Mises test");

        // the samples
        double[] x1 = new double[]{-0.54289848, 0.08999578, -1.77719573, -0.67991860, -0.65741590, -0.25776164, 1.02024626, 1.26434300, 0.51068476, -0.23998229};
        double[] x2 = new double[]{1.7053818, 1.0260726, 1.7695157, 1.5650577, 1.4945107, 1.8593791, 2.1760302, -0.9728721, 1.4208313, 1.5892663};
        CramerVonMises2Samples test = new CramerVonMises2Samples(x1, x2);
        System.out.println("H0: " + test.getNullHypothesis());
        System.out.println("H1: " + test.getAlternativeHypothesis());
        System.out.println("test statistics = " + test.statistics());
        System.out.println("p-value = " + test.pValue());
        System.out.println("is null rejected at 5% = " + test.isNullRejected(0.05));
    }

    public void Anderson_Darling_test() {
        System.out.println("Anderson Darling test");

        // the samples
        double[] x1 = new double[]{38.7, 41.5, 43.8, 44.5, 45.5, 46.0, 47.7, 58.0};
        double[] x2 = new double[]{39.2, 39.3, 39.7, 41.4, 41.8, 42.9, 43.3, 45.8};
        double[] x3 = new double[]{34.0, 35.0, 39.0, 40.0, 43.0, 43.0, 44.0, 45.0};
        double[] x4 = new double[]{34.0, 34.8, 34.8, 35.4, 37.2, 37.8, 41.2, 42.8};

        AndersonDarling test = new AndersonDarling(x1, x2, x3, x4);
        System.out.println("H0: " + test.getNullHypothesis());
        System.out.println("H1: " + test.getAlternativeHypothesis());
        System.out.println("test statistics = " + test.statistics());
        System.out.println("p-value = " + test.pValue());
        System.out.println("alternative test statistics = " + test.statisticsAlternative());
        System.out.println("alternative p-value = " + test.pValueAlternative());
        System.out.println("is null rejected at 5% = " + test.isNullRejected(0.05));
    }

    public void Kolmogorov_Smirnov_test() {
        System.out.println("Kolmogorov Smirnov test");

        // one-sample KS test
        KolmogorovSmirnov1Sample test1 = new KolmogorovSmirnov1Sample(
                new double[]{ // with duplicates
                    1.2142038235675114, 0.8271665834857130, -2.2786245743283295, 0.8414895245471727,
                    -1.4327682855296735, -0.2501807766164897, -1.9512765152306415, 0.6963626117638846,
                    0.4741320101265005, 1.2142038235675114
                },
                new NormalDistribution(),
                KolmogorovSmirnov.Side.TWO_SIDED // options are: TWO_SIDED, GREATER, LESS
        );
        System.out.println("H0: " + test1.getNullHypothesis());
        System.out.println("test statistics = " + test1.statistics());
        System.out.println("p-value = " + test1.pValue());
        System.out.println("is null rejected at 5% = " + test1.isNullRejected(0.05));

        // two-sample KS test
        KolmogorovSmirnov2Samples test2 = new KolmogorovSmirnov2Samples(
                new double[]{ // x = rnorm(10)
                    1.2142038235675114, 0.8271665834857130, -2.2786245743283295, 0.8414895245471727,
                    -1.4327682855296735, -0.2501807766164897, -1.9512765152306415, 0.6963626117638846,
                    0.4741320101265005, -1.2340784297133520
                },
                new double[]{ // x = rnorm(15)
                    1.7996197748754565, -1.1371109188816089, 0.8179707525071304, 0.3809791236763478,
                    0.1644848304811257, 0.3397412780581336, -2.2571685407244795, 0.4137315314876659,
                    0.7318687611171864, 0.9905218801425318, -0.4748590846019594, 0.8882674167954235,
                    1.0534065683777052, 0.2553123235884622, -2.3172807717538038},
                KolmogorovSmirnov.Side.GREATER // options are: TWO_SIDED, GREATER, LESS
        );
        System.out.println("H0: " + test2.getNullHypothesis());
        System.out.println("test statistics = " + test2.statistics());
        System.out.println("p-value = " + test2.pValue());
        System.out.println("is null rejected at 5% = " + test2.isNullRejected(0.05));
    }

    public void Lilliefors_test() {
        System.out.println("Lilliefors test");

        double[] sample = new double[]{-1.7, -1, -1, -.73, -.61, -.5, -.24, .45, .62, .81, 1, 5};
        Lilliefors test = new Lilliefors(sample);
        System.out.println("H0: " + test.getNullHypothesis());
        System.out.println("H1: " + test.getAlternativeHypothesis());
        System.out.println("test statistics = " + test.statistics());
        System.out.println("p-value = " + test.pValue());
        System.out.println("is null rejected at 5% = " + test.isNullRejected(0.05));
    }

    public void DAgostino_test() {
        System.out.println("D'Agostino's test");

        double[] samples = new double[]{
            39, 35, 33, 33, 32, 30, 30, 30, 28, 28,
            27, 27, 27, 27, 27, 26, 26, 26, 26, 26,
            26, 25, 25, 25, 25, 25, 25, 24, 24, 24,
            24, 24, 23, 23, 23, 23, 23, 23, 23, 23,
            23, 23, 23, 23, 23, 22, 22, 22, 22, 21,
            21, 21, 21, 21, 21, 21, 20, 20, 19, 19,
            18, 16
        };
        DAgostino test = new DAgostino(samples);
        System.out.println("H0: " + test.getNullHypothesis());
        System.out.println("H1: " + test.getAlternativeHypothesis());
        System.out.println("skewness test statistics " + test.Z1());
        System.out.println("p-value for skewness test = " + test.pvalueZ1());
        System.out.println("kurtosis test statistics " + test.Z2());
        System.out.println("test statistics = " + test.statistics());
        System.out.println("p-value = " + test.pValue());
        System.out.println("is null rejected at 5% = " + test.isNullRejected(0.05));
    }

    public void Jarque_Bera_test() {
        System.out.println("Jarque-Bera test");

        double[] samples = new double[]{
            39, 35, 33, 33, 32, 30, 30, 30, 28, 28,
            27, 27, 27, 27, 27, 26, 26, 26, 26, 26,
            26, 25, 25, 25, 25, 25, 25, 24, 24, 24,
            24, 24, 23, 23, 23, 23, 23, 23, 23, 23,
            23, 23, 23, 23, 23, 22, 22, 22, 22, 21,
            21, 21, 21, 21, 21, 21, 20, 20, 19, 19,
            18, 16
        };
        JarqueBera test = new JarqueBera(
                samples,
                false // not using the exact Jarque-Bera distribution
        );
        System.out.println("H0: " + test.getNullHypothesis());
        System.out.println("H1: " + test.getAlternativeHypothesis());
        System.out.println("test statistics = " + test.statistics());
        System.out.println("p-value = " + test.pValue());
        System.out.println("is null rejected at 5% = " + test.isNullRejected(0.05));
    }

    public void Shapiro_Wilk_test() {
        System.out.println("Shapiro-Wilk test");

        double[] sample = new double[]{-1.7, -1, -1, -.73, -.61, -.5, -.24, .45, .62, .81, 1, 5};
        ShapiroWilk test = new ShapiroWilk(sample);
        System.out.println("H0: " + test.getNullHypothesis());
        System.out.println("H1: " + test.getAlternativeHypothesis());
        System.out.println("test statistics = " + test.statistics());
        System.out.println("p-value = " + test.pValue());
        System.out.println("is null rejected at 5% = " + test.isNullRejected(0.05));
    }

    public void hypothesis_testing() {
        System.out.println("hypothesis testing");

        int n = 100;
        BinomialDistribution dist2 = new BinomialDistribution(
                n,
                0.5 // p
        );
        double stdev = sqrt(dist2.variance()) / n;
        System.out.println("standard deviation = " + stdev);

        double z_score = (0.37 - 0.5) / stdev;
        System.out.println("z-score = " + z_score);
        double p_value = new NormalDistribution() // default ctor for standard normal distribution
                .cdf(z_score);
        System.out.println("p-value = " + p_value);
    }

    public void multinomial_distribution() {
        System.out.println("multinomial distribution");

        // k = 3, each of the 3 probabilities of success
        double[] prob = new double[]{0.1, 0.2, 0.7};
        int n = 100;
        MultinomialDistribution dist
                = new MultinomialDistribution(n, prob);

        // an outcome of the n trials
        Vector x = new DenseVector(new double[]{10, 20, 70});
        System.out.println(String.format("f(%s) = %f", x, dist.density(x)));
    }

    public void Dirichlet_distribution() {
        System.out.println("Dirichlet distribution");

        // the parameters
        double[] a = new double[]{1, 2, 3, 4, 5};
        DirichletDistribution dist = new DirichletDistribution(a);
        Vector x = new DenseVector(0.1, 0.2, 0.3, 0.2, 0.2);
        System.out.println(String.format("f(%s) = %f", x, dist.density(x)));
    }

    public void multivariate_t_distribution() {
        System.out.println("multivariate t distribution");

        int p = 2; // dimension
        Vector mu = new DenseVector(1., 2.); // mean
        Matrix Sigma = new DenseMatrix(p, p).ONE(); // scale matrix 

        int v = 1; // degree of freedom
        MultivariateTDistribution t
                = new MultivariateTDistribution(v, mu, Sigma);
        Vector x = new DenseVector(1.23, 4.56);
        System.out.println(String.format("f(%s) = %f", x, t.density(x)));

        v = 2;
        t = new MultivariateTDistribution(v, mu, Sigma);
        x = new DenseVector(1.23, 4.56);
        System.out.println(String.format("f(%s) = %f", x, t.density(x)));

        v = 3;
        t = new MultivariateTDistribution(v, mu, Sigma);
        x = new DenseVector(1.23, 4.56);
        System.out.println(String.format("f(%s) = %f", x, t.density(x)));

        v = 4;
        t = new MultivariateTDistribution(v, mu, Sigma);
        x = new DenseVector(1.23, 4.56);
        System.out.println(String.format("f(%s) = %f", x, t.density(x)));

        v = 5;
        t = new MultivariateTDistribution(v, mu, Sigma);
        x = new DenseVector(1.23, 4.56);
        System.out.println(String.format("f(%s) = %f", x, t.density(x)));

        v = 6;
        t = new MultivariateTDistribution(v, mu, Sigma);
        x = new DenseVector(1.23, 4.56);
        System.out.println(String.format("f(%s) = %f", x, t.density(x)));
    }

    public void multivariate_normal_distribution() {
        System.out.println("multivariate normal distribution");

        System.out.println("construct a 3-dimensional multivariate standard normal distribution");
        MultivariateProbabilityDistribution mvnorm1
                = new MultivariateNormalDistribution(
                        3 // dimension
                );
        System.out.println("mean = " + mvnorm1.mean());
        System.out.println("variance = " + mvnorm1.covariance());

        Vector x1 = new DenseVector(0.0, 0.0, 0.0);
        System.out.println(String.format("f(%s) = %f", x1, mvnorm1.density(x1)));
        x1 = new DenseVector(1.0, 0.0, 0.0);
        System.out.println(String.format("f(%s) = %f", x1, mvnorm1.density(x1)));
        x1 = new DenseVector(0.0, 0.5, 0.0);
        System.out.println(String.format("f(%s) = %f", x1, mvnorm1.density(x1)));
        x1 = new DenseVector(0.0, 0.0, 0.3);
        System.out.println(String.format("f(%s) = %f", x1, mvnorm1.density(x1)));
        x1 = new DenseVector(1.0, 0.5, 0.0);
        System.out.println(String.format("f(%s) = %f", x1, mvnorm1.density(x1)));
        x1 = new DenseVector(0.0, 0.5, 0.3);
        System.out.println(String.format("f(%s) = %f", x1, mvnorm1.density(x1)));
        x1 = new DenseVector(1.0, 0.0, 0.3);
        System.out.println(String.format("f(%s) = %f", x1, mvnorm1.density(x1)));
        x1 = new DenseVector(1.0, 0.5, 0.3);
        System.out.println(String.format("f(%s) = %f", x1, mvnorm1.density(x1)));

        System.out.println("construct a 2-dimensional multivariate normal distribution");
        DenseVector mu = new DenseVector(1.0, 2.0);
        DenseMatrix sigma = new DenseMatrix(new double[][]{{4.0, 2.0}, {2.0, 3.0}});
        MultivariateProbabilityDistribution mvnorm2
                = new MultivariateNormalDistribution(mu, sigma);
        System.out.println("mean = " + mvnorm2.mean()); // same as mu
        System.out.println("variance = " + mvnorm2.covariance()); // same as sigma
        Vector x2 = new DenseVector(0.3, 0.4);
        System.out.println(String.format("f(%s) = %f", x2, mvnorm2.density(x2)));
        x2 = new DenseVector(0.4, 0.3);
        System.out.println(String.format("f(%s) = %f", x2, mvnorm2.density(x2)));
    }

    public void empirical_distribution() {
        System.out.println("empirical distribution");

        // the data set
        double[] X = new double[]{0, 1, 2, 3, 4, 5, 6, 7, 8, 99};
        // construct the empirical distribution from the data set
        EmpiricalDistribution dist = new EmpiricalDistribution(X);

        System.out.println("mean = " + dist.mean());
        System.out.println("variance = " + dist.variance());
        System.out.println("skew = " + dist.skew());
        System.out.println("kurtosis = " + dist.kurtosis());

        double[] xs = new double[]{
            0, 0.0001, 0.001, 0.01, 0.1, 1, 2, 3};
        for (double x : xs) {
            System.out.println(String.format("F(%f) = %f", x, dist.cdf(x)));
        }
        for (double x : xs) {
            System.out.println(String.format("f(%f) = %f", x, dist.density(x)));
        }

        boolean allEquals = true;
        for (double u = 0.1; u < 1d; u += 0.1) {
            double x = dist.quantile(u);
            double y = dist.cdf(x);
            if (abs(u - y) > 1e-9) {
                allEquals = false;
            }
        }
        System.out.println("F(Q(u)) = u for all u = " + allEquals);
    }

    public void Weibull_distribution() {
        System.out.println("Weibull distribution");

        WeibullDistribution dist = new WeibullDistribution(
                1, // lambda, the scale parameter = 1
                1 // the shape parameter = 1
        );

        System.out.println("mean = " + dist.mean());
        System.out.println("variance = " + dist.variance());
        System.out.println("skew = " + dist.skew());
        System.out.println("kurtosis = " + dist.kurtosis());
        System.out.println("entropy = " + dist.entropy());

        double[] xs = new double[]{
            0, 0.0001, 0.001, 0.01, 0.1, 1, 2, 3};
        for (double x : xs) {
            System.out.println(String.format("F(%f) = %f", x, dist.cdf(x)));
        }
        for (double x : xs) {
            System.out.println(String.format("f(%f) = %f", x, dist.density(x)));
        }

        boolean allEquals = true;
        for (double u = 0.001; u < 1d; u += 0.001) {
            double x = dist.quantile(u);
            double y = dist.cdf(x);
            if (abs(u - y) > 1e-9) {
                allEquals = false;
            }
        }
        System.out.println("F(Q(u)) = u for all u = " + allEquals);
    }

    public void beta_distribution() {
        System.out.println("Beta distribution");

        BetaDistribution dist = new BetaDistribution(
                0.5, // alpha = 0.5
                1.5 // beta = 1.5
        );

        System.out.println("mean = " + dist.mean());
        System.out.println("variance = " + dist.variance());
        System.out.println("skew = " + dist.skew());
        System.out.println("kurtosis = " + dist.kurtosis());

        double[] xs = new double[]{
            0, 0.0001, 0.001, 0.01, 0.1, 1, 2, 3};
        for (double x : xs) {
            System.out.println(String.format("F(%f) = %f", x, dist.cdf(x)));
        }
        for (double x : xs) {
            System.out.println(String.format("f(%f) = %f", x, dist.density(x)));
        }

        boolean allEquals = true;
        for (double u = 0.001; u < 1d; u += 0.001) {
            double x = dist.quantile(u);
            double y = dist.cdf(x);
            if (abs(u - y) > 1e-9) {
                allEquals = false;
            }
        }
        System.out.println("F(Q(u)) = u for all u = " + allEquals);
    }

    public void gamma_distribution() {
        System.out.println("Gamma distribution");

        GammaDistribution dist = new GammaDistribution(
                1., // k = 1
                2. // theta = 2
        );

        System.out.println("mean = " + dist.mean());
        System.out.println("variance = " + dist.variance());
        System.out.println("skew = " + dist.skew());
        System.out.println("kurtosis = " + dist.kurtosis());

        double[] xs = new double[]{
            0, 0.0001, 0.001, 0.01, 0.1, 1, 2, 3, 10, 100, 1000};
        for (double x : xs) {
            System.out.println(String.format("F(%f) = %f", x, dist.cdf(x)));
        }
        for (double x : xs) {
            System.out.println(String.format("f(%f) = %f", x, dist.density(x)));
        }

        boolean allEquals = true;
        for (double u = 0.001; u < 1d; u += 0.001) {
            double x = dist.quantile(u);
            double y = dist.cdf(x);
            if (abs(u - y) > 1e-9) {
                allEquals = false;
            }
        }
        System.out.println("F(Q(u)) = u for all u = " + allEquals);
    }

    public void Rayleigh_distribution() {
        System.out.println("Rayleigh distribution");

        RayleighDistribution dist = new RayleighDistribution(
                2 // sigma = 2
        );

        System.out.println("mean = " + dist.mean());
        System.out.println("median = " + dist.median());
        System.out.println("variance = " + dist.variance());
        System.out.println("skew = " + dist.skew());
        System.out.println("kurtosis = " + dist.kurtosis());
        System.out.println("entropy = " + dist.entropy());

        double[] xs = new double[]{
            0, 0.0001, 0.001, 0.01, 0.1, 1, 2, 3, 10, 100, 1000};
        for (double x : xs) {
            System.out.println(String.format("F(%f) = %f", x, dist.cdf(x)));
        }
        for (double x : xs) {
            System.out.println(String.format("f(%f) = %f", x, dist.density(x)));
        }

        boolean allEquals = true;
        for (double u = 0.000001; u < 1d; u += 0.000001) {
            double x = dist.quantile(u);
            double y = dist.cdf(x);
            if (abs(u - y) > 1e-9) {
                allEquals = false;
            }
        }
        System.out.println("F(Q(u)) = u for all u = " + allEquals);
    }

    public void chi_square_distribution() {
        System.out.println("chi-square distribution");

        ChiSquareDistribution dist = new ChiSquareDistribution(
                1.5 // degree of freedom
        );

        System.out.println("mean = " + dist.mean());
        System.out.println("median = " + dist.median());
        System.out.println("variance = " + dist.variance());
        System.out.println("skew = " + dist.skew());
        System.out.println("kurtosis = " + dist.kurtosis());
        System.out.println("entropy = " + dist.entropy());

        double[] xs = new double[]{
            0, 0.0001, 0.001, 0.01, 0.1, 1, 2, 3, 10, 100, 1000};
        for (double x : xs) {
            System.out.println(String.format("F(%f) = %f", x, dist.cdf(x)));
        }
        for (double x : xs) {
            System.out.println(String.format("f(%f) = %f", x, dist.density(x)));
        }

        boolean allEquals = true;
        for (double u = 0.1; u < 1d; u += 0.1) {
            double x = dist.quantile(u);
            double y = dist.cdf(x);
            if (abs(u - y) > 1e-9) {
                allEquals = false;
            }
        }
        System.out.println("F(Q(u)) = u for all u = " + allEquals);
    }

    public void F_distribution() {
        System.out.println("F distribution");

        FDistribution dist = new FDistribution(100.5, 10); // degrees of freedom = {100.5, 10}

        System.out.println("mean = " + dist.mean());
        System.out.println("variance = " + dist.variance());
        System.out.println("skew = " + dist.skew());
        System.out.println("kurtosis = " + dist.kurtosis());

        double[] ks = DoubleUtils.seq(0., 20., 1.); // a sequence of numbers from 0 to 20
        for (double k : ks) {
            System.out.println(String.format("F(%f) = %f", k, dist.cdf(k)));
        }
        for (double k : ks) {
            System.out.println(String.format("f(%f) = %f", k, dist.density(k)));
        }

        boolean allEquals = true;
        for (double u = 0.1; u < 1d; u += 0.1) {
            double x = dist.quantile(u);
            double y = dist.cdf(x);
            if (abs(u - y) > 1e-9) {
                allEquals = false;
            }
        }
        System.out.println("F(Q(u)) = u for all u = " + allEquals);
    }

    public void t_distribution() {
        System.out.println("t distribution");

        TDistribution dist = new TDistribution(4.5); // degree of freedom = 4.5

        System.out.println("mean = " + dist.mean());
        System.out.println("median = " + dist.median());
        System.out.println("variance = " + dist.variance());
        System.out.println("skew = " + dist.skew());
        System.out.println("kurtosis = " + dist.kurtosis());
        System.out.println("entropy = " + dist.entropy());

        double[] ks = DoubleUtils.seq(0., 20., 1.); // a sequence of numbers from 0 to 20
        for (double k : ks) {
            System.out.println(String.format("F(%f) = %f", k, dist.cdf(k)));
        }
        for (double k : ks) {
            System.out.println(String.format("f(%f) = %f", k, dist.density(k)));
        }

        boolean allEquals = true;
        for (double u = 0.1; u < 1d; u += 0.1) {
            double x = dist.quantile(u);
            double y = dist.cdf(x);
            if (abs(u - y) > 1e-9) {
                allEquals = false;
            }
        }
        System.out.println("F(Q(u)) = u for all u = " + allEquals);
    }

    public void binomial_distribution() {
        System.out.println("binomial distribution");

        BinomialDistribution dist1 = new BinomialDistribution(
                20, // n
                0.7 // p
        );

        System.out.println("mean = " + dist1.mean());
        System.out.println("median = " + dist1.median());
        System.out.println("variance = " + dist1.variance());
        System.out.println("skew = " + dist1.skew());
        System.out.println("kurtosis = " + dist1.kurtosis());
        System.out.println("entropy = " + dist1.entropy());

        double[] ks = DoubleUtils.seq(0., 20., 1.); // a sequence of numbers from 0 to 20
        for (double k : ks) {
            System.out.println(String.format("F(%f) = %f", k, dist1.cdf(k)));
        }
        for (double k : ks) {
            System.out.println(String.format("f(%f) = %f", k, dist1.density(k)));
        }

        for (double u = 0.1; u < 1d; u += 0.1) {
            double x = dist1.quantile(u);
            System.out.println(String.format("Q(%f) = %f", u, x));
        }
    }

    public void Poisson_distribution() {
        System.out.println("Poisson distribution");

        PoissonDistribution dist = new PoissonDistribution(
                2. // lambda = 2.
        );

        System.out.println("mean = " + dist.mean());
        System.out.println("median = " + dist.median());
        System.out.println("variance = " + dist.variance());
        System.out.println("skew = " + dist.skew());
        System.out.println("kurtosis = " + dist.kurtosis());
        System.out.println("entropy = " + dist.entropy());

        double[] ks = new double[]{
            0, 1, 2, 3, 4, 5};
        for (double k : ks) {
            System.out.println(String.format("F(%f) = %f", k, dist.cdf(k)));
        }
        for (double k : ks) {
            System.out.println(String.format("f(%f) = %f", k, dist.density(k)));
        }

        for (double u = 0.1; u < 1d; u += 0.1) {
            double x = dist.quantile(u);
            System.out.println(String.format("Q(%f) = %f", u, x));
        }
    }

    public void exponential_distribution() {
        System.out.println("exponential distribution");

        ExponentialDistribution dist = new ExponentialDistribution(
                1.0 // lambda = 1.
        );

        System.out.println("mean = " + dist.mean());
        System.out.println("median = " + dist.median());
        System.out.println("variance = " + dist.variance());
        System.out.println("skew = " + dist.skew());
        System.out.println("kurtosis = " + dist.kurtosis());
        System.out.println("entropy = " + dist.entropy());

        double[] xs = new double[]{
            -1000000, -10000, -1000, -100, -10, -1, -0.1, -0.01, -0.001, -0.0001,
            0, 0.0001, 0.001, 0.01, 0.1, 1, 2, 3, 10, 100, 1000};
        for (double x : xs) {
            System.out.println(String.format("F(%f) = %f", x, dist.cdf(x)));
        }
        for (double x : xs) {
            System.out.println(String.format("f(%f) = %f", x, dist.density(x)));
        }

        boolean allEquals = true;
        for (double u = 0.000001; u < 1d; u += 0.000001) {
            double x = dist.quantile(u);
            double y = dist.cdf(x);
            if (abs(u - y) > 1e-9) {
                allEquals = false;
            }
        }
        System.out.println("F(Q(u)) = u for all u = " + allEquals);
    }

    public void lognormal_distributions() {
        System.out.println("lognormal distribution");

        LogNormalDistribution dist = new LogNormalDistribution(
                1., // mean
                2. // variance
        );

        System.out.println("mean = " + dist.mean());
        System.out.println("median = " + dist.median());
        System.out.println("variance = " + dist.variance());
        System.out.println("skew = " + dist.skew());
        System.out.println("kurtosis = " + dist.kurtosis());
        System.out.println("entropy = " + dist.entropy());

        double[] xs = new double[]{
            -1000000, -10000, -1000, -100, -10, -1, -0.1, -0.01, -0.001, -0.0001,
            0, 0.0001, 0.001, 0.01, 0.1, 1, 2, 3, 10, 100, 1000};
        for (double x : xs) {
            System.out.println(String.format("F(%f) = %f", x, dist.cdf(x)));
        }
        for (double x : xs) {
            System.out.println(String.format("f(%f) = %f", x, dist.density(x)));
        }

        boolean allEquals = true;
        for (double u = 0.000001; u < 1d; u += 0.000001) {
            double x = dist.quantile(u);
            double y = dist.cdf(x);
            if (abs(u - y) > 1e-9) {
                allEquals = false;
            }
        }
        System.out.println("F(Q(u)) = u for all u = " + allEquals);
    }

    public void normal_distributions() {
        System.out.println("normal distribution");

        NormalDistribution dist = new NormalDistribution(
                0., // mean
                1. // variance
        );

        System.out.println("mean = " + dist.mean());
        System.out.println("median = " + dist.median());
        System.out.println("variance = " + dist.variance());
        System.out.println("skew = " + dist.skew());
        System.out.println("kurtosis = " + dist.kurtosis());
        System.out.println("entropy = " + dist.entropy()); // Math.log(2 * Math.PI * Math.E)

        double[] xs = new double[]{
            -1000000, -10000, -1000, -100, -10, -1, -0.1, -0.01, -0.001, -0.0001,
            0, 0.0001, 0.001, 0.01, 0.1, 1, 2, 3, 10, 100, 1000};
        for (double x : xs) {
            System.out.println(String.format("F(%f) = %f", x, dist.cdf(x)));
        }
        for (double x : xs) {
            System.out.println(String.format("f(%f) = %f", x, dist.density(x)));
        }

        boolean allEquals = true;
        for (double u = 0.000001; u < 1d; u += 0.000001) {
            double x = dist.quantile(u);
            double y = dist.cdf(x);
            if (abs(u - y) > 1e-9) {
                allEquals = false;
            }
        }
        System.out.println("F(Q(u)) = u for all u = " + allEquals);
    }

    public void LedoitWolf2016() {
        /*
    * These are the stock returns for MSFT, YHOO, GOOG, AAPL, MS, XOM from 
    * Aug 20, 2012 to Jan 15, 2013 (i.e., returns for 100 days).
    *
    * Case1 n>>p
    * n=100,p=6
    *
    * R code:
    
## Matrix A ##
matrix_A<-matrix(c(        0.001968, 0.000668, -0.008926, -0.013668, 0.004057, -0.005492,
                           -0.008511, -0.003340, 0.011456, 0.019523, -0.002020, 0.002991,
                           -0.009244, -0.003351, -0.000561, -0.009327, -0.024291, -0.004703,
                           0.009997, 0.003362, 0.002704, 0.000879, 0.004149, 0.008413,
                           0.004289, -0.004692, -0.013866, 0.018797, -0.002066, -0.003543,
                           -0.001971, -0.008754, 0.011999, -0.001308, 0.004831, 0.004129,
                           0.000658, 0.008152, 0.015888, -0.001965, 0.014423, -0.002284,
                           -0.010855, -0.011456, -0.009200, -0.014260, 0.006093, -0.007899,
                           0.016628, -0.001363, 0.005002, 0.002073, 0.006729, 0.001154,
                           -0.014066, 0.016382, -0.005912, 0.014617, 0.033422, -0.002075,
                           0.000000, 0.013432, -0.000470, -0.007025, 0.010996, 0.002426,
                           0.031520, 0.001325, 0.027442, 0.009023, 0.036468, 0.019011,
                           -0.012544, 0.007280, 0.009651, 0.006165, 0.051235, 0.010403,
                           -0.007492, -0.007227, -0.007619, -0.026013, -0.027598, -0.004924,
                           0.002297, 0.003309, -0.012244, -0.003244, 0.038647, 0.001574,
                           -0.000327, 0.015831, -0.001893, 0.013914, 0.009884, -0.000786,
                           0.005241, 0.012987, 0.021943, 0.019693, 0.027634, 0.018766,
                           0.008798, 0.010897, 0.005156, 0.012164, 0.019048, 0.011802,
                           0.000000, -0.005707, 0.000423, 0.012294, -0.024189, -0.004252,
                           -0.000969, 0.014668, 0.011690, 0.003043, -0.009577, -0.002847,
                           -0.004203, -0.003143, 0.012836, 0.000272, -0.003413, -0.011748,
                           0.012662, -0.004414, 0.000852, -0.004850, -0.020548, 0.010443,
                           -0.008015, -0.003167, 0.008062, 0.001999, -0.007576, 0.004398,
                           -0.013251, 0.016518, 0.020968, -0.013287, -0.002349, -0.000438,
                           -0.012774, -0.020000, -0.000294, -0.024969, -0.025898, -0.001533,
                           -0.007299, -0.004464, 0.005740, -0.012409, -0.010272, -0.005594,
                           -0.000334, 0.027546, 0.004035, 0.024254, 0.025031, 0.006287,
                           -0.013039, -0.003741, -0.002644, -0.020863, -0.005956, -0.003836,
                           -0.009146, -0.009387, 0.009649, -0.011565, 0.002996, 0.003851,
                           0.005812, 0.006949, -0.006288, 0.002910, 0.007168, -0.000877,
                           0.006798, 0.016939, 0.007279, 0.015343, 0.007117, -0.000219,
                           0.005739, 0.003701, 0.007279, -0.006927, 0.025913, 0.005706,
                           -0.006042, -0.011063, -0.000521, -0.021318, 0.001722, 0.003492,
                           -0.002364, -0.003729, -0.012779, -0.022090, -0.002865, 0.001414,
                           -0.016926, -0.011229, -0.018144, -0.003636, -0.005747, -0.005863,
                           -0.010331, -0.001262, 0.000632, 0.007963, 0.002890, -0.012014,
                           -0.001044, 0.005685, 0.009294, -0.020000, 0.026513, 0.001548,
                           0.008708, -0.002513, -0.008956, 0.002575, -0.030882, -0.001545,
                           0.010704, -0.012594, -0.005062, 0.008008, 0.025492, 0.005306,
                           -0.000683, 0.015306, 0.005020, 0.023692, 0.006780, 0.009567,
                           0.003419, 0.010678, 0.014489, -0.007977, 0.034792, 0.010892,
                           -0.003066, -0.005594, -0.080067, -0.018576, -0.037961, 0.000970,
                           -0.029050, -0.010000, -0.019007, -0.036030, -0.014656, -0.014209,
                           -0.022527, -0.004419, -0.004576, 0.039666, -0.004577, 0.000437,
                           0.001801, 0.057070, 0.002475, -0.032607, -0.019540, -0.021829,
                           -0.005392, -0.007199, -0.004483, 0.005667, 0.004103, -0.003347,
                           -0.000723, 0.003625, 0.000679, -0.011824, -0.004670, 0.006158,
                           0.011935, 0.010837, -0.003851, -0.009097, -0.003519, 0.002114,
                           0.011794, 0.002978, 0.007628, -0.014370, 0.022955, 0.005996,
                           0.034264, 0.006532, 0.010716, 0.002059, 0.013234, 0.004746,
                           -0.000683, 0.009440, 0.000480, -0.033090, 0.009654, -0.014501,
                           0.004443, 0.015196, -0.007210, 0.013550, -0.001687, 0.004013,
                           0.007826, 0.005181, -0.001816, -0.003024, 0.024789, 0.010769,
                           -0.026334, -0.004009, -0.021416, -0.038263, -0.085761, -0.031415,
                           -0.009015, -0.008626, -0.022230, -0.036290, -0.006615, -0.012588,
                           0.000700, 0.001160, 0.016465, 0.017313, 0.005448, 0.001608,
                           -0.021329, 0.014484, 0.004329, -0.007732, 0.009633, 0.001261,
                           -0.032154, 0.019417, -0.010287, 0.000129, -0.014908, -0.009734,
                           -0.009228, -0.001120, -0.009863, -0.011089, -0.026029, -0.004626,
                           -0.006706, 0.003365, -0.008107, -0.020973, 0.010566, 0.000813,
                           -0.005251, -0.001677, -0.000124, 0.003919, -0.004920, 0.003599,
                           0.007919, 0.027996, 0.032495, 0.072108, 0.021014, 0.014112,
                           -0.000748, -0.006536, 0.002634, -0.008520, -0.010291, -0.001939,
                           0.008985, 0.008772, -0.006120, 0.001408, -0.006116, 0.005829,
                           0.027829, 0.009239, 0.003154, 0.017447, 0.011077, 0.012271,
                           -0.011191, 0.010232, -0.010210, 0.031549, 0.010956, -0.005276,
                           -0.011318, 0.009062, 0.014460, -0.008057, 0.001204, -0.014331,
                           0.010340, -0.001057, 0.019323, -0.003146, 0.015033, 0.008586,
                           -0.014985, -0.002115, 0.012023, 0.011013, -0.001185, 0.000227,
                           -0.012245, -0.005299, 0.009366, -0.006923, 0.000593, 0.000227,
                           -0.007137, -0.011721, -0.004468, 0.001555, -0.023711, -0.006013,
                           -0.002270, 0.020485, -0.006070, -0.017639, 0.008500, -0.004794,
                           0.011377, -0.002113, -0.004645, -0.064357, 0.022276, 0.006193,
                           0.002250, 0.016411, 0.004812, 0.015683, -0.014134, 0.003078,
                           -0.010101, 0.000000, -0.010013, -0.025565, 0.013740, 0.006818,
                           0.018141, 0.011979, 0.001768, -0.006432, 0.002357, -0.002144,
                           0.014105, 0.004632, 0.016720, 0.021838, 0.043504, 0.006560,
                           -0.002928, -0.007172, 0.000976, -0.004415, -0.002817, 0.005169,
                           -0.004772, -0.001548, 0.007369, -0.017273, 0.005650, -0.009726,
                           -0.011066, 0.014987, -0.001053, -0.037569, 0.014045, -0.005645,
                           0.010817, 0.002546, 0.026811, 0.017733, 0.026593, 0.008969,
                           0.016974, -0.003555, 0.000402, 0.029046, 0.031840, 0.007764,
                           -0.009071, -0.001019, -0.001331, -0.014216, -0.001569, -0.012506,
                           0.013548, 0.004592, 0.003125, -0.008702, 0.009429, 0.005088,
                           -0.008309, -0.017268, -0.009317, -0.004600, -0.018163, -0.018675,
                           -0.014208, 0.015504, -0.008566, 0.001617, 0.001586, -0.003554,
                           -0.007391, -0.004071, -0.000888, -0.013784, -0.003694, 0.001726,
                           0.003723, 0.001533, -0.003640, 0.004016, -0.005826, -0.002412,
                           -0.015208, -0.005102, -0.008892, -0.010620, -0.007991, -0.020262,
                           0.006026, 0.020513, 0.010528, 0.044310, 0.026853, 0.017039,
                           0.034070, 0.009045, 0.022435, 0.031682, 0.026151, 0.024957,
                           -0.013396, -0.014940, 0.000581, -0.012622, -0.002039, -0.001804,
                           -0.018716, 0.004044, 0.019760, -0.027855, 0.031154, 0.004630,
                           -0.001870, -0.023162, -0.004363, -0.005882, -0.019316, -0.011578,
                           -0.005245, 0.013402, -0.001973, 0.002691, -0.007576, 0.006255,
                           0.005650, -0.017294, 0.006573, -0.015629, -0.001527, -0.003843,
                           -0.008989, -0.017081, 0.004552, 0.012396, 0.036697, 0.010892,
                           0.013983, 0.015798, -0.002009, -0.006132, -0.008358, 0.005724,
                           0.002236, 0.007258, -0.022622, -0.035653, -0.004958, -0.000335,
                           0.011900, 0.004632, 0.002323, -0.031550, 0.017937, -0.000558), ncol=6,nrow=100)
    
    
    *
    * library("nlshrink")
    * A<-read.csv("Matrix A.txt",sep=",",header=FALSE)
    * A<-as.matrix(as.data.frame(A))
    * nlshrink_cov(A)
         */
        Matrix X = new DenseMatrix(new double[][]{
            {0.001968, 0.000668, -0.008926, -0.013668, 0.004057, -0.005492},
            {-0.008511, -0.003340, 0.011456, 0.019523, -0.002020, 0.002991},
            {-0.009244, -0.003351, -0.000561, -0.009327, -0.024291, -0.004703},
            {0.009997, 0.003362, 0.002704, 0.000879, 0.004149, 0.008413},
            {0.004289, -0.004692, -0.013866, 0.018797, -0.002066, -0.003543},
            {-0.001971, -0.008754, 0.011999, -0.001308, 0.004831, 0.004129},
            {0.000658, 0.008152, 0.015888, -0.001965, 0.014423, -0.002284},
            {-0.010855, -0.011456, -0.009200, -0.014260, 0.006093, -0.007899},
            {0.016628, -0.001363, 0.005002, 0.002073, 0.006729, 0.001154},
            {-0.014066, 0.016382, -0.005912, 0.014617, 0.033422, -0.002075},
            {0.000000, 0.013432, -0.000470, -0.007025, 0.010996, 0.002426},
            {0.031520, 0.001325, 0.027442, 0.009023, 0.036468, 0.019011},
            {-0.012544, 0.007280, 0.009651, 0.006165, 0.051235, 0.010403},
            {-0.007492, -0.007227, -0.007619, -0.026013, -0.027598, -0.004924},
            {0.002297, 0.003309, -0.012244, -0.003244, 0.038647, 0.001574},
            {-0.000327, 0.015831, -0.001893, 0.013914, 0.009884, -0.000786},
            {0.005241, 0.012987, 0.021943, 0.019693, 0.027634, 0.018766},
            {0.008798, 0.010897, 0.005156, 0.012164, 0.019048, 0.011802},
            {0.000000, -0.005707, 0.000423, 0.012294, -0.024189, -0.004252},
            {-0.000969, 0.014668, 0.011690, 0.003043, -0.009577, -0.002847},
            {-0.004203, -0.003143, 0.012836, 0.000272, -0.003413, -0.011748},
            {0.012662, -0.004414, 0.000852, -0.004850, -0.020548, 0.010443},
            {-0.008015, -0.003167, 0.008062, 0.001999, -0.007576, 0.004398},
            {-0.013251, 0.016518, 0.020968, -0.013287, -0.002349, -0.000438},
            {-0.012774, -0.020000, -0.000294, -0.024969, -0.025898, -0.001533},
            {-0.007299, -0.004464, 0.005740, -0.012409, -0.010272, -0.005594},
            {-0.000334, 0.027546, 0.004035, 0.024254, 0.025031, 0.006287},
            {-0.013039, -0.003741, -0.002644, -0.020863, -0.005956, -0.003836},
            {-0.009146, -0.009387, 0.009649, -0.011565, 0.002996, 0.003851},
            {0.005812, 0.006949, -0.006288, 0.002910, 0.007168, -0.000877},
            {0.006798, 0.016939, 0.007279, 0.015343, 0.007117, -0.000219},
            {0.005739, 0.003701, 0.007279, -0.006927, 0.025913, 0.005706},
            {-0.006042, -0.011063, -0.000521, -0.021318, 0.001722, 0.003492},
            {-0.002364, -0.003729, -0.012779, -0.022090, -0.002865, 0.001414},
            {-0.016926, -0.011229, -0.018144, -0.003636, -0.005747, -0.005863},
            {-0.010331, -0.001262, 0.000632, 0.007963, 0.002890, -0.012014},
            {-0.001044, 0.005685, 0.009294, -0.020000, 0.026513, 0.001548},
            {0.008708, -0.002513, -0.008956, 0.002575, -0.030882, -0.001545},
            {0.010704, -0.012594, -0.005062, 0.008008, 0.025492, 0.005306},
            {-0.000683, 0.015306, 0.005020, 0.023692, 0.006780, 0.009567},
            {0.003419, 0.010678, 0.014489, -0.007977, 0.034792, 0.010892},
            {-0.003066, -0.005594, -0.080067, -0.018576, -0.037961, 0.000970},
            {-0.029050, -0.010000, -0.019007, -0.036030, -0.014656, -0.014209},
            {-0.022527, -0.004419, -0.004576, 0.039666, -0.004577, 0.000437},
            {0.001801, 0.057070, 0.002475, -0.032607, -0.019540, -0.021829},
            {-0.005392, -0.007199, -0.004483, 0.005667, 0.004103, -0.003347},
            {-0.000723, 0.003625, 0.000679, -0.011824, -0.004670, 0.006158},
            {0.011935, 0.010837, -0.003851, -0.009097, -0.003519, 0.002114},
            {0.011794, 0.002978, 0.007628, -0.014370, 0.022955, 0.005996},
            {0.034264, 0.006532, 0.010716, 0.002059, 0.013234, 0.004746},
            {-0.000683, 0.009440, 0.000480, -0.033090, 0.009654, -0.014501},
            {0.004443, 0.015196, -0.007210, 0.013550, -0.001687, 0.004013},
            {0.007826, 0.005181, -0.001816, -0.003024, 0.024789, 0.010769},
            {-0.026334, -0.004009, -0.021416, -0.038263, -0.085761, -0.031415},
            {-0.009015, -0.008626, -0.022230, -0.036290, -0.006615, -0.012588},
            {0.000700, 0.001160, 0.016465, 0.017313, 0.005448, 0.001608},
            {-0.021329, 0.014484, 0.004329, -0.007732, 0.009633, 0.001261},
            {-0.032154, 0.019417, -0.010287, 0.000129, -0.014908, -0.009734},
            {-0.009228, -0.001120, -0.009863, -0.011089, -0.026029, -0.004626},
            {-0.006706, 0.003365, -0.008107, -0.020973, 0.010566, 0.000813},
            {-0.005251, -0.001677, -0.000124, 0.003919, -0.004920, 0.003599},
            {0.007919, 0.027996, 0.032495, 0.072108, 0.021014, 0.014112},
            {-0.000748, -0.006536, 0.002634, -0.008520, -0.010291, -0.001939},
            {0.008985, 0.008772, -0.006120, 0.001408, -0.006116, 0.005829},
            {0.027829, 0.009239, 0.003154, 0.017447, 0.011077, 0.012271},
            {-0.011191, 0.010232, -0.010210, 0.031549, 0.010956, -0.005276},
            {-0.011318, 0.009062, 0.014460, -0.008057, 0.001204, -0.014331},
            {0.010340, -0.001057, 0.019323, -0.003146, 0.015033, 0.008586},
            {-0.014985, -0.002115, 0.012023, 0.011013, -0.001185, 0.000227},
            {-0.012245, -0.005299, 0.009366, -0.006923, 0.000593, 0.000227},
            {-0.007137, -0.011721, -0.004468, 0.001555, -0.023711, -0.006013},
            {-0.002270, 0.020485, -0.006070, -0.017639, 0.008500, -0.004794},
            {0.011377, -0.002113, -0.004645, -0.064357, 0.022276, 0.006193},
            {0.002250, 0.016411, 0.004812, 0.015683, -0.014134, 0.003078},
            {-0.010101, 0.000000, -0.010013, -0.025565, 0.013740, 0.006818},
            {0.018141, 0.011979, 0.001768, -0.006432, 0.002357, -0.002144},
            {0.014105, 0.004632, 0.016720, 0.021838, 0.043504, 0.006560},
            {-0.002928, -0.007172, 0.000976, -0.004415, -0.002817, 0.005169},
            {-0.004772, -0.001548, 0.007369, -0.017273, 0.005650, -0.009726},
            {-0.011066, 0.014987, -0.001053, -0.037569, 0.014045, -0.005645},
            {0.010817, 0.002546, 0.026811, 0.017733, 0.026593, 0.008969},
            {0.016974, -0.003555, 0.000402, 0.029046, 0.031840, 0.007764},
            {-0.009071, -0.001019, -0.001331, -0.014216, -0.001569, -0.012506},
            {0.013548, 0.004592, 0.003125, -0.008702, 0.009429, 0.005088},
            {-0.008309, -0.017268, -0.009317, -0.004600, -0.018163, -0.018675},
            {-0.014208, 0.015504, -0.008566, 0.001617, 0.001586, -0.003554},
            {-0.007391, -0.004071, -0.000888, -0.013784, -0.003694, 0.001726},
            {0.003723, 0.001533, -0.003640, 0.004016, -0.005826, -0.002412},
            {-0.015208, -0.005102, -0.008892, -0.010620, -0.007991, -0.020262},
            {0.006026, 0.020513, 0.010528, 0.044310, 0.026853, 0.017039},
            {0.034070, 0.009045, 0.022435, 0.031682, 0.026151, 0.024957},
            {-0.013396, -0.014940, 0.000581, -0.012622, -0.002039, -0.001804},
            {-0.018716, 0.004044, 0.019760, -0.027855, 0.031154, 0.004630},
            {-0.001870, -0.023162, -0.004363, -0.005882, -0.019316, -0.011578},
            {-0.005245, 0.013402, -0.001973, 0.002691, -0.007576, 0.006255},
            {0.005650, -0.017294, 0.006573, -0.015629, -0.001527, -0.003843},
            {-0.008989, -0.017081, 0.004552, 0.012396, 0.036697, 0.010892},
            {0.013983, 0.015798, -0.002009, -0.006132, -0.008358, 0.005724},
            {0.002236, 0.007258, -0.022622, -0.035653, -0.004958, -0.000335},
            {0.011900, 0.004632, 0.002323, -0.031550, 0.017937, -0.000558}
        });

        LedoitWolf2016.Result result1
                = new LedoitWolf2016().estimate(X);
        // the Ledoi-Wolf nonlinearly-shrunk covariance matrix
        Matrix S_nlShrunk = result1.getShrunkCovarianceMatrix();
        System.out.println("Ledoit-Wolf-2016 non-linearly shrunk covariance matrix is:");
        System.out.println(S_nlShrunk);

        LedoitWolf2004.Result result2
                = new LedoitWolf2004(false).compute(X); // use biased sample (as Wolf's code)
        // the Ledoi-Wolf linearly-shrunk covariance matrix
        Matrix S_lshrunk = result2.getCovarianceMatrix();
        System.out.println("Ledoit-Wolf-2004 linearly shrunk covariance matrix is:");
        System.out.println(S_lshrunk);

        // the same covariance matrix
        Matrix S = new SampleCovariance(X);
        System.out.println("sample covariance =");
        System.out.println(S);
    }

    public void LedoitWolf2004() {
        /*
         * These are the stock returns for MSFT, YHOO, GOOG, AAPL, MS, XOM from
         * Aug 20, 2012 to Jan 15, 2013 (i.e., returns for 100 days).
         */
        Matrix X = new DenseMatrix(new double[][]{
            {0.001968, 0.000668, -0.008926, -0.013668, 0.004057, -0.005492},
            {-0.008511, -0.003340, 0.011456, 0.019523, -0.002020, 0.002991},
            {-0.009244, -0.003351, -0.000561, -0.009327, -0.024291, -0.004703},
            {0.009997, 0.003362, 0.002704, 0.000879, 0.004149, 0.008413},
            {0.004289, -0.004692, -0.013866, 0.018797, -0.002066, -0.003543},
            {-0.001971, -0.008754, 0.011999, -0.001308, 0.004831, 0.004129},
            {0.000658, 0.008152, 0.015888, -0.001965, 0.014423, -0.002284},
            {-0.010855, -0.011456, -0.009200, -0.014260, 0.006093, -0.007899},
            {0.016628, -0.001363, 0.005002, 0.002073, 0.006729, 0.001154},
            {-0.014066, 0.016382, -0.005912, 0.014617, 0.033422, -0.002075},
            {0.000000, 0.013432, -0.000470, -0.007025, 0.010996, 0.002426},
            {0.031520, 0.001325, 0.027442, 0.009023, 0.036468, 0.019011},
            {-0.012544, 0.007280, 0.009651, 0.006165, 0.051235, 0.010403},
            {-0.007492, -0.007227, -0.007619, -0.026013, -0.027598, -0.004924},
            {0.002297, 0.003309, -0.012244, -0.003244, 0.038647, 0.001574},
            {-0.000327, 0.015831, -0.001893, 0.013914, 0.009884, -0.000786},
            {0.005241, 0.012987, 0.021943, 0.019693, 0.027634, 0.018766},
            {0.008798, 0.010897, 0.005156, 0.012164, 0.019048, 0.011802},
            {0.000000, -0.005707, 0.000423, 0.012294, -0.024189, -0.004252},
            {-0.000969, 0.014668, 0.011690, 0.003043, -0.009577, -0.002847},
            {-0.004203, -0.003143, 0.012836, 0.000272, -0.003413, -0.011748},
            {0.012662, -0.004414, 0.000852, -0.004850, -0.020548, 0.010443},
            {-0.008015, -0.003167, 0.008062, 0.001999, -0.007576, 0.004398},
            {-0.013251, 0.016518, 0.020968, -0.013287, -0.002349, -0.000438},
            {-0.012774, -0.020000, -0.000294, -0.024969, -0.025898, -0.001533},
            {-0.007299, -0.004464, 0.005740, -0.012409, -0.010272, -0.005594},
            {-0.000334, 0.027546, 0.004035, 0.024254, 0.025031, 0.006287},
            {-0.013039, -0.003741, -0.002644, -0.020863, -0.005956, -0.003836},
            {-0.009146, -0.009387, 0.009649, -0.011565, 0.002996, 0.003851},
            {0.005812, 0.006949, -0.006288, 0.002910, 0.007168, -0.000877},
            {0.006798, 0.016939, 0.007279, 0.015343, 0.007117, -0.000219},
            {0.005739, 0.003701, 0.007279, -0.006927, 0.025913, 0.005706},
            {-0.006042, -0.011063, -0.000521, -0.021318, 0.001722, 0.003492},
            {-0.002364, -0.003729, -0.012779, -0.022090, -0.002865, 0.001414},
            {-0.016926, -0.011229, -0.018144, -0.003636, -0.005747, -0.005863},
            {-0.010331, -0.001262, 0.000632, 0.007963, 0.002890, -0.012014},
            {-0.001044, 0.005685, 0.009294, -0.020000, 0.026513, 0.001548},
            {0.008708, -0.002513, -0.008956, 0.002575, -0.030882, -0.001545},
            {0.010704, -0.012594, -0.005062, 0.008008, 0.025492, 0.005306},
            {-0.000683, 0.015306, 0.005020, 0.023692, 0.006780, 0.009567},
            {0.003419, 0.010678, 0.014489, -0.007977, 0.034792, 0.010892},
            {-0.003066, -0.005594, -0.080067, -0.018576, -0.037961, 0.000970},
            {-0.029050, -0.010000, -0.019007, -0.036030, -0.014656, -0.014209},
            {-0.022527, -0.004419, -0.004576, 0.039666, -0.004577, 0.000437},
            {0.001801, 0.057070, 0.002475, -0.032607, -0.019540, -0.021829},
            {-0.005392, -0.007199, -0.004483, 0.005667, 0.004103, -0.003347},
            {-0.000723, 0.003625, 0.000679, -0.011824, -0.004670, 0.006158},
            {0.011935, 0.010837, -0.003851, -0.009097, -0.003519, 0.002114},
            {0.011794, 0.002978, 0.007628, -0.014370, 0.022955, 0.005996},
            {0.034264, 0.006532, 0.010716, 0.002059, 0.013234, 0.004746},
            {-0.000683, 0.009440, 0.000480, -0.033090, 0.009654, -0.014501},
            {0.004443, 0.015196, -0.007210, 0.013550, -0.001687, 0.004013},
            {0.007826, 0.005181, -0.001816, -0.003024, 0.024789, 0.010769},
            {-0.026334, -0.004009, -0.021416, -0.038263, -0.085761, -0.031415},
            {-0.009015, -0.008626, -0.022230, -0.036290, -0.006615, -0.012588},
            {0.000700, 0.001160, 0.016465, 0.017313, 0.005448, 0.001608},
            {-0.021329, 0.014484, 0.004329, -0.007732, 0.009633, 0.001261},
            {-0.032154, 0.019417, -0.010287, 0.000129, -0.014908, -0.009734},
            {-0.009228, -0.001120, -0.009863, -0.011089, -0.026029, -0.004626},
            {-0.006706, 0.003365, -0.008107, -0.020973, 0.010566, 0.000813},
            {-0.005251, -0.001677, -0.000124, 0.003919, -0.004920, 0.003599},
            {0.007919, 0.027996, 0.032495, 0.072108, 0.021014, 0.014112},
            {-0.000748, -0.006536, 0.002634, -0.008520, -0.010291, -0.001939},
            {0.008985, 0.008772, -0.006120, 0.001408, -0.006116, 0.005829},
            {0.027829, 0.009239, 0.003154, 0.017447, 0.011077, 0.012271},
            {-0.011191, 0.010232, -0.010210, 0.031549, 0.010956, -0.005276},
            {-0.011318, 0.009062, 0.014460, -0.008057, 0.001204, -0.014331},
            {0.010340, -0.001057, 0.019323, -0.003146, 0.015033, 0.008586},
            {-0.014985, -0.002115, 0.012023, 0.011013, -0.001185, 0.000227},
            {-0.012245, -0.005299, 0.009366, -0.006923, 0.000593, 0.000227},
            {-0.007137, -0.011721, -0.004468, 0.001555, -0.023711, -0.006013},
            {-0.002270, 0.020485, -0.006070, -0.017639, 0.008500, -0.004794},
            {0.011377, -0.002113, -0.004645, -0.064357, 0.022276, 0.006193},
            {0.002250, 0.016411, 0.004812, 0.015683, -0.014134, 0.003078},
            {-0.010101, 0.000000, -0.010013, -0.025565, 0.013740, 0.006818},
            {0.018141, 0.011979, 0.001768, -0.006432, 0.002357, -0.002144},
            {0.014105, 0.004632, 0.016720, 0.021838, 0.043504, 0.006560},
            {-0.002928, -0.007172, 0.000976, -0.004415, -0.002817, 0.005169},
            {-0.004772, -0.001548, 0.007369, -0.017273, 0.005650, -0.009726},
            {-0.011066, 0.014987, -0.001053, -0.037569, 0.014045, -0.005645},
            {0.010817, 0.002546, 0.026811, 0.017733, 0.026593, 0.008969},
            {0.016974, -0.003555, 0.000402, 0.029046, 0.031840, 0.007764},
            {-0.009071, -0.001019, -0.001331, -0.014216, -0.001569, -0.012506},
            {0.013548, 0.004592, 0.003125, -0.008702, 0.009429, 0.005088},
            {-0.008309, -0.017268, -0.009317, -0.004600, -0.018163, -0.018675},
            {-0.014208, 0.015504, -0.008566, 0.001617, 0.001586, -0.003554},
            {-0.007391, -0.004071, -0.000888, -0.013784, -0.003694, 0.001726},
            {0.003723, 0.001533, -0.003640, 0.004016, -0.005826, -0.002412},
            {-0.015208, -0.005102, -0.008892, -0.010620, -0.007991, -0.020262},
            {0.006026, 0.020513, 0.010528, 0.044310, 0.026853, 0.017039},
            {0.034070, 0.009045, 0.022435, 0.031682, 0.026151, 0.024957},
            {-0.013396, -0.014940, 0.000581, -0.012622, -0.002039, -0.001804},
            {-0.018716, 0.004044, 0.019760, -0.027855, 0.031154, 0.004630},
            {-0.001870, -0.023162, -0.004363, -0.005882, -0.019316, -0.011578},
            {-0.005245, 0.013402, -0.001973, 0.002691, -0.007576, 0.006255},
            {0.005650, -0.017294, 0.006573, -0.015629, -0.001527, -0.003843},
            {-0.008989, -0.017081, 0.004552, 0.012396, 0.036697, 0.010892},
            {0.013983, 0.015798, -0.002009, -0.006132, -0.008358, 0.005724},
            {0.002236, 0.007258, -0.022622, -0.035653, -0.004958, -0.000335},
            {0.011900, 0.004632, 0.002323, -0.031550, 0.017937, -0.000558}
        });
        /*
         * From Wolf's implementation (http://www.econ.uzh.ch/faculty/wolf/publications.html#9):
         * phi = 4.11918014563813e-06
         * rho = 2.59272437810913e-06
         * gamma = 1.64807384775746e-08
         * kappa = 9.26205927972248e+01
         * shrinkage = 9.26205927972248e-01
         * sigma =
         * 1e-4 *
         * 1.515632920116000 0.485389976957753 0.569819071905581 0.832527350192132 0.847148840587289
         * 0.397609074332363
         * 0.485389976957753 1.385321425539000 0.536232629487419 0.789078831932216 0.788206205379818
         * 0.348026133393798
         * 0.569819071905581 0.536232629487419 1.821791453675000 0.934388133855881 0.952823814063320
         * 0.422051935833047
         * 0.832527350192132 0.789078831932216 0.934388133855881 3.948763689179000 1.349588317003072
         * 0.628591706535889
         * 0.847148840587289 0.788206205379818 0.952823814063320 1.349588317003072 3.907173836600000
         * 0.644231481562656
         * 0.397609074332363 0.348026133393798 0.422051935833047 0.628591706535889 0.644231481562656
         * 0.792958306075000
         */
        LedoitWolf2004.Result result
                = new LedoitWolf2004(false).compute(X); // use biased sample (as Wolf's code)
        // the Ledoi-Wolf linearly-shrunk covariance matrix
        Matrix S_lshrunk = result.getCovarianceMatrix();
        System.out.println("Ledoit-Wolf-2004 linearly shrunk covariance matrix is:");
        System.out.println(S_lshrunk);

        // the same covariance matrix
        Matrix S = new SampleCovariance(X);
        System.out.println("sample covariance =");
        System.out.println(S);
    }

    public void quantile() {
        System.out.println("quantiles");

        double[] x = new double[]{0, 1, 2, 3, 3, 3, 6, 7, 8, 9}; // with repeated observations
        double[] qs = new double[]{1e-10, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.}; // qu

        System.out.println("APPROXIMATELY_MEDIAN_UNBIASED");
        Quantile stat1 = new Quantile(
                x,
                Quantile.QuantileType.APPROXIMATELY_MEDIAN_UNBIASED
        );
        System.out.println("number of samples = " + stat1.N());
        for (double q : qs) {
            System.out.println(String.format("Q(%f) = %f", q, stat1.value(q)));
        }
        System.out.println("the median = " + stat1.value(0.5));
        System.out.println("the 100% quantile = " + stat1.value(1.));
        System.out.println("the maximum = " + new Max(x).value());

        System.out.println("NEAREST_EVEN_ORDER_STATISTICS");
        Quantile stat2 = new Quantile(
                x,
                Quantile.QuantileType.NEAREST_EVEN_ORDER_STATISTICS);
        System.out.println("number of samples = " + stat2.N());
        for (double q : qs) {
            System.out.println(String.format("Q(%f) = %f", q, stat2.value(q)));
        }
        System.out.println("the median = " + stat2.value(0.5));
        System.out.println("the 1e-10 quantile = " + stat2.value(1e-10));
        System.out.println("the minimun = " + new Min(x).value());
    }

    public void rank() {
        System.out.println("rank");

        double[] x = new double[]{3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5};
        System.out.println("ranking: " + Arrays.toString(x));

        Rank rank = new Rank(x, Rank.TiesMethod.AS_26); // default implementation
        System.out.println("AS_26 rank: " + Arrays.toString(rank.ranks()));

        rank = new Rank(x, Rank.TiesMethod.AVERAGE);
        System.out.println("AVERAGE rank: " + Arrays.toString(rank.ranks()));

        rank = new Rank(x, Rank.TiesMethod.FIRST);
        System.out.println("FIRST rank: " + Arrays.toString(rank.ranks()));

        rank = new Rank(x, Rank.TiesMethod.LAST);
        System.out.println("LAST rank: " + Arrays.toString(rank.ranks()));

        rank = new Rank(x, Rank.TiesMethod.MAX);
        System.out.println("MAX rank: " + Arrays.toString(rank.ranks()));

        rank = new Rank(x, Rank.TiesMethod.MIN);
        System.out.println("MIN rank: " + Arrays.toString(rank.ranks()));

        rank = new Rank(x, Rank.TiesMethod.RANDOM);
        System.out.println("RANDOM rank: " + Arrays.toString(rank.ranks()));
    }

    public void sample_statistics() {
        System.out.println("sample statistics");

        // the sample data set
        double[] X1 = new double[]{2, 3, 3, 1};

        // compute the mean of the data set
        Mean mean = new Mean(X1);
        System.out.println("sample size = " + mean.N());
        System.out.println("sample mean = " + mean.value());

        // the sample data set
        double[] X2 = new double[]{82, 94, 90, 83, 87};
        double[] W2 = new double[]{1, 4, 8, 4, 4};

        // compute the mean of the data set
        WeightedMean weighted_mean = new WeightedMean(
                X2, // the data
                W2 // the weights
        );
        System.out.println("sample size = " + mean.N());
        System.out.println("sample weighted mean = " + weighted_mean.value());

        // the sample data set
        double[] X3 = {2, 3, 3, 1};

        // compute the biased and unbiased vairances and standard deviations
        Variance var1 = new Variance(X3, false); // biased
        System.out.println("sample standard deviation (biased) = " + var1.standardDeviation());
        System.out.println("sample variance (biased) = " + var1.value());
        Variance var2 = new Variance(X3, true); // unbiased
        System.out.println("sample standard deviation (unbiased) = " + var2.standardDeviation());
        System.out.println("sample variance (unbiased) = " + var2.value());

        // compute the biased and unbiased vairances and standard deviations
        WeightedVariance wvar1 = new WeightedVariance(
                X2, // the data
                W2, // the weights
                false); // biased
        System.out.println("sample weighted standard deviation (biased) = " + wvar1.stdev());
        System.out.println("sample weighted variance (biased) = " + wvar1.value());
        WeightedVariance wvar2 = new WeightedVariance(
                X2, // the data
                W2, // the weights
                false); // unbiased
        System.out.println("sample standard deviation (unbiased) = " + wvar2.stdev());
        System.out.println("sample variance (unbiased) = " + wvar2.value());

        // the sample data set
        double[] X4 = {1, 1, 1, 1, 2, 2, 2, 3, 0};

        // compute the skewness of the data set
        Skewness skewness = new Skewness(X4);
        System.out.println("sample mean = " + skewness.mean());
        System.out.println("sample standard deviation = " + sqrt(skewness.variance()));
        System.out.println("sample skewness = " + skewness.value());

        // compute the kurtosis of the data set
        Kurtosis kurtosis = new Kurtosis(X4);
        System.out.println("sample mean = " + kurtosis.mean());
        System.out.println("sample standard deviation = " + sqrt(kurtosis.variance()));
        System.out.println("sample kurtosis = " + kurtosis.value());

        // compute moments of a data set
        Moments moments = new Moments(6); // up to the 6th moments

        // data generated using rexp in R with λ = 1
        moments.addData(new double[]{
            1.050339964176429, 0.906121669295144, 0.116647826876888,
            4.579895872370673, 1.714264543643022, 0.436467861756682,
            0.860735191921604, 1.771233864044571, 0.623149028640023,
            1.058291583279980
        });
        System.out.println("sample size = " + moments.N());
        System.out.println("1st central moment = " + moments.centralMoment(1)); // ok
        System.out.println("2nd central moment = " + moments.centralMoment(2)); // ok
        System.out.println("3rd central moment = " + moments.centralMoment(3)); // off, // not enough data
        System.out.println("4th central moment = " + moments.centralMoment(4)); // way off, // not enough data
        System.out.println("5th central moment = " + moments.centralMoment(5)); // meaningless, not enough data
        System.out.println("6th central moment = " + moments.centralMoment(6)); // meaningless, not enough data

        // the sample data sets
        double[] X5 = new double[]{106, 86, 100, 101, 99, 103, 97, 113, 112, 110};
        double[] X6 = new double[]{7, 0, 27, 50, 28, 29, 20, 12, 6, 17};
        // compute the sample correlation of the data sets
        System.out.println("the sample correlation = " + new SpearmanRankCorrelation(X5, X6).value());

        // each column is a sample set; there are 5 data sets
        Matrix X7 = new DenseMatrix(new double[][]{
            {1.4022225, -0.04625344, 1.26176112, -1.8394428, 0.7182637},
            {-0.2230975, 0.91561987, 1.17086252, 0.2282348, 0.0690674},
            {0.6939930, 1.94611387, -0.82939259, 1.0905923, 0.1458883},
            {-0.4050039, 0.18818663, -0.29040783, 0.6937185, 0.4664052},
            {0.6587918, -0.10749210, 3.27376532, 0.5141217, 0.7691778},
            {-2.5275280, 0.64942255, 0.07506224, -1.0787524, 1.6217606}
        });
        // compute the sample covariance of the data sets
        Matrix cov = new SampleCovariance(X7); // a 5x5 matrix
        System.out.println("sample covariance =");
        System.out.println(cov);
    }

    public void section1() {

        /*
 * Covariance Selection
         */
        Matrix A = new DenseMatrix(new double[][]{{1, 1, 1}, {0, 2, 5}, {2, 5, -1}});
        LedoitWolf2004 ledoitWolf2004 = new LedoitWolf2004(true);
        ledoitWolf2004.compute(A).getCovarianceMatrix();
        //Estimates the covariance matrix for a given matrix Y (each columnin Y is a time-series), 
        //with the given shrinkage parameter.
        System.out.println(ledoitWolf2004.compute(A, 0.2).getCovarianceMatrix());

        DenseMatrix A8 = new DenseMatrix(
                new SymmetricMatrix(new double[][]{{2}, {3, 1}}));
        CovarianceSelectionProblem problem = new CovarianceSelectionProblem(A8, 0);
        //X - the inverse of a covariance matrix (to be estimated)
        problem.penalizedCardinality(A8);
        //X - the inverse of a covariance matrix (to be estimated)
        System.out.println(problem.penalizedL1(A8));

        /*
 * everything about moment
         */
        // Mean of the sample data 
        double[] data = {1, 3, 4};
        Mean mean = new Mean(data);
        mean.addData(6);
        System.out.println(mean.value());
        System.out.println(mean.N());
        System.out.println(mean.toString());

        // Variance of the sample data
        Variance var = new Variance(data);
        var.addData(6);
        System.out.println(var.standardDeviation());
        System.out.println(var.value());

        // Skewness of the sample data
        double[] skewdata = {1, 1, 1, 1, 2, 2, 2, 3, 0};
        Skewness skew = new Skewness(skewdata);
        System.out.println(skew.value());

        // Kurtosis of the sample data
        Kurtosis kur = new Kurtosis(skewdata);
        System.out.println(kur.value());

///*
// * Everything about covariance
// */
//		
//		// Covariance selection by using LASSO 
//		CovarianceSelectionLASSO covarLASSO = new CovarianceSelectionLASSO(problem);
//		System.out.println(covarLASSO.covariance());
//		
//		//covariance selection
//		//GLASSOFAST is the Graphical LASSO algorithm to solve the covariance selectionproblem.
//		CovarianceSelectionGLASSOFAST covarLASSOFAST = new CovarianceSelectionGLASSOFAST(problem);
//		System.out.println(covarLASSOFAST.covariance());
//		System.out.println(covarLASSOFAST.isConverged());
//		
//		// Covariance of the data with smaller observation but larger factors
//		LedoitWolf2016 covLedoi = new LedoitWolf2016();
//		System.out.println(covLedoi.estimate(A).getShrunkCovarianceMatrix());
//		System.out.println(covLedoi.estimate(A).lambda());
        // Calculation of covariance
        double[] data1 = {1, 2, 3};
        double[] data2 = {0, 2, 5};
        Covariance cov = new Covariance(data1, data2);
        System.out.println(cov);

        /*
 * Everything about Correlation
         */
        CorrelationMatrix cm = new CorrelationMatrix(A8);
        System.out.println("\n correlation" + cm);

        /*
 * Weighted mean
         */
        double[] grade = {82, 94, 90, 83, 87};
        double[] w = {1, 4, 8, 4, 4};
        WeightedMean wm = new WeightedMean(grade, w);
        System.out.println(wm.value());

        /*
 * Wighted variance
         */
        WeightedVariance wv = new WeightedVariance(grade, w);
        System.out.println(wv.value());

        /*
 * Everything about rank
         */
        // Maximum and Minimum
        double[] data3 = {0.25, 3.14, 2.52, 4.26, 5.14, 7.5, 3.46};
        Max max = new Max(data3);
        System.out.println(max.value());
        Min min = new Min(data3);
        System.out.println(min.value());

        // Quantile
        double[] data4 = {1, 3, 5, 5, 7, 8, 8, 8, 8, 8, 8, 9, 9, 9, 9, 9, 10, 10, 10, 10};
        Quantile qtl = new Quantile(data4);
        System.out.println(qtl.value(0.25)); // Q_1
        System.out.println(qtl.value(0.5)); // Q_2
        System.out.println(qtl.value(0.75)); // Q_3

        // Rank
        double[] data5 = {0.25, 2.4, 1.56, 4.23, 3.76, 5.23, 0.78, 1.42};
        Rank rnk = new Rank(data5);
        System.out.println(Arrays.toString(rnk.ranks()));

        // dev.nm.stat.distribution.discrete
        Mass<Double> mass = new Mass<Double>(0.0, 0.5);
        Mass<Double> mass2 = new Mass<Double>(1.0, 0.5);
        System.out.println(mass.outcome());
        System.out.println(mass.probability());

        List<Mass<Double>> massList = new ArrayList<>();
        massList.add(mass);
        massList.add(mass2);
        ProbabilityMassQuantile<Double> quantile = new ProbabilityMassQuantile<Double>(massList);

        System.out.println("quantile:" + quantile.quantile(0.25));
//		
//		
//		RandomLongGenerator uniform = new MersenneTwister();
//		ProbabilityMassSampler<Double> sampler = new ProbabilityMassSampler <Double>(massList, uniform);
//		System.out.println(sampler.next());
//		
//		
//		
//		//dev.nm.stat.distribution.multivariate
//		
//		
//		//AbstractBivariateProbabilityDistribution
//		BivariateEVDColesTawn tawn = new BivariateEVDColesTawn(0.2, 0.3);
//		System.out.println(tawn.density(0.2, 0.4));
//		
//		
//		//DirichletDistribution
//		Vector v1 = new DenseVector(new double[]{1.1, -2.2, 3.3});
//		double[] data6 = {0.25,2.4,1.56,4.23,3.76,5.23,0.78,1.42};
//		DirichletDistribution distribution = new DirichletDistribution(data6);
//		System.out.println(distribution.cdf(v1));
//		System.out.println(distribution.covariance());

        System.out.println("quantile:" + quantile.quantile(0.7));

//		RandomLongGenerator uniform = new MersenneTwister();
//		ProbabilityMassSampler<Double> sampler = new ProbabilityMassSampler <Double>(massList, uniform);
//		System.out.println(sampler.next());
//		
//		
//		
//		//dev.nm.stat.distribution.multivariate
//		
//		
//		//AbstractBivariateProbabilityDistribution
//		BivariateEVDColesTawn tawn = new BivariateEVDColesTawn(0.2, 0.3);
//		System.out.println(tawn.density(0.2, 0.4));
//		//DirichletDistribution
//		Vector v1 = new DenseVector(new double[]{1.1, -2.2, 3.3});
//		double[] data6 = {0.25,2.4,1.56,4.23,3.76,5.23,0.78,1.42};
//		DirichletDistribution distribution = new DirichletDistribution(data6);
//		System.out.println(distribution.cdf(v1));
//		System.out.println(distribution.covariance());
//		System.out.println("quantile:" + quantile.quantile(2));
//		RandomLongGenerator uniform1 = new MersenneTwister();
//		ProbabilityMassSampler<Double> sampler = new ProbabilityMassSampler <Double>(massList, uniform1);
//		System.out.println(sampler.next());
//		
//		
//		
//		//dev.nm.stat.distribution.multivariate
//		
//		
//		//AbstractBivariateProbabilityDistribution
//		BivariateEVDColesTawn tawn = new BivariateEVDColesTawn(0.2, 0.3);
//		System.out.println(tawn.density(0.2, 0.4));
        //DirichletDistribution
//		Vector v1 = new DenseVector(new double[]{1.1, -2.2, 3.3});
//		double[] data6 = {0.25,2.4,1.56,4.23,3.76,5.23,0.78,1.42};
//		DirichletDistribution distribution = new DirichletDistribution(data6);
//		System.out.println(distribution.cdf(v1));
//		System.out.println(distribution.covariance());
        // Univariate
//		BetaDistribution betaDistribution = new BetaDistribution(0.5, 0.5);
//		System.out.println("betaDistribution: " + betaDistribution.cdf(0.5));
//		System.out.println("betaDistribution: " + betaDistribution.density(0.5));
//		System.out.println("betaDistribution: " + betaDistribution.entropy());
//		System.out.println("betaDistribution: " + betaDistribution.mean());
//		System.out.println("betaDistribution: " + betaDistribution.variance());
//		BinomialDistribution binomialDistribution = new BinomialDistribution(5, 0.5);
//		System.out.println("binomialDistribution: " + binomialDistribution.cdf(3));
//		System.out.println("binomialDistribution: " + binomialDistribution.density(3));
//		System.out.println("binomialDistribution: " + binomialDistribution.entropy());
//		System.out.println("binomialDistribution: " + binomialDistribution.mean());
//		System.out.println("binomialDistribution: " + binomialDistribution.variance());
//		ChiSquareDistribution chiSquareDistribution = new ChiSquareDistribution(3);
//		System.out.println("chiSquareDistribution: " + chiSquareDistribution.cdf(0.97));
//		System.out.println("chiSquareDistribution: " + chiSquareDistribution.density(0.9));
//		System.out.println("chiSquareDistribution: " + chiSquareDistribution.entropy());
//		System.out.println("chiSquareDistribution: " + chiSquareDistribution.mean());
//		System.out.println("chiSquareDistribution: " + chiSquareDistribution.variance());
//		
//		EmpiricalDistribution empiricalDistribution = new EmpiricalDistribution(data5);
//		System.out.println("empiricalDistribution: " + empiricalDistribution.cdf(0.5));
//		System.out.println("empiricalDistribution: " + empiricalDistribution.density(0.5));
////		System.out.println("empiricalDistribution: " + empiricalDistribution.entropy());
//		System.out.println("empiricalDistribution: " + empiricalDistribution.mean());
//		System.out.println("empiricalDistribution: " + empiricalDistribution.variance());
//		
//		ExponentialDistribution exponentialDistribution = new ExponentialDistribution(1);
//		System.out.println("exponentialDistribution: " + exponentialDistribution.cdf(0.5));
//		System.out.println("exponentialDistribution: " + exponentialDistribution.density(0.5));
//		System.out.println("exponentialDistribution: " + exponentialDistribution.entropy());
//		System.out.println("exponentialDistribution: " + exponentialDistribution.mean());
//		System.out.println("exponentialDistribution: " + exponentialDistribution.variance());
//		FDistribution fDistribution = new FDistribution(0.5, 0.5);
//		System.out.println("fDistribution: " + fDistribution.cdf(0.5));
//		System.out.println("fDistribution: " + fDistribution.density(0.5));
////		System.out.println(fDistribution.entropy());
//		//System.out.println(fDistribution.mean());
//		//System.out.println(fDistribution.variance());
//		
//		GammaDistribution gammaDistribution = new GammaDistribution(0.5, 0.5);
//		System.out.println("gammaDistribution: " + gammaDistribution.cdf(0.5));
//		System.out.println("gammaDistribution: " + gammaDistribution.density(0.5));
////		System.out.println("gammaDistribution: " + gammaDistribution.entropy());
//		System.out.println("gammaDistribution: " + gammaDistribution.mean());
//		System.out.println("gammaDistribution: " + gammaDistribution.variance());
//		
//		LogNormalDistribution logNormalDistribution = new LogNormalDistribution(0.5, 0.5);
//		System.out.println("logNormalDistribution: " + logNormalDistribution.cdf(0.5));
//		System.out.println("logNormalDistribution: " + logNormalDistribution.density(0.5));
//		System.out.println("logNormalDistribution: " + logNormalDistribution.entropy());
//		System.out.println("logNormalDistribution: " + logNormalDistribution.mean());
//		System.out.println("logNormalDistribution: " + logNormalDistribution.variance());
//		
//		NormalDistribution normalDistribution = new NormalDistribution(0, 1);
//		System.out.println("normalDistribution: " + normalDistribution.cdf(0.5));
//		System.out.println("normalDistribution: " + normalDistribution.density(0.5));
//		System.out.println("normalDistribution: " + normalDistribution.entropy());
//		System.out.println("normalDistribution: " + normalDistribution.mean());
//		System.out.println("normalDistribution: " + normalDistribution.variance());
//		
//		PoissonDistribution poissonDistribution = new PoissonDistribution(0.5);
//		System.out.println("poissonDistribution: " + poissonDistribution.cdf(0.5));
//		System.out.println("poissonDistribution: " + poissonDistribution.density(0.5));
//		System.out.println("poissonDistribution: " + poissonDistribution.entropy());
//		System.out.println("poissonDistribution: " + poissonDistribution.mean());
//		System.out.println("poissonDistribution: " + poissonDistribution.variance());
//		
//		
//		RayleighDistribution rayleighDistribution = new RayleighDistribution(0.5);
//		System.out.println("rayleighDistribution: " + rayleighDistribution.cdf(0.5));
//		System.out.println("rayleighDistribution: " + rayleighDistribution.density(0.5));
//		System.out.println("rayleighDistribution: " + rayleighDistribution.entropy());
//		System.out.println("rayleighDistribution: " + rayleighDistribution.mean());
//		System.out.println("rayleighDistribution: " + rayleighDistribution.variance());
//		
//		TDistribution tDistribution = new TDistribution(4);
//		System.out.println("tDistribution: " + tDistribution.cdf(0.5));
//		System.out.println("tDistribution: " + tDistribution.density(0.5));
//		System.out.println("tDistribution: " + tDistribution.entropy());
//		System.out.println("tDistribution: " + tDistribution.mean());
//		System.out.println("tDistribution: " + tDistribution.variance());
//		
//		TriangularDistribution triangularDistribution = new TriangularDistribution(1, 2, 3);
//		System.out.println("triangularDistribution: " + triangularDistribution.cdf(0.5));
//		System.out.println("triangularDistribution: " + triangularDistribution.density(0.5));
//		System.out.println("triangularDistribution: " + triangularDistribution.entropy());
//		System.out.println("triangularDistribution: " + triangularDistribution.mean());
//		System.out.println("triangularDistribution: " + triangularDistribution.variance());
//		
//		
//		TruncatedNormalDistribution truncatedNormalDistribution = new TruncatedNormalDistribution(0.5, 1);
//		System.out.println("truncatedNormalDistribution: " + truncatedNormalDistribution.cdf(0.5));
//		System.out.println("truncatedNormalDistribution: " + truncatedNormalDistribution.density(0.5));
//		System.out.println("truncatedNormalDistribution: " + truncatedNormalDistribution.entropy());
//		System.out.println("truncatedNormalDistribution: " + truncatedNormalDistribution.mean());
//		System.out.println("truncatedNormalDistribution: " + truncatedNormalDistribution.variance());
//		
        WeibullDistribution weibullDistribution = new WeibullDistribution(0.5, 0.5);
        System.out.println("weibullDistribution: " + weibullDistribution.cdf(0.5));
        System.out.println("weibullDistribution: " + weibullDistribution.density(0.5));
        System.out.println("weibullDistribution: " + weibullDistribution.entropy());
        System.out.println("weibullDistribution: " + weibullDistribution.mean());
        System.out.println("weibullDistribution: " + weibullDistribution.variance());
    }

}
