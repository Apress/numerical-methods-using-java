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
import dev.nm.algebra.linear.matrix.doubles.operation.MatrixFactory;
import dev.nm.algebra.linear.vector.doubles.Vector;
import dev.nm.algebra.linear.vector.doubles.dense.DenseVector;
import dev.nm.number.DoubleUtils;
import dev.nm.stat.regression.linear.LMProblem;
import dev.nm.stat.regression.linear.glm.GLMProblem;
import dev.nm.stat.regression.linear.glm.GeneralizedLinearModel;
import dev.nm.stat.regression.linear.glm.distribution.GLMBinomial;
import dev.nm.stat.regression.linear.glm.distribution.GLMFamily;
import dev.nm.stat.regression.linear.glm.distribution.GLMPoisson;
import dev.nm.stat.regression.linear.glm.distribution.link.LinkInverse;
import dev.nm.stat.regression.linear.glm.distribution.link.LinkLogit;
import dev.nm.stat.regression.linear.glm.modelselection.BackwardElimination;
import dev.nm.stat.regression.linear.glm.modelselection.EliminationByAIC;
import dev.nm.stat.regression.linear.glm.modelselection.ForwardSelection;
import dev.nm.stat.regression.linear.glm.modelselection.SelectionByAIC;
import dev.nm.stat.regression.linear.glm.quasi.GeneralizedLinearModelQuasiFamily;
import dev.nm.stat.regression.linear.glm.quasi.QuasiGLMProblem;
import dev.nm.stat.regression.linear.glm.quasi.family.QuasiBinomial;
import dev.nm.stat.regression.linear.glm.quasi.family.QuasiFamily;
import dev.nm.stat.regression.linear.glm.quasi.family.QuasiGaussian;
import dev.nm.stat.regression.linear.lasso.ConstrainedLASSOProblem;
import dev.nm.stat.regression.linear.lasso.ConstrainedLASSObyLARS;
import dev.nm.stat.regression.linear.lasso.ConstrainedLASSObyQP;
import dev.nm.stat.regression.linear.lasso.lars.LARSFitting;
import dev.nm.stat.regression.linear.lasso.lars.LARSProblem;
import dev.nm.stat.regression.linear.logistic.LogisticRegression;
import dev.nm.stat.regression.linear.ols.OLSRegression;
import java.io.IOException;
import java.util.Arrays;

/**
 * Numerical Methods Using Java: For Data Science, Analysis, and Engineering
 *
 * @author haksunli
 * @see
 * https://www.amazon.com/Numerical-Methods-Using-Java-Engineering/dp/1484267966
 * https://nm.dev/
 */
public class Chapter14 {

    public static void main(String[] args) throws Exception {
        System.out.println("Chapter 14 demos");

        Chapter14 chapter14 = new Chapter14();
        chapter14.ols();
        chapter14.weightedOLS();
        chapter14.logistic();
        chapter14.glm_binomial_logit();
        chapter14.glm_poisson_log();
        chapter14.glm_quasibinomial_logit();
        chapter14.glm_gaussian_inverse();
        chapter14.forward_selection();
        chapter14.backward_elimination();
        chapter14.lasso_QP();
        chapter14.lasso_LARS();
        chapter14.LARS();
    }

    public void LARS() throws IOException {
        System.out.println("LARS fitting");

        // construct a LARS problem
        LARSProblem problem = new LARSProblem(
                diabetes_y,
                diabetes_X,
                true); // use LASSO variation

        // run the LARS fitting algorithm
        LARSFitting fit = new LARSFitting(problem, 1e-8, 100);

        LARSFitting.Estimators estimators = fit.getEstimators();
        // The the sequence of actions taken: they are the variables added or dropped in each iteration.
        System.out.println("action sequence: " + estimators.actions());
        // The entire sequence of estimated LARS regression coefficients, scaled by the L2 norm of each row.
        System.out.println("sequence of estimated LARS regression coefficients:");
        System.out.println(estimators.scaledBetas());
    }

    public void lasso_LARS() throws IOException {
        System.out.println("LASSO regression by LARS");

        // the regularization penalty
        double t = 0.;

        // construct a constrained LASSO problem
        ConstrainedLASSOProblem problem
                = new ConstrainedLASSOProblem(diabetes_y, diabetes_X, t);
        // run LASSO regression
        Vector betaHat = new ConstrainedLASSObyLARS(problem).beta().betaHat();
        System.out.println("beta^ = " + betaHat);

        t = 1500.;
        problem = new ConstrainedLASSOProblem(diabetes_y, diabetes_X, t);
        betaHat = new ConstrainedLASSObyLARS(problem).beta().betaHat();
        System.out.println("beta^ = " + betaHat);

        // relaxing the constraint; essentially the same as the OLS solution
        t = 10000.;
        problem = new ConstrainedLASSOProblem(diabetes_y, diabetes_X, t);
        betaHat = new ConstrainedLASSObyLARS(problem).beta().betaHat();
        System.out.println("beta^ = " + betaHat);
    }

    public void lasso_QP() throws IOException {
        System.out.println("LASSO regression by QP");

        // the regularization penalty
        double t = 0.;

        // construct a constrained LASSO problem
        ConstrainedLASSOProblem problem
                = new ConstrainedLASSOProblem(diabetes_y, diabetes_X, t);
        // run LASSO regression
        Vector betaHat = new ConstrainedLASSObyQP(problem).beta().betaHat();
        System.out.println("beta^ = " + betaHat);

        t = 1500.;
        problem = new ConstrainedLASSOProblem(diabetes_y, diabetes_X, t);
        betaHat = new ConstrainedLASSObyQP(problem).beta().betaHat();
        System.out.println("beta^ = " + betaHat);

        // relaxing the constraint; essentially the same as the OLS solution
        t = 10000.;
        problem = new ConstrainedLASSOProblem(diabetes_y, diabetes_X, t);
        betaHat = new ConstrainedLASSObyQP(problem).beta().betaHat();
        System.out.println("beta^ = " + betaHat);
    }

    public void backward_elimination() throws IOException {
        System.out.println("backward elimination on GLM regression");

        // read the birth weight data from a csv file
        double[][] birthwt
                = DoubleUtils.readCSV2d(
                        this.getClass().getClassLoader().getResourceAsStream("birthwt.csv"),
                        true,
                        true
                );

        // convert the csv file into a matrix for manipulation
        Matrix A = new DenseMatrix(birthwt);
        // the independent variable, y
        Vector y = A.getColumn(1);
        // the design matrix of dependent variable, X
        Matrix X = MatrixFactory.columns(A, 2, A.nCols() - 1); // ignore last column

        // construct a linear model problem
        GLMProblem problem = new GLMProblem(
                y, // responses
                X, // explanatory variables
                true, // with intercept
                new GLMFamily(new GLMBinomial()) // GLM with binomial distribution
        );

        BackwardElimination backwardElimination
                = new BackwardElimination(problem, new EliminationByAIC());

        System.out.println("elimination sequence:");
        System.out.println(Arrays.toString(backwardElimination.getFlags()));
    }

    public void forward_selection() throws IOException {
        System.out.println("forward selection on GLM regression");

        // read the birth weight data from a csv file
        double[][] birthwt
                = DoubleUtils.readCSV2d(
                        this.getClass().getClassLoader().getResourceAsStream("birthwt.csv"),
                        true,
                        true
                );

        // convert the csv file into a matrix for manipulation
        Matrix A = new DenseMatrix(birthwt);
        // the independent variable, y
        Vector y = A.getColumn(1);
        // the design matrix of dependent variable, X
        Matrix X = MatrixFactory.columns(A, 2, A.nCols() - 1); // ignore last column

        // construct a linear model problem
        GLMProblem problem = new GLMProblem(
                y, // responses
                X, // explanatory variables
                true, // with intercept
                new GLMFamily(new GLMBinomial()) // GLM with binomial distribution
        );

        // run a step-wise forward selection on covariates
        ForwardSelection forwardSelection
                = new ForwardSelection(problem, new SelectionByAIC());

        System.out.println("selection sequence:");
        System.out.println(Arrays.toString(forwardSelection.getFlags()));
    }

    public void glm_gaussian_inverse() throws Exception {
        System.out.println("GLM regression using the quasi-Gaussian distribution and inverse link function");

        // construct a linear model problem
        QuasiGLMProblem problem = new QuasiGLMProblem(
                // the independent variable, y
                new DenseVector(new double[]{1, 1, 0, 1, 1}),
                // the design matrix of dependent variable, X
                new DenseMatrix(
                        new double[][]{
                            {1.52},
                            {3.22},
                            {4.32},
                            {10.1034},
                            {12.1}
                        }),
                // with intercept
                true,
                new QuasiFamily(
                        new QuasiGaussian(), // the quasi-normal distribution
                        new LinkInverse() // inverse link function
                ));

        // solve a GLM regression problem
        GeneralizedLinearModelQuasiFamily glm
                = new GeneralizedLinearModelQuasiFamily(problem);

        System.out.println("beta hat");
        // the means of betas
        System.out.println("beta^ = " + glm.beta().betaHat());
        // the standard errors of betas
        System.out.println("beta^ standard error = " + glm.beta().stderr());
        // a beta/variable is significant if its t-stat is bigger than 2
        System.out.println("beta^ t = " + glm.beta().t());

        System.out.println("\nresiduals");
        System.out.println("fitted values: " + glm.residuals().fitted());
        System.out.println("deviance residuals: " + glm.residuals().devianceResiduals());
        System.out.println("deviance: " + glm.residuals().deviance());
        System.out.println("over dispersion: " + glm.residuals().overdispersion());
    }

    public void glm_quasibinomial_logit() throws Exception {
        System.out.println("GLM regression using the quasi-binomial distribution and logit link function");

        // construct a linear model problem
        QuasiGLMProblem problem = new QuasiGLMProblem(
                // the independent variable, y
                new DenseVector(new double[]{1, 1, 0, 1, 1}),
                // the design matrix of dependent variable, X
                new DenseMatrix(
                        new double[][]{
                            {1.52},
                            {3.22},
                            {4.32},
                            {10.1034},
                            {12.1}
                        }),
                // with intercept
                true,
                new QuasiFamily(
                        new QuasiBinomial(), // the quasi-binomial distribution
                        new LinkLogit() // logit link function
                ));

        // solve a GLM regression problem
        GeneralizedLinearModelQuasiFamily glm
                = new GeneralizedLinearModelQuasiFamily(problem);

        System.out.println("beta hat");
        // the means of betas
        System.out.println("beta^ = " + glm.beta().betaHat());
        // the standard errors of betas
        System.out.println("beta^ standard error = " + glm.beta().stderr());
        // a beta/variable is significant if its t-stat is bigger than 2
        System.out.println("beta^ t = " + glm.beta().t());

        System.out.println("\nresiduals");
        System.out.println("fitted values: " + glm.residuals().fitted());
        System.out.println("deviance residuals: " + glm.residuals().devianceResiduals());
        System.out.println("deviance: " + glm.residuals().deviance());
        System.out.println("over dispersion: " + glm.residuals().overdispersion());
    }

    public void glm_poisson_log() throws Exception {
        System.out.println("GLM regression using the Poisson distribution and log link function");

        // construct a linear model problem
        GLMProblem problem = new GLMProblem(
                // the independent variable, y
                new DenseVector(new double[]{4, 1, 4, 5, 7}),
                // the design matrix of dependent variable, X
                new DenseMatrix(
                        new double[][]{
                            {1.52, 2.11},
                            {3.22, 4.32},
                            {4.32, 1.23},
                            {10.1034, 8.43},
                            {12.1, 7.31}
                        }),
                // with intercept
                true,
                // use the binomial distribution
                new GLMFamily(new GLMPoisson()));

        // solve a GLM regression problem
        GeneralizedLinearModel glm = new GeneralizedLinearModel(problem);

        System.out.println("beta hat");
        // the means of betas
        System.out.println("beta^ = " + glm.beta().betaHat());
        // the standard errors of betas
        System.out.println("beta^ standard error = " + glm.beta().stderr());
        // a beta/variable is significant if its t-stat is bigger than 2
        System.out.println("beta^ t = " + glm.beta().t());

        System.out.println("\nresiduals");
        System.out.println("fitted values: " + glm.residuals().fitted());
        System.out.println("deviance residuals: " + glm.residuals().devianceResiduals());
        System.out.println("deviance: " + glm.residuals().deviance());
        System.out.println("over dispersion: " + glm.residuals().overdispersion());

        System.out.println("\ninformation criteria");
        System.out.println("AIC = " + glm.AIC());
    }

    public void glm_binomial_logit() throws Exception {
        System.out.println("GLM regression using the binomial distribution and logit link function");

        // construct a linear model problem
        GLMProblem problem = new GLMProblem(
                // the independent variable, y
                new DenseVector(new double[]{1, 1, 0, 1, 1}),
                // the design matrix of dependent variable, X
                new DenseMatrix(
                        new double[][]{
                            {1.52},
                            {3.22},
                            {4.32},
                            {10.1034},
                            {12.1}
                        }),
                // with intercept
                true,
                // use the binomial distribution
                new GLMFamily(new GLMBinomial()));

        // solve a GLM regression problem
        GeneralizedLinearModel glm = new GeneralizedLinearModel(problem);

        System.out.println("beta hat");
        // the means of betas
        System.out.println("beta^ = " + glm.beta().betaHat());
        // the standard errors of betas
        System.out.println("beta^ standard error = " + glm.beta().stderr());
        // a beta/variable is significant if its t-stat is bigger than 2
        System.out.println("beta^ t = " + glm.beta().t());

        System.out.println("\nresiduals");
        System.out.println("fitted values: " + glm.residuals().fitted());
        System.out.println("deviance residuals: " + glm.residuals().devianceResiduals());
        System.out.println("deviance: " + glm.residuals().deviance());
        System.out.println("over dispersion: " + glm.residuals().overdispersion());

        System.out.println("\ninformation criteria");
        System.out.println("AIC = " + glm.AIC());
    }

    public void logistic() throws Exception {
        System.out.println("logistic regression");

        // construct a linear model problem
        LMProblem problem = new LMProblem(
                // the independent variable, y, {pass, fail}
                new DenseVector(new double[]{0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1}),
                // the design matrix of dependent variable, X, number of hours of study
                new DenseMatrix(
                        new double[][]{
                            {0.5},
                            {0.75},
                            {1.},
                            {1.25},
                            {1.5},
                            {1.75},
                            {1.75},
                            {2.},
                            {2.25},
                            {2.5},
                            {2.75},
                            {3.},
                            {3.25},
                            {3.5},
                            {4.},
                            {4.25},
                            {4.5},
                            {4.75},
                            {5.},
                            {5.5}
                        }),
                // with intercept
                true);

        // solve a logistic regression problem
        LogisticRegression logistic = new LogisticRegression(problem);

        System.out.println("beta hat");
        // the means of betas
        System.out.println("beta^ = " + logistic.beta().betaHat());
        // the standard errors of betas
        System.out.println("beta^ standard error = " + logistic.beta().stderr());
        // a beta/variable is significant if its t-stat is bigger than 2
        System.out.println("beta^ t = " + logistic.beta().t());

        System.out.println("\nresiduals");
        System.out.println("fitted values: " + logistic.residuals().fitted());
        System.out.println("deviance residuals: " + logistic.residuals().devianceResiduals());
        System.out.println("deviance: " + logistic.residuals().deviance());
        System.out.println("null deviance: " + logistic.residuals().nullDeviance());

        System.out.println("\ninformation criteria");
        System.out.println("AIC = " + logistic.AIC());
    }

    public void weightedOLS() throws Exception {
        System.out.println("weighted ordinary least squares");

        // construct a linear model problem
        LMProblem problem = new LMProblem(
                // the independent variable, y
                new DenseVector(new double[]{2.32, 0.452, 4.53, 12.34, 32.2}),
                // the design matrix of dependent variable, X
                new DenseMatrix(
                        new double[][]{
                            {1.52, 2.23, 4.31},
                            {3.22, 6.34, 3.46},
                            {4.32, 12.2, 23.1},
                            {10.1034, 43.2, 22.3},
                            {12.1, 2.12, 3.27}
                        }),
                // with intercept
                true,
                // the weights assigned to each observation
                new DenseVector(new double[]{0.2, 0.4, 0.1, 0.3, 0.1})); // do not sum to 1

        // solve a weighted OLS problem
        OLSRegression ols = new OLSRegression(problem);

        System.out.println("beta hat");
        // the means of betas
        System.out.println("beta^ = " + ols.beta().betaHat());
        // the standard errors of betas
        System.out.println("beta^ standard error = " + ols.beta().stderr());
        // a beta/variable is significant if its t-stat is bigger than 2
        System.out.println("beta^ t = " + ols.beta().t());

        System.out.println("\nresiduals");
        System.out.println("residual F-stat = " + ols.residuals().Fstat());
        System.out.println("fitted values: " + ols.residuals().fitted());
        System.out.println("weighted residuals: " + ols.residuals().weightedResiduals());
        System.out.println("residuals: " + ols.residuals().residuals());
        System.out.println("residual standard error = " + ols.residuals().stderr());
        System.out.println("RSS = " + ols.residuals().RSS());
        System.out.println("TSS = " + ols.residuals().TSS());
        System.out.println("R2 = " + ols.residuals().R2());
        System.out.println("AR2 = " + ols.residuals().AR2());

        System.out.println("\ninfluential points");
        System.out.println("standarized residuals: " + ols.residuals().standardized());
        System.out.println("studentized residuals: " + ols.residuals().studentized());
        System.out.println("leverage/hat values: " + ols.residuals().leverage());
        System.out.println("DFFITS: " + ols.diagnostics().DFFITS());
        System.out.println("cook distances: " + ols.diagnostics().cookDistances());
        System.out.println("Hadi: " + ols.diagnostics().Hadi());

        System.out.println("\ninformation criteria");
        System.out.println("AIC = " + ols.informationCriteria().AIC());
        System.out.println("BIC = " + ols.informationCriteria().BIC());
    }

    public void ols() throws Exception {
        System.out.println("ordinary least squares");

        // construct a linear model problem
        LMProblem problem = new LMProblem(
                // the independent variable, y
                new DenseVector(new double[]{2.32, 0.452, 4.53, 12.34, 32.2}),
                // the design matrix of dependent variable, X
                new DenseMatrix(
                        new double[][]{
                            {1.52, 2.23, 4.31},
                            {3.22, 6.34, 3.46},
                            {4.32, 12.2, 23.1},
                            {10.1034, 43.2, 22.3},
                            {12.1, 2.12, 3.27}
                        }),
                true // with intercept
        );

        // solve an OLS problem
        OLSRegression ols = new OLSRegression(problem);

        System.out.println("beta hat");
        // the means of betas
        System.out.println("beta^ = " + ols.beta().betaHat());
        // the standard errors of betas
        System.out.println("beta^ standard error = " + ols.beta().stderr());
        // a beta/variable is significant if its t-stat is bigger than 2
        System.out.println("beta^ t = " + ols.beta().t());

        System.out.println("\nresiduals");
        System.out.println("residual F-stat = " + ols.residuals().Fstat());
        System.out.println("fitted values: " + ols.residuals().fitted());
        System.out.println("residuals: " + ols.residuals().residuals());
        System.out.println("residual standard error = " + ols.residuals().stderr());
        System.out.println("RSS = " + ols.residuals().RSS());
        System.out.println("TSS = " + ols.residuals().TSS());
        System.out.println("R2 = " + ols.residuals().R2());
        System.out.println("AR2 = " + ols.residuals().AR2());

        System.out.println("\ninfluential points");
        System.out.println("standarized residuals: " + ols.residuals().standardized());
        System.out.println("studentized residuals: " + ols.residuals().studentized());
        System.out.println("leverage/hat values: " + ols.residuals().leverage());
        System.out.println("DFFITS: " + ols.diagnostics().DFFITS());
        System.out.println("cook distances: " + ols.diagnostics().cookDistances());
        System.out.println("Hadi: " + ols.diagnostics().Hadi());

        System.out.println("\ninformation criteria");
        System.out.println("AIC = " + ols.informationCriteria().AIC());
        System.out.println("BIC = " + ols.informationCriteria().BIC());
    }

    /**
     * diabetes_X is already demeaned and scaled but not normalized to 1.
     */
    public static Matrix diabetes_X = new DenseMatrix(
            new double[][]{{0.03807591, 0.05068012, 0.06169621, 0.02187235, -0.0442235, -0.03482076, -0.04340085, -0.002592262, 0.01990842, -0.01764613},
            {-0.001882017, -0.04464164, -0.05147406, -0.02632783, -0.008448724, -0.01916334, 0.07441156, -0.03949338, -0.06832974, -0.09220405},
            {0.0852989, 0.05068012, 0.04445121, -0.00567061, -0.04559945, -0.03419447, -0.03235593, -0.002592262, 0.002863771, -0.02593034},
            {-0.08906294, -0.04464164, -0.01159501, -0.03665645, 0.01219057, 0.02499059, -0.03603757, 0.03430886, 0.02269202, -0.009361911},
            {0.00538306, -0.04464164, -0.03638469, 0.02187235, 0.003934852, 0.01559614, 0.008142084, -0.002592262, -0.03199144, -0.04664087},
            {-0.09269548, -0.04464164, -0.04069594, -0.01944209, -0.06899065, -0.07928784, 0.04127682, -0.0763945, -0.04118039, -0.09634616},
            {-0.04547248, 0.05068012, -0.04716281, -0.01599922, -0.04009564, -0.02480001, 0.000778808, -0.03949338, -0.06291295, -0.03835666},
            {0.06350368, 0.05068012, -0.001894706, 0.06662967, 0.09061988, 0.1089144, 0.02286863, 0.01770335, -0.03581673, 0.003064409},
            {0.04170844, 0.05068012, 0.06169621, -0.04009932, -0.01395254, 0.006201686, -0.02867429, -0.002592262, -0.01495648, 0.01134862},
            {-0.07090025, -0.04464164, 0.03906215, -0.03321358, -0.01257658, -0.03450761, -0.02499266, -0.002592262, 0.06773633, -0.01350402},
            {-0.09632802, -0.04464164, -0.08380842, 0.008100872, -0.1033895, -0.09056119, -0.01394774, -0.0763945, -0.06291295, -0.03421455},
            {0.02717829, 0.05068012, 0.01750591, -0.03321358, -0.007072771, 0.04597154, -0.06549067, 0.07120998, -0.09643322, -0.0590672},
            {0.01628068, -0.04464164, -0.02884001, -0.009113481, -0.004320866, -0.009768886, 0.04495846, -0.03949338, -0.03075121, -0.04249877},
            {0.00538306, 0.05068012, -0.001894706, 0.008100872, -0.004320866, -0.01571871, -0.00290283, -0.002592262, 0.03839325, -0.01350402},
            {0.04534098, -0.04464164, -0.02560657, -0.01255635, 0.01769438, -6.128358e-05, 0.08177484, -0.03949338, -0.03199144, -0.07563562},
            {-0.05273755, 0.05068012, -0.01806189, 0.08040116, 0.08924393, 0.1076618, -0.03971921, 0.1081111, 0.03605579, -0.04249877},
            {-0.005514555, -0.04464164, 0.04229559, 0.04941532, 0.02457414, -0.02386057, 0.07441156, -0.03949338, 0.05228, 0.02791705},
            {0.07076875, 0.05068012, 0.01211685, 0.05630106, 0.03420581, 0.04941617, -0.03971921, 0.03430886, 0.02736771, -0.001077698},
            {-0.0382074, -0.04464164, -0.01051720, -0.03665645, -0.03734373, -0.01947649, -0.02867429, -0.002592262, -0.01811827, -0.01764613},
            {-0.02730979, -0.04464164, -0.01806189, -0.04009932, -0.002944913, -0.01133463, 0.03759519, -0.03949338, -0.008944019, -0.05492509},
            {-0.04910502, -0.04464164, -0.05686312, -0.04354219, -0.04559945, -0.04327577, 0.000778808, -0.03949338, -0.01190068, 0.01549073},
            {-0.0854304, 0.05068012, -0.02237314, 0.001215131, -0.03734373, -0.02636575, 0.01550536, -0.03949338, -0.07212845, -0.01764613},
            {-0.0854304, -0.04464164, -0.00405033, -0.009113481, -0.002944913, 0.007767428, 0.02286863, -0.03949338, -0.0611766, -0.01350402},
            {0.04534098, 0.05068012, 0.0606184, 0.03105334, 0.02870200, -0.0473467, -0.05444576, 0.07120998, 0.1335990, 0.1356118},
            {-0.06363517, -0.04464164, 0.03582872, -0.02288496, -0.03046397, -0.01885019, -0.006584468, -0.002592262, -0.02595242, -0.05492509},
            {-0.06726771, 0.05068012, -0.01267283, -0.04009932, -0.01532849, 0.004635943, -0.0581274, 0.03430886, 0.01919903, -0.03421455},
            {-0.1072256, -0.04464164, -0.07734155, -0.02632783, -0.08962994, -0.09619786, 0.02655027, -0.0763945, -0.04257210, -0.005219804},
            {-0.02367725, -0.04464164, 0.05954058, -0.04009932, -0.04284755, -0.04358892, 0.01182372, -0.03949338, -0.01599827, 0.04034337},
            {0.05260606, -0.04464164, -0.02129532, -0.07452802, -0.04009564, -0.0376391, -0.006584468, -0.03949338, -0.0006092542, -0.05492509},
            {0.06713621, 0.05068012, -0.006205954, 0.0631868, -0.04284755, -0.09588471, 0.05232174, -0.0763945, 0.0594238, 0.05276969},
            {-0.06000263, -0.04464164, 0.04445121, -0.01944209, -0.009824677, -0.007576847, 0.02286863, -0.03949338, -0.02712865, -0.009361911},
            {-0.02367725, -0.04464164, -0.06548562, -0.08141377, -0.03871969, -0.05360967, 0.05968501, -0.0763945, -0.03712835, -0.04249877},
            {0.03444337, 0.05068012, 0.1252871, 0.02875810, -0.05385517, -0.01290037, -0.1023071, 0.1081111, 0.0002714857, 0.02791705},
            {0.03081083, -0.04464164, -0.05039625, -0.00222774, -0.0442235, -0.0899349, 0.1185912, -0.0763945, -0.01811827, 0.003064409},
            {0.01628068, -0.04464164, -0.06333, -0.05731367, -0.05798303, -0.04891244, 0.008142084, -0.03949338, -0.0594727, -0.06735141},
            {0.04897352, 0.05068012, -0.03099563, -0.04928031, 0.0493413, -0.004132214, 0.1333178, -0.05351581, 0.02131085, 0.01963284},
            {0.01264814, -0.04464164, 0.02289497, 0.05285819, 0.00806271, -0.02855779, 0.03759519, -0.03949338, 0.054724, -0.02593034},
            {-0.009147093, -0.04464164, 0.01103904, -0.05731367, -0.02496016, -0.04296262, 0.03023191, -0.03949338, 0.01703713, -0.005219804},
            {-0.001882017, 0.05068012, 0.07139652, 0.09761551, 0.08786798, 0.0754075, -0.02131102, 0.07120998, 0.07142403, 0.02377494},
            {-0.001882017, 0.05068012, 0.01427248, -0.07452802, 0.002558899, 0.006201686, -0.01394774, -0.002592262, 0.01919903, 0.003064409},
            {0.00538306, 0.05068012, -0.008361578, 0.02187235, 0.05484511, 0.07321546, -0.02499266, 0.03430886, 0.01255315, 0.09419076},
            {-0.09996055, -0.04464164, -0.06764124, -0.1089567, -0.07449446, -0.07271173, 0.01550536, -0.03949338, -0.04986847, -0.009361911},
            {-0.06000263, 0.05068012, -0.01051720, -0.0148516, -0.04972731, -0.02354742, -0.0581274, 0.0158583, -0.009918957, -0.03421455},
            {0.01991321, -0.04464164, -0.02345095, -0.07108515, 0.02044629, -0.01008203, 0.1185912, -0.0763945, -0.04257210, 0.07348023},
            {0.04534098, 0.05068012, 0.06816308, 0.008100872, -0.01670444, 0.004635943, -0.07653559, 0.07120998, 0.03243323, -0.01764613},
            {0.02717829, 0.05068012, -0.03530688, 0.03220097, -0.01120063, 0.001504459, -0.01026611, -0.002592262, -0.01495648, -0.05078298},
            {-0.05637009, -0.04464164, -0.01159501, -0.03321358, -0.0469754, -0.04765985, 0.004460446, -0.03949338, -0.007979398, -0.08806194},
            {-0.07816532, -0.04464164, -0.0730303, -0.05731367, -0.08412613, -0.07427747, -0.02499266, -0.03949338, -0.01811827, -0.08391984},
            {0.06713621, 0.05068012, -0.04177375, 0.01154374, 0.002558899, 0.005888537, 0.04127682, -0.03949338, -0.0594727, -0.02178823},
            {-0.04183994, 0.05068012, 0.01427248, -0.00567061, -0.01257658, 0.006201686, -0.07285395, 0.07120998, 0.03546194, -0.01350402},
            {0.03444337, -0.04464164, -0.007283766, 0.01498661, -0.0442235, -0.03732595, -0.00290283, -0.03949338, -0.02139368, 0.007206516},
            {0.05987114, 0.05068012, 0.0164281, 0.02875810, -0.04147159, -0.02918409, -0.02867429, -0.002592262, -0.002396681, -0.02178823},
            {-0.05273755, -0.04464164, -0.00943939, -0.00567061, 0.03970963, 0.04471895, 0.02655027, -0.002592262, -0.01811827, -0.01350402},
            {-0.009147093, -0.04464164, -0.01590626, 0.07007254, 0.01219057, 0.02217226, 0.01550536, -0.002592262, -0.03324879, 0.04862759},
            {-0.04910502, -0.04464164, 0.02505060, 0.008100872, 0.02044629, 0.01778818, 0.05232174, -0.03949338, -0.04118039, 0.007206516},
            {-0.04183994, -0.04464164, -0.04931844, -0.03665645, -0.007072771, -0.02260797, 0.08545648, -0.03949338, -0.06648815, 0.007206516},
            {-0.04183994, -0.04464164, 0.04121778, -0.02632783, -0.03183992, -0.03043668, -0.03603757, 0.002942906, 0.03365681, -0.01764613},
            {-0.02730979, -0.04464164, -0.06333, -0.05042793, -0.08962994, -0.1043397, 0.05232174, -0.0763945, -0.05615757, -0.06735141},
            {0.04170844, -0.04464164, -0.0644078, 0.03564384, 0.01219057, -0.05799375, 0.1811791, -0.0763945, -0.0006092542, -0.05078298},
            {0.06350368, 0.05068012, -0.02560657, 0.01154374, 0.06447678, 0.04847673, 0.03023191, -0.002592262, 0.03839325, 0.01963284},
            {-0.07090025, -0.04464164, -0.00405033, -0.04009932, -0.06623874, -0.07866155, 0.05232174, -0.0763945, -0.05140054, -0.03421455},
            {-0.04183994, 0.05068012, 0.004572167, -0.0538708, -0.0442235, -0.0273052, -0.08021722, 0.07120998, 0.0366458, 0.01963284},
            {-0.02730979, 0.05068012, -0.007283766, -0.04009932, -0.01120063, -0.01383982, 0.05968501, -0.03949338, -0.08238148, -0.02593034},
            {-0.03457486, -0.04464164, -0.03746250, -0.06075654, 0.02044629, 0.04346635, -0.01394774, -0.002592262, -0.03075121, -0.07149352},
            {0.06713621, 0.05068012, -0.02560657, -0.04009932, -0.06348684, -0.05987264, -0.00290283, -0.03949338, -0.01919705, 0.01134862},
            {-0.04547248, 0.05068012, -0.02452876, 0.05974393, 0.005310804, 0.01496984, -0.05444576, 0.07120998, 0.04234490, 0.01549073},
            {-0.009147093, 0.05068012, -0.01806189, -0.03321358, -0.0208323, 0.01215151, -0.07285395, 0.07120998, 0.0002714857, 0.01963284},
            {0.04170844, 0.05068012, -0.01482845, -0.01714685, -0.005696818, 0.008393725, -0.01394774, -0.001854240, -0.01190068, 0.003064409},
            {0.03807591, 0.05068012, -0.02991782, -0.04009932, -0.03321588, -0.02417372, -0.01026611, -0.002592262, -0.01290794, 0.003064409},
            {0.01628068, -0.04464164, -0.046085, -0.00567061, -0.07587041, -0.06143838, -0.01394774, -0.03949338, -0.05140054, 0.01963284},
            {-0.001882017, -0.04464164, -0.06979687, -0.01255635, -0.0001930070, -0.009142589, 0.07072993, -0.03949338, -0.06291295, 0.04034337},
            {-0.001882017, -0.04464164, 0.03367309, 0.1251585, 0.02457414, 0.02624319, -0.01026611, -0.002592262, 0.02671426, 0.06105391},
            {0.06350368, 0.05068012, -0.00405033, -0.01255635, 0.1030035, 0.04878988, 0.05600338, -0.002592262, 0.08449528, -0.01764613},
            {0.01264814, 0.05068012, -0.02021751, -0.00222774, 0.03833367, 0.05317395, -0.006584468, 0.03430886, -0.005145308, -0.009361911},
            {0.01264814, 0.05068012, 0.002416542, 0.05630106, 0.02732605, 0.01716188, 0.04127682, -0.03949338, 0.003711738, 0.07348023},
            {-0.009147093, 0.05068012, -0.03099563, -0.02632783, -0.01120063, -0.001000729, -0.02131102, -0.002592262, 0.006209316, 0.02791705},
            {-0.03094232, 0.05068012, 0.02828403, 0.07007254, -0.1267807, -0.1068449, -0.05444576, -0.04798064, -0.03075121, 0.01549073},
            {-0.09632802, -0.04464164, -0.03638469, -0.07452802, -0.03871969, -0.02761835, 0.01550536, -0.03949338, -0.07408887, -0.001077698},
            {0.00538306, -0.04464164, -0.05794093, -0.02288496, -0.0676147, -0.06832765, -0.05444576, -0.002592262, 0.04289569, -0.08391984},
            {-0.1035931, -0.04464164, -0.03746250, -0.02632783, 0.002558899, 0.01998022, 0.01182372, -0.002592262, -0.06832974, -0.02593034},
            {0.07076875, -0.04464164, 0.01211685, 0.04252958, 0.07135654, 0.0534871, 0.05232174, -0.002592262, 0.02539313, -0.005219804},
            {0.01264814, 0.05068012, -0.02237314, -0.02977071, 0.01081462, 0.02843523, -0.02131102, 0.03430886, -0.006080248, -0.001077698},
            {-0.01641217, -0.04464164, -0.03530688, -0.02632783, 0.03282986, 0.01716188, 0.1001830, -0.03949338, -0.07020931, -0.07977773},
            {-0.0382074, -0.04464164, 0.009961227, -0.04698506, -0.05935898, -0.05298337, -0.01026611, -0.03949338, -0.01599827, -0.04249877},
            {0.001750522, -0.04464164, -0.03961813, -0.1009234, -0.02908802, -0.03012354, 0.04495846, -0.05019471, -0.06832974, -0.129483},
            {0.04534098, -0.04464164, 0.07139652, 0.001215131, -0.009824677, -0.001000729, 0.01550536, -0.03949338, -0.04118039, -0.07149352},
            {-0.07090025, 0.05068012, -0.07518593, -0.04009932, -0.05110326, -0.01509241, -0.03971921, -0.002592262, -0.09643322, -0.03421455},
            {0.04534098, -0.04464164, -0.006205954, 0.01154374, 0.06310082, 0.01622244, 0.0965014, -0.03949338, 0.04289569, -0.03835666},
            {-0.05273755, 0.05068012, -0.04069594, -0.06764228, -0.03183992, -0.0370128, 0.03759519, -0.03949338, -0.03452372, 0.06933812},
            {-0.04547248, -0.04464164, -0.04824063, -0.01944209, -0.0001930070, -0.01603186, 0.06704829, -0.03949338, -0.02479119, 0.01963284},
            {0.01264814, -0.04464164, -0.02560657, -0.04009932, -0.03046397, -0.04515466, 0.0780932, -0.0763945, -0.07212845, 0.01134862},
            {0.04534098, -0.04464164, 0.0519959, -0.0538708, 0.06310082, 0.06476045, -0.01026611, 0.03430886, 0.03723201, 0.01963284},
            {-0.02004471, -0.04464164, 0.004572167, 0.09761551, 0.005310804, -0.02072908, 0.06336665, -0.03949338, 0.01255315, 0.01134862},
            {-0.04910502, -0.04464164, -0.0644078, -0.102071, -0.002944913, -0.01540556, 0.06336665, -0.04724262, -0.03324879, -0.05492509},
            {-0.07816532, -0.04464164, -0.01698407, -0.01255635, -0.0001930070, -0.01352667, 0.07072993, -0.03949338, -0.04118039, -0.09220405},
            {-0.07090025, -0.04464164, -0.05794093, -0.08141377, -0.04559945, -0.02887094, -0.04340085, -0.002592262, 0.001143797, -0.005219804},
            {0.0562386, 0.05068012, 0.009961227, 0.04941532, -0.004320866, -0.01227407, -0.04340085, 0.03430886, 0.06078775, 0.03205916},
            {-0.02730979, -0.04464164, 0.0886415, -0.02518021, 0.02182224, 0.04252691, -0.03235593, 0.03430886, 0.002863771, 0.07762233},
            {0.001750522, 0.05068012, -0.005128142, -0.01255635, -0.01532849, -0.01383982, 0.008142084, -0.03949338, -0.006080248, -0.06735141},
            {-0.001882017, -0.04464164, -0.0644078, 0.01154374, 0.02732605, 0.03751653, -0.01394774, 0.03430886, 0.0117839, -0.05492509},
            {0.01628068, -0.04464164, 0.01750591, -0.02288496, 0.06034892, 0.0444058, 0.03023191, -0.002592262, 0.03723201, -0.001077698},
            {0.01628068, 0.05068012, -0.04500719, 0.0631868, 0.01081462, -0.0003744320, 0.06336665, -0.03949338, -0.03075121, 0.03620126},
            {-0.09269548, -0.04464164, 0.02828403, -0.01599922, 0.03695772, 0.02499059, 0.05600338, -0.03949338, -0.005145308, -0.001077698},
            {0.05987114, 0.05068012, 0.04121778, 0.01154374, 0.04108558, 0.07071027, -0.03603757, 0.03430886, -0.01090444, -0.03007245},
            {-0.02730979, -0.04464164, 0.06492964, -0.00222774, -0.02496016, -0.01728445, 0.02286863, -0.03949338, -0.0611766, -0.0632093},
            {0.02354575, 0.05068012, -0.03207344, -0.04009932, -0.03183992, -0.02166853, -0.01394774, -0.002592262, -0.01090444, 0.01963284},
            {-0.09632802, -0.04464164, -0.07626374, -0.04354219, -0.04559945, -0.03482076, 0.008142084, -0.03949338, -0.0594727, -0.08391984},
            {0.02717829, -0.04464164, 0.04984027, -0.05501842, -0.002944913, 0.04064802, -0.0581274, 0.05275942, -0.05295879, -0.005219804},
            {0.01991321, 0.05068012, 0.04552903, 0.02990572, -0.06211089, -0.05580171, -0.07285395, 0.02692863, 0.04560081, 0.04034337},
            {0.03807591, 0.05068012, -0.00943939, 0.002362754, 0.001182946, 0.03751653, -0.05444576, 0.05017634, -0.02595242, 0.1066171},
            {0.04170844, 0.05068012, -0.03207344, -0.02288496, -0.04972731, -0.04014429, 0.03023191, -0.03949338, -0.1260974, 0.01549073},
            {0.01991321, -0.04464164, 0.004572167, -0.02632783, 0.02319819, 0.01027262, 0.06704829, -0.03949338, -0.02364456, -0.04664087},
            {-0.0854304, -0.04464164, 0.02073935, -0.02632783, 0.005310804, 0.01966707, -0.00290283, -0.002592262, -0.02364456, 0.003064409},
            {0.01991321, 0.05068012, 0.01427248, 0.0631868, 0.01494247, 0.02029337, -0.04708248, 0.03430886, 0.04666077, 0.09004865},
            {0.02354575, -0.04464164, 0.1101977, 0.0631868, 0.01356652, -0.03294187, -0.02499266, 0.02065544, 0.09924023, 0.02377494},
            {-0.03094232, 0.05068012, 0.001338730, -0.00567061, 0.06447678, 0.04941617, -0.04708248, 0.1081111, 0.08379677, 0.003064409},
            {0.04897352, 0.05068012, 0.05846277, 0.07007254, 0.01356652, 0.02060651, -0.02131102, 0.03430886, 0.02200405, 0.02791705},
            {0.05987114, -0.04464164, -0.02129532, 0.0872869, 0.04521344, 0.03156671, -0.04708248, 0.07120998, 0.07912108, 0.1356118},
            {-0.05637009, 0.05068012, -0.01051720, 0.02531523, 0.02319819, 0.04002172, -0.03971921, 0.03430886, 0.02061233, 0.0569118},
            {0.01628068, -0.04464164, -0.04716281, -0.00222774, -0.01945635, -0.04296262, 0.03391355, -0.03949338, 0.02736771, 0.02791705},
            {-0.04910502, -0.04464164, 0.004572167, 0.01154374, -0.03734373, -0.01853704, -0.01762938, -0.002592262, -0.03980959, -0.02178823},
            {0.06350368, -0.04464164, 0.01750591, 0.02187235, 0.00806271, 0.02154596, -0.03603757, 0.03430886, 0.01990842, 0.01134862},
            {0.04897352, 0.05068012, 0.08109682, 0.02187235, 0.04383748, 0.06413415, -0.05444576, 0.07120998, 0.03243323, 0.04862759},
            {0.00538306, 0.05068012, 0.03475090, -0.001080116, 0.1525378, 0.198788, -0.06180903, 0.1852344, 0.01556684, 0.07348023},
            {-0.005514555, -0.04464164, 0.02397278, 0.008100872, -0.03459183, -0.03889169, 0.02286863, -0.03949338, -0.01599827, -0.01350402},
            {-0.005514555, 0.05068012, -0.008361578, -0.00222774, -0.03321588, -0.06363042, -0.03603757, -0.002592262, 0.08058546, 0.007206516},
            {-0.08906294, -0.04464164, -0.06117437, -0.02632783, -0.05523112, -0.05454912, 0.04127682, -0.0763945, -0.09393565, -0.05492509},
            {0.03444337, 0.05068012, -0.001894706, -0.01255635, 0.03833367, 0.01371725, 0.0780932, -0.03949338, 0.004551890, -0.09634616},
            {-0.05273755, -0.04464164, -0.06225218, -0.02632783, -0.005696818, -0.005071659, 0.03023191, -0.03949338, -0.03075121, -0.07149352},
            {0.009015599, -0.04464164, 0.0164281, 0.004658002, 0.009438663, 0.01058576, -0.02867429, 0.03430886, 0.03896837, 0.1190434},
            {-0.06363517, 0.05068012, 0.0961862, 0.1045013, -0.002944913, -0.004758511, -0.006584468, -0.002592262, 0.02269202, 0.07348023},
            {-0.09632802, -0.04464164, -0.06979687, -0.06764228, -0.01945635, -0.01070833, 0.01550536, -0.03949338, -0.04687948, -0.07977773},
            {0.01628068, 0.05068012, -0.02129532, -0.009113481, 0.03420581, 0.04785043, 0.000778808, -0.002592262, -0.01290794, 0.02377494},
            {-0.04183994, 0.05068012, -0.05362969, -0.04009932, -0.08412613, -0.07177228, -0.00290283, -0.03949338, -0.07212845, -0.03007245},
            {-0.07453279, -0.04464164, 0.0433734, -0.03321358, 0.01219057, 0.0002518649, 0.06336665, -0.03949338, -0.02712865, -0.04664087},
            {-0.005514555, -0.04464164, 0.05630715, -0.03665645, -0.04835136, -0.04296262, -0.07285395, 0.03799897, 0.05078151, 0.0569118},
            {-0.09269548, -0.04464164, -0.0816528, -0.05731367, -0.06073493, -0.0680145, 0.0486401, -0.0763945, -0.06648815, -0.02178823},
            {0.00538306, -0.04464164, 0.04984027, 0.09761551, -0.01532849, -0.01634500, -0.006584468, -0.002592262, 0.01703713, -0.01350402},
            {0.03444337, 0.05068012, 0.1112756, 0.07695829, -0.03183992, -0.03388132, -0.02131102, -0.002592262, 0.02801651, 0.07348023},
            {0.02354575, -0.04464164, 0.06169621, 0.05285819, -0.03459183, -0.04891244, -0.02867429, -0.002592262, 0.054724, -0.005219804},
            {0.04170844, 0.05068012, 0.01427248, 0.04252958, -0.03046397, -0.001313877, -0.04340085, -0.002592262, -0.03324879, 0.01549073},
            {-0.02730979, -0.04464164, 0.04768465, -0.04698506, 0.03420581, 0.05724488, -0.08021722, 0.1302518, 0.04506617, 0.1314697},
            {0.04170844, 0.05068012, 0.01211685, 0.03908671, 0.05484511, 0.0444058, 0.004460446, -0.002592262, 0.04560081, -0.001077698},
            {-0.03094232, -0.04464164, 0.005649979, -0.009113481, 0.01907033, 0.006827983, 0.07441156, -0.03949338, -0.04118039, -0.04249877},
            {0.03081083, 0.05068012, 0.04660684, -0.01599922, 0.02044629, 0.05066877, -0.0581274, 0.07120998, 0.006209316, 0.007206516},
            {-0.04183994, -0.04464164, 0.1285206, 0.0631868, -0.03321588, -0.03262872, 0.01182372, -0.03949338, -0.01599827, -0.05078298},
            {-0.03094232, 0.05068012, 0.05954058, 0.001215131, 0.01219057, 0.03156671, -0.04340085, 0.03430886, 0.01482271, 0.007206516},
            {-0.05637009, -0.04464164, 0.09295276, -0.01944209, 0.01494247, 0.02342485, -0.02867429, 0.02545259, 0.02605609, 0.04034337},
            {-0.06000263, 0.05068012, 0.01535029, -0.01944209, 0.03695772, 0.04816358, 0.01918700, -0.002592262, -0.03075121, -0.001077698},
            {-0.04910502, 0.05068012, -0.005128142, -0.04698506, -0.0208323, -0.02041593, -0.06917231, 0.07120998, 0.06123791, -0.03835666},
            {0.02354575, -0.04464164, 0.0703187, 0.02531523, -0.03459183, -0.01446611, -0.03235593, -0.002592262, -0.01919705, -0.009361911},
            {0.001750522, -0.04464164, -0.00405033, -0.00567061, -0.008448724, -0.02386057, 0.05232174, -0.03949338, -0.008944019, -0.01350402},
            {-0.03457486, 0.05068012, -0.0008168938, 0.07007254, 0.03970963, 0.06695249, -0.06549067, 0.1081111, 0.02671426, 0.07348023},
            {0.04170844, 0.05068012, -0.04392938, 0.0631868, -0.004320866, 0.01622244, -0.01394774, -0.002592262, -0.03452372, 0.01134862},
            {0.06713621, 0.05068012, 0.02073935, -0.00567061, 0.02044629, 0.02624319, -0.00290283, -0.002592262, 0.008640283, 0.003064409},
            {-0.02730979, 0.05068012, 0.0606184, 0.04941532, 0.08511607, 0.0863677, -0.00290283, 0.03430886, 0.03781448, 0.04862759},
            {-0.01641217, -0.04464164, -0.01051720, 0.001215131, -0.03734373, -0.03576021, 0.01182372, -0.03949338, -0.02139368, -0.03421455},
            {-0.001882017, 0.05068012, -0.03315126, -0.01829447, 0.03145391, 0.04284006, -0.01394774, 0.01991742, 0.01022564, 0.02791705},
            {-0.01277963, -0.04464164, -0.06548562, -0.06993753, 0.001182946, 0.01684873, -0.00290283, -0.007020397, -0.03075121, -0.05078298},
            {-0.005514555, -0.04464164, 0.0433734, 0.0872869, 0.01356652, 0.007141131, -0.01394774, -0.002592262, 0.04234490, -0.01764613},
            {-0.009147093, -0.04464164, -0.06225218, -0.07452802, -0.02358421, -0.01321352, 0.004460446, -0.03949338, -0.03581673, -0.04664087},
            {-0.04547248, 0.05068012, 0.06385183, 0.07007254, 0.1332744, 0.1314611, -0.03971921, 0.1081111, 0.07573759, 0.08590655},
            {-0.05273755, -0.04464164, 0.03043966, -0.07452802, -0.02358421, -0.01133463, -0.00290283, -0.002592262, -0.03075121, -0.001077698},
            {0.01628068, 0.05068012, 0.07247433, 0.07695829, -0.008448724, 0.005575389, -0.006584468, -0.002592262, -0.02364456, 0.06105391},
            {0.04534098, -0.04464164, -0.0191397, 0.02187235, 0.02732605, -0.01352667, 0.1001830, -0.03949338, 0.01776348, -0.01350402},
            {-0.04183994, -0.04464164, -0.06656343, -0.04698506, -0.03734373, -0.04327577, 0.0486401, -0.03949338, -0.05615757, -0.01350402},
            {-0.05637009, 0.05068012, -0.06009656, -0.03665645, -0.08825399, -0.07083284, -0.01394774, -0.03949338, -0.07814091, -0.1046304},
            {0.07076875, -0.04464164, 0.06924089, 0.03793909, 0.02182224, 0.001504459, -0.03603757, 0.03910600, 0.07763279, 0.1066171},
            {0.001750522, 0.05068012, 0.05954058, -0.00222774, 0.06172487, 0.0631947, -0.0581274, 0.1081111, 0.06898221, 0.1273276},
            {-0.001882017, -0.04464164, -0.02668438, 0.04941532, 0.05897297, -0.01603186, -0.04708248, 0.07120998, 0.1335990, 0.01963284},
            {0.02354575, 0.05068012, -0.02021751, -0.03665645, -0.01395254, -0.01509241, 0.05968501, -0.03949338, -0.09643322, -0.01764613},
            {-0.02004471, -0.04464164, -0.046085, -0.09862812, -0.07587041, -0.05987264, -0.01762938, -0.03949338, -0.05140054, -0.04664087},
            {0.04170844, 0.05068012, 0.07139652, 0.008100872, 0.03833367, 0.01590929, -0.01762938, 0.03430886, 0.07341008, 0.08590655},
            {-0.06363517, 0.05068012, -0.07949718, -0.00567061, -0.07174256, -0.06644876, -0.01026611, -0.03949338, -0.01811827, -0.05492509},
            {0.01628068, 0.05068012, 0.009961227, -0.04354219, -0.0965097, -0.09463212, -0.03971921, -0.03949338, 0.01703713, 0.007206516},
            {0.06713621, -0.04464164, -0.03854032, -0.02632783, -0.03183992, -0.02636575, 0.008142084, -0.03949338, -0.02712865, 0.003064409},
            {0.04534098, 0.05068012, 0.01966154, 0.03908671, 0.02044629, 0.02593004, 0.008142084, -0.002592262, -0.003303713, 0.01963284},
            {0.04897352, -0.04464164, 0.02720622, -0.02518021, 0.02319819, 0.01841448, -0.06180903, 0.08006625, 0.07222365, 0.03205916},
            {0.04170844, -0.04464164, -0.008361578, -0.02632783, 0.02457414, 0.01622244, 0.07072993, -0.03949338, -0.04836172, -0.03007245},
            {-0.02367725, -0.04464164, -0.01590626, -0.01255635, 0.02044629, 0.04127431, -0.04340085, 0.03430886, 0.01407245, -0.009361911},
            {-0.0382074, 0.05068012, 0.004572167, 0.03564384, -0.01120063, 0.005888537, -0.04708248, 0.03430886, 0.01630495, -0.001077698},
            {0.04897352, -0.04464164, -0.04285156, -0.0538708, 0.04521344, 0.05004247, 0.03391355, -0.002592262, -0.02595242, -0.0632093},
            {0.04534098, 0.05068012, 0.005649979, 0.05630106, 0.06447678, 0.08918603, -0.03971921, 0.07120998, 0.01556684, -0.009361911},
            {0.04534098, 0.05068012, -0.03530688, 0.0631868, -0.004320866, -0.001627026, -0.01026611, -0.002592262, 0.01556684, 0.0569118},
            {0.01628068, -0.04464164, 0.02397278, -0.02288496, -0.02496016, -0.02605261, -0.03235593, -0.002592262, 0.03723201, 0.03205916},
            {-0.07453279, 0.05068012, -0.01806189, 0.008100872, -0.01945635, -0.02480001, -0.06549067, 0.03430886, 0.06731722, -0.01764613},
            {-0.08179786, 0.05068012, 0.04229559, -0.01944209, 0.03970963, 0.05755803, -0.06917231, 0.1081111, 0.04718617, -0.03835666},
            {-0.06726771, -0.04464164, -0.0547075, -0.02632783, -0.07587041, -0.08210618, 0.0486401, -0.0763945, -0.086829, -0.1046304},
            {0.00538306, -0.04464164, -0.002972518, 0.04941532, 0.07410845, 0.07071027, 0.04495846, -0.002592262, -0.001498587, -0.009361911},
            {-0.001882017, -0.04464164, -0.06656343, 0.001215131, -0.002944913, 0.003070201, 0.01182372, -0.002592262, -0.02028875, -0.02593034},
            {0.009015599, -0.04464164, -0.01267283, 0.02875810, -0.01808039, -0.005071659, -0.04708248, 0.03430886, 0.02337484, -0.005219804},
            {-0.005514555, 0.05068012, -0.04177375, -0.04354219, -0.07999827, -0.07615636, -0.03235593, -0.03949338, 0.01022564, -0.009361911},
            {0.0562386, 0.05068012, -0.03099563, 0.008100872, 0.01907033, 0.02123281, 0.03391355, -0.03949338, -0.02952762, -0.0590672},
            {0.009015599, 0.05068012, -0.005128142, -0.06419941, 0.06998059, 0.0838625, -0.03971921, 0.07120998, 0.03953988, 0.01963284},
            {-0.06726771, -0.04464164, -0.05901875, 0.03220097, -0.05110326, -0.04953874, -0.01026611, -0.03949338, 0.002007841, 0.02377494},
            {0.02717829, 0.05068012, 0.02505060, 0.01498661, 0.02595010, 0.04847673, -0.03971921, 0.03430886, 0.007837142, 0.02377494},
            {-0.02367725, -0.04464164, -0.046085, -0.03321358, 0.03282986, 0.03626394, 0.03759519, -0.002592262, -0.03324879, 0.01134862},
            {0.04897352, 0.05068012, 0.003494355, 0.07007254, -0.008448724, 0.0134041, -0.05444576, 0.03430886, 0.01331597, 0.03620126},
            {-0.05273755, -0.04464164, 0.05415152, -0.02632783, -0.05523112, -0.03388132, -0.01394774, -0.03949338, -0.07408887, -0.0590672},
            {0.04170844, -0.04464164, -0.04500719, 0.03449621, 0.04383748, -0.01571871, 0.03759519, -0.01440062, 0.0898987, 0.007206516},
            {0.0562386, -0.04464164, -0.05794093, -0.007965858, 0.0520932, 0.04910302, 0.05600338, -0.02141183, -0.02832024, 0.04448548},
            {-0.03457486, 0.05068012, -0.05578531, -0.01599922, -0.009824677, -0.007889995, 0.03759519, -0.03949338, -0.05295879, 0.02791705},
            {0.08166637, 0.05068012, 0.001338730, 0.03564384, 0.1263947, 0.09106492, 0.01918700, 0.03430886, 0.08449528, -0.03007245},
            {-0.001882017, 0.05068012, 0.03043966, 0.05285819, 0.03970963, 0.05661859, -0.03971921, 0.07120998, 0.02539313, 0.02791705},
            {0.1107267, 0.05068012, 0.006727791, 0.02875810, -0.02771206, -0.007263698, -0.04708248, 0.03430886, 0.002007841, 0.07762233},
            {-0.03094232, -0.04464164, 0.04660684, 0.01498661, -0.01670444, -0.04703355, 0.000778808, -0.002592262, 0.06345592, -0.02593034},
            {0.001750522, 0.05068012, 0.02612841, -0.009113481, 0.02457414, 0.03845598, -0.02131102, 0.03430886, 0.00943641, 0.003064409},
            {0.009015599, -0.04464164, 0.04552903, 0.02875810, 0.01219057, -0.01383982, 0.02655027, -0.03949338, 0.04613233, 0.03620126},
            {0.03081083, -0.04464164, 0.04013997, 0.07695829, 0.01769438, 0.03782968, -0.02867429, 0.03430886, -0.001498587, 0.1190434},
            {0.03807591, 0.05068012, -0.01806189, 0.06662967, -0.05110326, -0.01665815, -0.07653559, 0.03430886, -0.01190068, -0.01350402},
            {0.009015599, -0.04464164, 0.01427248, 0.01498661, 0.05484511, 0.04722413, 0.07072993, -0.03949338, -0.03324879, -0.0590672},
            {0.09256398, -0.04464164, 0.03690653, 0.02187235, -0.02496016, -0.01665815, 0.000778808, -0.03949338, -0.02251217, -0.02178823},
            {0.06713621, -0.04464164, 0.003494355, 0.03564384, 0.0493413, 0.03125356, 0.07072993, -0.03949338, -0.0006092542, 0.01963284},
            {0.001750522, -0.04464164, -0.07087468, -0.02288496, -0.001568960, -0.001000729, 0.02655027, -0.03949338, -0.02251217, 0.007206516},
            {0.03081083, -0.04464164, -0.03315126, -0.02288496, -0.0469754, -0.08116674, 0.1038647, -0.0763945, -0.03980959, -0.05492509},
            {0.02717829, 0.05068012, 0.09403057, 0.09761551, -0.03459183, -0.03200243, -0.04340085, -0.002592262, 0.0366458, 0.1066171},
            {0.01264814, 0.05068012, 0.03582872, 0.04941532, 0.05346915, 0.0741549, -0.06917231, 0.1450122, 0.04560081, 0.04862759},
            {0.07440129, -0.04464164, 0.03151747, 0.1010584, 0.04658939, 0.03689023, 0.01550536, -0.002592262, 0.03365681, 0.04448548},
            {-0.04183994, -0.04464164, -0.06548562, -0.04009932, -0.005696818, 0.01434355, -0.04340085, 0.03430886, 0.007026863, -0.01350402},
            {-0.08906294, -0.04464164, -0.04177375, -0.01944209, -0.06623874, -0.07427747, 0.008142084, -0.03949338, 0.001143797, -0.03007245},
            {0.02354575, 0.05068012, -0.03961813, -0.00567061, -0.04835136, -0.03325502, 0.01182372, -0.03949338, -0.1016435, -0.06735141},
            {-0.04547248, -0.04464164, -0.03854032, -0.02632783, -0.01532849, 0.0008781618, -0.03235593, -0.002592262, 0.001143797, -0.03835666},
            {-0.02367725, 0.05068012, -0.02560657, 0.04252958, -0.05385517, -0.04765985, -0.02131102, -0.03949338, 0.001143797, 0.01963284},
            {-0.09996055, -0.04464164, -0.02345095, -0.06419941, -0.05798303, -0.06018579, 0.01182372, -0.03949338, -0.01811827, -0.05078298},
            {-0.02730979, -0.04464164, -0.06656343, -0.1123996, -0.04972731, -0.04139688, 0.000778808, -0.03949338, -0.03581673, -0.009361911},
            {0.03081083, 0.05068012, 0.03259528, 0.04941532, -0.04009564, -0.04358892, -0.06917231, 0.03430886, 0.06301662, 0.003064409},
            {-0.1035931, 0.05068012, -0.046085, -0.02632783, -0.02496016, -0.02480001, 0.03023191, -0.03949338, -0.03980959, -0.05492509},
            {0.06713621, 0.05068012, -0.02991782, 0.05744869, -0.0001930070, -0.01571871, 0.07441156, -0.05056372, -0.03845911, 0.007206516},
            {-0.05273755, -0.04464164, -0.01267283, -0.06075654, -0.0001930070, 0.008080576, 0.01182372, -0.002592262, -0.02712865, -0.05078298},
            {-0.02730979, 0.05068012, -0.01590626, -0.02977071, 0.003934852, -0.0006875805, 0.04127682, -0.03949338, -0.02364456, 0.01134862},
            {-0.0382074, 0.05068012, 0.07139652, -0.05731367, 0.1539137, 0.1558867, 0.000778808, 0.071948, 0.05027649, 0.06933812},
            {0.009015599, -0.04464164, -0.03099563, 0.02187235, 0.00806271, 0.008706873, 0.004460446, -0.002592262, 0.00943641, 0.01134862},
            {0.01264814, 0.05068012, 0.0002609183, -0.01140873, 0.03970963, 0.05724488, -0.03971921, 0.05608052, 0.02405258, 0.03205916},
            {0.06713621, -0.04464164, 0.03690653, -0.05042793, -0.02358421, -0.03450761, 0.0486401, -0.03949338, -0.02595242, -0.03835666},
            {0.04534098, -0.04464164, 0.03906215, 0.04597245, 0.006686757, -0.02417372, 0.008142084, -0.01255556, 0.06432823, 0.0569118},
            {0.06713621, 0.05068012, -0.01482845, 0.05859631, -0.05935898, -0.03450761, -0.06180903, 0.01290621, -0.005145308, 0.04862759},
            {0.02717829, -0.04464164, 0.006727791, 0.03564384, 0.07961226, 0.07071027, 0.01550536, 0.03430886, 0.04067226, 0.01134862},
            {0.0562386, -0.04464164, -0.06871905, -0.0687899, -0.0001930070, -0.001000729, 0.04495846, -0.03764833, -0.04836172, -0.001077698},
            {0.03444337, 0.05068012, -0.00943939, 0.05974393, -0.03596778, -0.007576847, -0.07653559, 0.07120998, 0.0110081, -0.02178823},
            {0.02354575, -0.04464164, 0.01966154, -0.01255635, 0.08374012, 0.03876913, 0.06336665, -0.002592262, 0.0660482, 0.04862759},
            {0.04897352, 0.05068012, 0.07462995, 0.06662967, -0.009824677, -0.002253323, -0.04340085, 0.03430886, 0.03365681, 0.01963284},
            {0.03081083, 0.05068012, -0.008361578, 0.004658002, 0.01494247, 0.02749578, 0.008142084, -0.00812743, -0.02952762, 0.0569118},
            {-0.1035931, 0.05068012, -0.02345095, -0.02288496, -0.08687804, -0.06770135, -0.01762938, -0.03949338, -0.07814091, -0.07149352},
            {0.01628068, 0.05068012, -0.046085, 0.01154374, -0.03321588, -0.01603186, -0.01026611, -0.002592262, -0.0439854, -0.04249877},
            {-0.06000263, 0.05068012, 0.05415152, -0.01944209, -0.04972731, -0.04891244, 0.02286863, -0.03949338, -0.0439854, -0.005219804},
            {-0.02730979, -0.04464164, -0.03530688, -0.02977071, -0.05660707, -0.05862005, 0.03023191, -0.03949338, -0.04986847, -0.129483},
            {0.04170844, -0.04464164, -0.03207344, -0.06190417, 0.07961226, 0.05098192, 0.05600338, -0.009972486, 0.04506617, -0.0590672},
            {-0.08179786, -0.04464164, -0.0816528, -0.04009932, 0.002558899, -0.01853704, 0.07072993, -0.03949338, -0.01090444, -0.09220405},
            {-0.04183994, -0.04464164, 0.04768465, 0.05974393, 0.1277706, 0.1280164, -0.02499266, 0.1081111, 0.06389312, 0.04034337},
            {-0.01277963, -0.04464164, 0.0606184, 0.05285819, 0.04796534, 0.02937467, -0.01762938, 0.03430886, 0.0702113, 0.007206516},
            {0.06713621, -0.04464164, 0.05630715, 0.07351542, -0.01395254, -0.03920484, -0.03235593, -0.002592262, 0.07573759, 0.03620126},
            {-0.05273755, 0.05068012, 0.09834182, 0.0872869, 0.06034892, 0.04878988, -0.0581274, 0.1081111, 0.08449528, 0.04034337},
            {0.00538306, -0.04464164, 0.05954058, -0.05616605, 0.02457414, 0.05286081, -0.04340085, 0.05091436, -0.00421986, -0.03007245},
            {0.08166637, -0.04464164, 0.03367309, 0.008100872, 0.0520932, 0.05661859, -0.01762938, 0.03430886, 0.03486419, 0.06933812},
            {0.03081083, 0.05068012, 0.05630715, 0.07695829, 0.0493413, -0.01227407, -0.03603757, 0.07120998, 0.1200534, 0.09004865},
            {0.001750522, -0.04464164, -0.06548562, -0.00567061, -0.007072771, -0.01947649, 0.04127682, -0.03949338, -0.003303713, 0.007206516},
            {-0.04910502, -0.04464164, 0.1608549, -0.04698506, -0.02908802, -0.01978964, -0.04708248, 0.03430886, 0.02801651, 0.01134862},
            {-0.02730979, 0.05068012, -0.05578531, 0.02531523, -0.007072771, -0.02354742, 0.05232174, -0.03949338, -0.005145308, -0.05078298},
            {0.07803383, 0.05068012, -0.02452876, -0.04239456, 0.006686757, 0.05286081, -0.06917231, 0.08080427, -0.03712835, 0.0569118},
            {0.01264814, -0.04464164, -0.03638469, 0.04252958, -0.01395254, 0.01293438, -0.02683348, 0.005156973, -0.0439854, 0.007206516},
            {0.04170844, -0.04464164, -0.008361578, -0.05731367, 0.00806271, -0.03137613, 0.1517260, -0.0763945, -0.08023654, -0.01764613},
            {0.04897352, -0.04464164, -0.04177375, 0.1045013, 0.03558177, -0.02573946, 0.1774974, -0.0763945, -0.01290794, 0.01549073},
            {-0.01641217, 0.05068012, 0.1274427, 0.09761551, 0.01631843, 0.01747503, -0.02131102, 0.03430886, 0.03486419, 0.003064409},
            {-0.07453279, 0.05068012, -0.07734155, -0.04698506, -0.0469754, -0.03262872, 0.004460446, -0.03949338, -0.07212845, -0.01764613},
            {0.03444337, 0.05068012, 0.02828403, -0.03321358, -0.04559945, -0.009768886, -0.05076412, -0.002592262, -0.0594727, -0.02178823},
            {-0.03457486, 0.05068012, -0.02560657, -0.01714685, 0.001182946, -0.00287962, 0.008142084, -0.01550765, 0.01482271, 0.04034337},
            {-0.05273755, 0.05068012, -0.06225218, 0.01154374, -0.008448724, -0.03669965, 0.1222729, -0.0763945, -0.086829, 0.003064409},
            {0.05987114, -0.04464164, -0.0008168938, -0.08485664, 0.0754844, 0.07947843, 0.004460446, 0.03430886, 0.02337484, 0.02791705},
            {0.06350368, 0.05068012, 0.0886415, 0.07007254, 0.02044629, 0.03751653, -0.05076412, 0.07120998, 0.02930041, 0.07348023},
            {0.009015599, -0.04464164, -0.03207344, -0.02632783, 0.04246153, -0.01039518, 0.1590892, -0.0763945, -0.01190068, -0.03835666},
            {0.00538306, 0.05068012, 0.03043966, 0.08384403, -0.03734373, -0.0473467, 0.01550536, -0.03949338, 0.008640283, 0.01549073},
            {0.03807591, 0.05068012, 0.008883415, 0.04252958, -0.04284755, -0.02104223, -0.03971921, -0.002592262, -0.01811827, 0.007206516},
            {0.01264814, -0.04464164, 0.006727791, -0.05616605, -0.07587041, -0.06644876, -0.02131102, -0.03764833, -0.01811827, -0.09220405},
            {0.07440129, 0.05068012, -0.02021751, 0.04597245, 0.07410845, 0.03281930, -0.03603757, 0.07120998, 0.1063543, 0.03620126},
            {0.01628068, -0.04464164, -0.02452876, 0.03564384, -0.007072771, -0.003192768, -0.01394774, -0.002592262, 0.01556684, 0.01549073},
            {-0.005514555, 0.05068012, -0.01159501, 0.01154374, -0.02220825, -0.01540556, -0.02131102, -0.002592262, 0.0110081, 0.06933812},
            {0.01264814, -0.04464164, 0.02612841, 0.0631868, 0.1250187, 0.09169122, 0.06336665, -0.002592262, 0.05757286, -0.02178823},
            {-0.03457486, -0.04464164, -0.05901875, 0.001215131, -0.05385517, -0.07803525, 0.06704829, -0.0763945, -0.02139368, 0.01549073},
            {0.06713621, 0.05068012, -0.03638469, -0.08485664, -0.007072771, 0.01966707, -0.05444576, 0.03430886, 0.001143797, 0.03205916},
            {0.03807591, 0.05068012, -0.02452876, 0.004658002, -0.02633611, -0.02636575, 0.01550536, -0.03949338, -0.01599827, -0.02593034},
            {0.009015599, 0.05068012, 0.01858372, 0.03908671, 0.01769438, 0.01058576, 0.01918700, -0.002592262, 0.01630495, -0.01764613},
            {-0.09269548, 0.05068012, -0.0902753, -0.05731367, -0.02496016, -0.03043668, -0.006584468, -0.002592262, 0.02405258, 0.003064409},
            {0.07076875, -0.04464164, -0.005128142, -0.00567061, 0.08786798, 0.1029646, 0.01182372, 0.03430886, -0.008944019, 0.02791705},
            {-0.01641217, -0.04464164, -0.05255187, -0.03321358, -0.0442235, -0.03638651, 0.01918700, -0.03949338, -0.06832974, -0.03007245},
            {0.04170844, 0.05068012, -0.02237314, 0.02875810, -0.06623874, -0.04515466, -0.06180903, -0.002592262, 0.002863771, -0.05492509},
            {0.01264814, -0.04464164, -0.02021751, -0.01599922, 0.01219057, 0.02123281, -0.07653559, 0.1081111, 0.05988072, -0.02178823},
            {-0.0382074, -0.04464164, -0.0547075, -0.0779709, -0.03321588, -0.08649026, 0.1406810, -0.0763945, -0.01919705, -0.005219804},
            {0.04534098, -0.04464164, -0.006205954, -0.01599922, 0.1250187, 0.1251981, 0.01918700, 0.03430886, 0.03243323, -0.005219804},
            {0.07076875, 0.05068012, -0.01698407, 0.02187235, 0.04383748, 0.05630544, 0.03759519, -0.002592262, -0.07020931, -0.01764613},
            {-0.07453279, 0.05068012, 0.05522933, -0.04009932, 0.05346915, 0.05317395, -0.04340085, 0.07120998, 0.06123791, -0.03421455},
            {0.05987114, 0.05068012, 0.07678558, 0.02531523, 0.001182946, 0.01684873, -0.05444576, 0.03430886, 0.02993565, 0.04448548},
            {0.07440129, -0.04464164, 0.01858372, 0.0631868, 0.06172487, 0.04284006, 0.008142084, -0.002592262, 0.05803913, -0.0590672},
            {0.009015599, -0.04464164, -0.02237314, -0.03206595, -0.04972731, -0.0686408, 0.0780932, -0.07085934, -0.06291295, -0.03835666},
            {-0.07090025, -0.04464164, 0.09295276, 0.01269137, 0.02044629, 0.04252691, 0.000778808, 0.0003598277, -0.05454415, -0.001077698},
            {0.02354575, 0.05068012, -0.03099563, -0.00567061, -0.01670444, 0.01778818, -0.03235593, -0.002592262, -0.07408887, -0.03421455},
            {-0.05273755, 0.05068012, 0.03906215, -0.04009932, -0.005696818, -0.01290037, 0.01182372, -0.03949338, 0.01630495, 0.003064409},
            {0.06713621, -0.04464164, -0.06117437, -0.04009932, -0.02633611, -0.02448686, 0.03391355, -0.03949338, -0.05615757, -0.0590672},
            {0.001750522, -0.04464164, -0.008361578, -0.06419941, -0.03871969, -0.02448686, 0.004460446, -0.03949338, -0.06468302, -0.05492509},
            {0.02354575, 0.05068012, -0.03746250, -0.04698506, -0.0910059, -0.07553006, -0.03235593, -0.03949338, -0.03075121, -0.01350402},
            {0.03807591, 0.05068012, -0.01375064, -0.01599922, -0.03596778, -0.02198168, -0.01394774, -0.002592262, -0.02595242, -0.001077698},
            {0.01628068, -0.04464164, 0.07355214, -0.04124694, -0.004320866, -0.01352667, -0.01394774, -0.001116217, 0.04289569, 0.04448548},
            {-0.001882017, 0.05068012, -0.02452876, 0.05285819, 0.02732605, 0.03000097, 0.03023191, -0.002592262, -0.02139368, 0.03620126},
            {0.01264814, -0.04464164, 0.03367309, 0.03334859, 0.03007796, 0.02718263, -0.00290283, 0.008847085, 0.03119299, 0.02791705},
            {0.07440129, -0.04464164, 0.03475090, 0.09417264, 0.05759701, 0.02029337, 0.02286863, -0.002592262, 0.07380215, -0.02178823},
            {0.04170844, 0.05068012, -0.03854032, 0.05285819, 0.07686035, 0.1164299, -0.03971921, 0.07120998, -0.02251217, -0.01350402},
            {-0.009147093, 0.05068012, -0.03961813, -0.04009932, -0.008448724, 0.01622244, -0.06549067, 0.07120998, 0.01776348, -0.06735141},
            {0.009015599, 0.05068012, -0.001894706, 0.02187235, -0.03871969, -0.02480001, -0.006584468, -0.03949338, -0.03980959, -0.01350402},
            {0.06713621, 0.05068012, -0.03099563, 0.004658002, 0.02457414, 0.03563764, -0.02867429, 0.03430886, 0.02337484, 0.08176444},
            {0.001750522, -0.04464164, -0.046085, -0.03321358, -0.07311851, -0.08147988, 0.04495846, -0.06938329, -0.0611766, -0.07977773},
            {-0.009147093, 0.05068012, 0.001338730, -0.00222774, 0.07961226, 0.07008397, 0.03391355, -0.002592262, 0.02671426, 0.08176444},
            {-0.005514555, -0.04464164, 0.06492964, 0.03564384, -0.001568960, 0.01496984, -0.01394774, 0.0007288389, -0.01811827, 0.03205916},
            {0.09619652, -0.04464164, 0.04013997, -0.05731367, 0.04521344, 0.06068952, -0.02131102, 0.03615391, 0.01255315, 0.02377494},
            {-0.07453279, -0.04464164, -0.02345095, -0.00567061, -0.0208323, -0.01415296, 0.01550536, -0.03949338, -0.03845911, -0.03007245},
            {0.05987114, 0.05068012, 0.05307371, 0.05285819, 0.03282986, 0.01966707, -0.01026611, 0.03430886, 0.05520504, -0.001077698},
            {-0.02367725, -0.04464164, 0.04013997, -0.01255635, -0.009824677, -0.001000729, -0.00290283, -0.002592262, -0.01190068, -0.03835666},
            {0.009015599, -0.04464164, -0.02021751, -0.0538708, 0.03145391, 0.02060651, 0.05600338, -0.03949338, -0.01090444, -0.001077698},
            {0.01628068, 0.05068012, 0.01427248, 0.001215131, 0.001182946, -0.02135538, -0.03235593, 0.03430886, 0.07496834, 0.04034337},
            {0.01991321, -0.04464164, -0.03422907, 0.05515344, 0.06722868, 0.0741549, -0.006584468, 0.03283281, 0.02472532, 0.06933812},
            {0.08893144, -0.04464164, 0.006727791, 0.02531523, 0.03007796, 0.008706873, 0.06336665, -0.03949338, 0.00943641, 0.03205916},
            {0.01991321, -0.04464164, 0.004572167, 0.04597245, -0.01808039, -0.05454912, 0.06336665, -0.03949338, 0.02866072, 0.06105391},
            {-0.02367725, -0.04464164, 0.03043966, -0.00567061, 0.08236416, 0.09200436, -0.01762938, 0.07120998, 0.03304707, 0.003064409},
            {0.09619652, -0.04464164, 0.0519959, 0.07925353, 0.05484511, 0.03657709, -0.07653559, 0.1413221, 0.09864637, 0.06105391},
            {0.02354575, 0.05068012, 0.06169621, 0.06203918, 0.02457414, -0.03607336, -0.09126214, 0.1553445, 0.1333957, 0.08176444},
            {0.07076875, 0.05068012, -0.007283766, 0.04941532, 0.06034892, -0.004445362, -0.05444576, 0.1081111, 0.1290194, 0.0569118},
            {0.03081083, -0.04464164, 0.005649979, 0.01154374, 0.0782363, 0.07791268, -0.04340085, 0.1081111, 0.0660482, 0.01963284},
            {-0.001882017, -0.04464164, 0.05415152, -0.06649466, 0.0727325, 0.05661859, -0.04340085, 0.0848634, 0.08449528, 0.04862759},
            {0.04534098, 0.05068012, -0.008361578, -0.03321358, -0.007072771, 0.001191310, -0.03971921, 0.03430886, 0.02993565, 0.02791705},
            {0.07440129, -0.04464164, 0.114509, 0.02875810, 0.02457414, 0.02499059, 0.01918700, -0.002592262, -0.0006092542, -0.005219804},
            {-0.0382074, -0.04464164, 0.06708527, -0.06075654, -0.02908802, -0.02323427, -0.01026611, -0.002592262, -0.001498587, 0.01963284},
            {-0.01277963, 0.05068012, -0.05578531, -0.00222774, -0.02771206, -0.02918409, 0.01918700, -0.03949338, -0.01705210, 0.04448548},
            {0.009015599, 0.05068012, 0.03043966, 0.04252958, -0.002944913, 0.03689023, -0.06549067, 0.07120998, -0.02364456, 0.01549073},
            {0.08166637, 0.05068012, -0.02560657, -0.03665645, -0.0703666, -0.04640726, -0.03971921, -0.002592262, -0.04118039, -0.005219804},
            {0.03081083, -0.04464164, 0.1048087, 0.07695829, -0.01120063, -0.01133463, -0.0581274, 0.03430886, 0.05710419, 0.03620126},
            {0.02717829, 0.05068012, -0.006205954, 0.02875810, -0.01670444, -0.001627026, -0.0581274, 0.03430886, 0.02930041, 0.03205916},
            {-0.06000263, 0.05068012, -0.04716281, -0.02288496, -0.07174256, -0.0576806, -0.006584468, -0.03949338, -0.06291295, -0.05492509},
            {0.00538306, -0.04464164, -0.04824063, -0.01255635, 0.001182946, -0.006637401, 0.06336665, -0.03949338, -0.05140054, -0.0590672},
            {-0.02004471, -0.04464164, 0.08540807, -0.03665645, 0.09199583, 0.08949918, -0.06180903, 0.1450122, 0.08094791, 0.05276969},
            {0.01991321, 0.05068012, -0.01267283, 0.07007254, -0.01120063, 0.007141131, -0.03971921, 0.03430886, 0.00538437, 0.003064409},
            {-0.06363517, -0.04464164, -0.03315126, -0.03321358, 0.001182946, 0.02405115, -0.02499266, -0.002592262, -0.02251217, -0.0590672},
            {0.02717829, -0.04464164, -0.007283766, -0.05042793, 0.0754844, 0.05661859, 0.03391355, -0.002592262, 0.04344317, 0.01549073},
            {-0.01641217, -0.04464164, -0.01375064, 0.1320442, -0.009824677, -0.003819065, 0.01918700, -0.03949338, -0.03581673, -0.03007245},
            {0.03081083, 0.05068012, 0.05954058, 0.05630106, -0.02220825, 0.001191310, -0.03235593, -0.002592262, -0.02479119, -0.01764613},
            {0.0562386, 0.05068012, 0.02181716, 0.05630106, -0.007072771, 0.01810133, -0.03235593, -0.002592262, -0.02364456, 0.02377494},
            {-0.02004471, -0.04464164, 0.01858372, 0.09072977, 0.003934852, 0.008706873, 0.03759519, -0.03949338, -0.05780007, 0.007206516},
            {-0.1072256, -0.04464164, -0.01159501, -0.04009932, 0.0493413, 0.0644473, -0.01394774, 0.03430886, 0.007026863, -0.03007245},
            {0.08166637, 0.05068012, -0.002972518, -0.03321358, 0.04246153, 0.05787118, -0.01026611, 0.03430886, -0.0006092542, -0.001077698},
            {0.00538306, 0.05068012, 0.01750591, 0.03220097, 0.1277706, 0.1273901, -0.02131102, 0.07120998, 0.06257518, 0.01549073},
            {0.03807591, 0.05068012, -0.02991782, -0.07452802, -0.01257658, -0.01258722, 0.004460446, -0.002592262, 0.003711738, -0.03007245},
            {0.03081083, -0.04464164, -0.02021751, -0.00567061, -0.004320866, -0.02949724, 0.0780932, -0.03949338, -0.01090444, -0.001077698},
            {0.001750522, 0.05068012, -0.05794093, -0.04354219, -0.0965097, -0.04703355, -0.09862541, 0.03430886, -0.0611766, -0.07149352},
            {-0.02730979, 0.05068012, 0.0606184, 0.1079441, 0.01219057, -0.01759760, -0.00290283, -0.002592262, 0.0702113, 0.1356118},
            {-0.0854304, 0.05068012, -0.04069594, -0.03321358, -0.08137423, -0.06958024, -0.006584468, -0.03949338, -0.05780007, -0.04249877},
            {0.01264814, 0.05068012, -0.07195249, -0.04698506, -0.05110326, -0.0971373, 0.1185912, -0.0763945, -0.02028875, -0.03835666},
            {-0.05273755, -0.04464164, -0.05578531, -0.03665645, 0.08924393, -0.003192768, 0.008142084, 0.03430886, 0.1323726, 0.003064409},
            {-0.02367725, 0.05068012, 0.04552903, 0.02187235, 0.1098832, 0.08887288, 0.000778808, 0.03430886, 0.07419254, 0.06105391},
            {-0.07453279, 0.05068012, -0.00943939, 0.01498661, -0.03734373, -0.02166853, -0.01394774, -0.002592262, -0.03324879, 0.01134862},
            {-0.005514555, 0.05068012, -0.03315126, -0.01599922, 0.00806271, 0.01622244, 0.01550536, -0.002592262, -0.02832024, -0.07563562},
            {-0.06000263, 0.05068012, 0.04984027, 0.01842948, -0.01670444, -0.03012354, -0.01762938, -0.002592262, 0.04976866, -0.0590672},
            {-0.02004471, -0.04464164, -0.08488624, -0.02632783, -0.03596778, -0.03419447, 0.04127682, -0.05167075, -0.08238148, -0.04664087},
            {0.03807591, 0.05068012, 0.005649979, 0.03220097, 0.006686757, 0.01747503, -0.02499266, 0.03430886, 0.01482271, 0.06105391},
            {0.01628068, -0.04464164, 0.02073935, 0.02187235, -0.01395254, -0.01321352, -0.006584468, -0.002592262, 0.01331597, 0.04034337},
            {0.04170844, -0.04464164, -0.007283766, 0.02875810, -0.04284755, -0.04828615, 0.05232174, -0.0763945, -0.07212845, 0.02377494},
            {0.01991321, 0.05068012, 0.1048087, 0.07007254, -0.03596778, -0.02667890, -0.02499266, -0.002592262, 0.003711738, 0.04034337},
            {-0.04910502, 0.05068012, -0.02452876, 6.750728e-05, -0.0469754, -0.02824465, -0.06549067, 0.02840468, 0.01919903, 0.01134862},
            {0.001750522, 0.05068012, -0.006205954, -0.01944209, -0.009824677, 0.004949092, -0.03971921, 0.03430886, 0.01482271, 0.09833287},
            {0.03444337, -0.04464164, -0.03854032, -0.01255635, 0.009438663, 0.00526224, -0.006584468, -0.002592262, 0.03119299, 0.09833287},
            {-0.04547248, 0.05068012, 0.1371431, -0.01599922, 0.04108558, 0.03187986, -0.04340085, 0.07120998, 0.07102158, 0.04862759},
            {-0.009147093, 0.05068012, 0.1705552, 0.01498661, 0.03007796, 0.03375875, -0.02131102, 0.03430886, 0.03365681, 0.03205916},
            {-0.01641217, 0.05068012, 0.002416542, 0.01498661, 0.02182224, -0.01008203, -0.02499266, 0.03430886, 0.08553312, 0.08176444},
            {-0.009147093, -0.04464164, 0.03798434, -0.04009932, -0.02496016, -0.003819065, -0.04340085, 0.0158583, -0.005145308, 0.02791705},
            {0.01991321, -0.04464164, -0.05794093, -0.05731367, -0.001568960, -0.01258722, 0.07441156, -0.03949338, -0.0611766, -0.07563562},
            {0.05260606, 0.05068012, -0.00943939, 0.04941532, 0.05071725, -0.01916334, -0.01394774, 0.03430886, 0.119344, -0.01764613},
            {-0.02730979, 0.05068012, -0.02345095, -0.01599922, 0.01356652, 0.01277780, 0.02655027, -0.002592262, -0.01090444, -0.02178823},
            {-0.07453279, -0.04464164, -0.01051720, -0.00567061, -0.06623874, -0.0570543, -0.00290283, -0.03949338, -0.04257210, -0.001077698},
            {-0.1072256, -0.04464164, -0.03422907, -0.06764228, -0.06348684, -0.07051969, 0.008142084, -0.03949338, -0.0006092542, -0.07977773},
            {0.04534098, 0.05068012, -0.002972518, 0.1079441, 0.03558177, 0.02248541, 0.02655027, -0.002592262, 0.02801651, 0.01963284},
            {-0.001882017, -0.04464164, 0.06816308, -0.00567061, 0.1195149, 0.1302085, -0.02499266, 0.08670845, 0.04613233, -0.001077698},
            {0.01991321, 0.05068012, 0.009961227, 0.01842948, 0.01494247, 0.04471895, -0.06180903, 0.07120998, 0.00943641, -0.0632093},
            {0.01628068, 0.05068012, 0.002416542, -0.00567061, -0.005696818, 0.01089891, -0.05076412, 0.03430886, 0.02269202, -0.03835666},
            {-0.001882017, -0.04464164, -0.03854032, 0.02187235, -0.1088933, -0.1156131, 0.02286863, -0.0763945, -0.04687948, 0.02377494},
            {0.01628068, -0.04464164, 0.02612841, 0.05859631, -0.06073493, -0.04421522, -0.01394774, -0.03395821, -0.05140054, -0.02593034},
            {-0.07090025, 0.05068012, -0.08919748, -0.07452802, -0.04284755, -0.02573946, -0.03235593, -0.002592262, -0.01290794, -0.05492509},
            {0.04897352, -0.04464164, 0.0606184, -0.02288496, -0.02358421, -0.07271173, -0.04340085, -0.002592262, 0.1041376, 0.03620126},
            {0.00538306, 0.05068012, -0.02884001, -0.009113481, -0.03183992, -0.02887094, 0.008142084, -0.03949338, -0.01811827, 0.007206516},
            {0.03444337, 0.05068012, -0.02991782, 0.004658002, 0.09337179, 0.08699399, 0.03391355, -0.002592262, 0.02405258, -0.03835666},
            {0.02354575, 0.05068012, -0.0191397, 0.04941532, -0.06348684, -0.06112523, 0.004460446, -0.03949338, -0.02595242, -0.01350402},
            {0.01991321, -0.04464164, -0.04069594, -0.01599922, -0.008448724, -0.01759760, 0.05232174, -0.03949338, -0.03075121, 0.003064409},
            {-0.04547248, -0.04464164, 0.01535029, -0.07452802, -0.04972731, -0.01728445, -0.02867429, -0.002592262, -0.1043648, -0.07563562},
            {0.05260606, 0.05068012, -0.02452876, 0.05630106, -0.007072771, -0.005071659, -0.02131102, -0.002592262, 0.02671426, -0.03835666},
            {-0.005514555, 0.05068012, 0.001338730, -0.08485664, -0.01120063, -0.01665815, 0.0486401, -0.03949338, -0.04118039, -0.08806194},
            {0.009015599, 0.05068012, 0.06924089, 0.05974393, 0.01769438, -0.02323427, -0.04708248, 0.03430886, 0.1032923, 0.07348023},
            {-0.02367725, -0.04464164, -0.06979687, -0.06419941, -0.05935898, -0.05047819, 0.01918700, -0.03949338, -0.08913686, -0.05078298},
            {-0.04183994, 0.05068012, -0.02991782, -0.00222774, 0.02182224, 0.03657709, 0.01182372, -0.002592262, -0.04118039, 0.06519601},
            {-0.07453279, -0.04464164, -0.046085, -0.04354219, -0.02908802, -0.02323427, 0.01550536, -0.03949338, -0.03980959, -0.02178823},
            {0.03444337, -0.04464164, 0.01858372, 0.05630106, 0.01219057, -0.05454912, -0.06917231, 0.07120998, 0.1300806, 0.007206516},
            {-0.06000263, -0.04464164, 0.001338730, -0.02977071, -0.007072771, -0.02166853, 0.01182372, -0.002592262, 0.03181522, -0.05492509},
            {-0.0854304, 0.05068012, -0.03099563, -0.02288496, -0.06348684, -0.05423597, 0.01918700, -0.03949338, -0.09643322, -0.03421455},
            {0.05260606, -0.04464164, -0.00405033, -0.03091833, -0.0469754, -0.0583069, -0.01394774, -0.02583997, 0.03605579, 0.02377494},
            {0.01264814, -0.04464164, 0.01535029, -0.03321358, 0.04108558, 0.03219301, -0.00290283, -0.002592262, 0.04506617, -0.06735141},
            {0.05987114, 0.05068012, 0.02289497, 0.04941532, 0.01631843, 0.01183836, -0.01394774, -0.002592262, 0.03953988, 0.01963284},
            {-0.02367725, -0.04464164, 0.04552903, 0.09072977, -0.01808039, -0.03544706, 0.07072993, -0.03949338, -0.03452372, -0.009361911},
            {0.01628068, -0.04464164, -0.04500719, -0.05731367, -0.03459183, -0.05392282, 0.07441156, -0.0763945, -0.04257210, 0.04034337},
            {0.1107267, 0.05068012, -0.03315126, -0.02288496, -0.004320866, 0.02029337, -0.06180903, 0.07120998, 0.01556684, 0.04448548},
            {-0.02004471, -0.04464164, 0.097264, -0.00567061, -0.005696818, -0.02386057, -0.02131102, -0.002592262, 0.06168585, 0.04034337},
            {-0.01641217, -0.04464164, 0.05415152, 0.07007254, -0.03321588, -0.02793150, 0.008142084, -0.03949338, -0.02712865, -0.009361911},
            {0.04897352, 0.05068012, 0.1231315, 0.08384403, -0.1047654, -0.1008951, -0.06917231, -0.002592262, 0.0366458, -0.03007245},
            {-0.05637009, -0.04464164, -0.08057499, -0.08485664, -0.03734373, -0.0370128, 0.03391355, -0.03949338, -0.05615757, -0.1377672},
            {0.02717829, -0.04464164, 0.09295276, -0.05272318, 0.00806271, 0.03970857, -0.02867429, 0.02102446, -0.04836172, 0.01963284},
            {0.06350368, -0.04464164, -0.05039625, 0.1079441, 0.03145391, 0.01935392, -0.01762938, 0.02360753, 0.05803913, 0.04034337},
            {-0.05273755, 0.05068012, -0.01159501, 0.05630106, 0.05622106, 0.07290231, -0.03971921, 0.07120998, 0.03056649, -0.005219804},
            {-0.009147093, 0.05068012, -0.02776220, 0.008100872, 0.04796534, 0.03720338, -0.02867429, 0.03430886, 0.0660482, -0.04249877},
            {0.00538306, -0.04464164, 0.05846277, -0.04354219, -0.07311851, -0.07239858, 0.01918700, -0.0763945, -0.05140054, -0.02593034},
            {0.07440129, -0.04464164, 0.08540807, 0.0631868, 0.01494247, 0.01309095, 0.01550536, -0.002592262, 0.006209316, 0.08590655},
            {-0.05273755, -0.04464164, -0.0008168938, -0.02632783, 0.01081462, 0.007141131, 0.0486401, -0.03949338, -0.03581673, 0.01963284},
            {0.08166637, 0.05068012, 0.006727791, -0.004522987, 0.1098832, 0.1170562, -0.03235593, 0.0918746, 0.054724, 0.007206516},
            {-0.005514555, -0.04464164, 0.008883415, -0.05042793, 0.02595010, 0.04722413, -0.04340085, 0.07120998, 0.01482271, 0.003064409},
            {-0.02730979, -0.04464164, 0.08001901, 0.09876313, -0.002944913, 0.01810133, -0.01762938, 0.003311917, -0.02952762, 0.03620126},
            {-0.05273755, -0.04464164, 0.07139652, -0.07452802, -0.01532849, -0.001313877, 0.004460446, -0.02141183, -0.04687948, 0.003064409},
            {0.009015599, -0.04464164, -0.02452876, -0.02632783, 0.0988756, 0.0941964, 0.07072993, -0.002592262, -0.02139368, 0.007206516},
            {-0.02004471, -0.04464164, -0.0547075, -0.0538708, -0.06623874, -0.05736745, 0.01182372, -0.03949338, -0.07408887, -0.005219804},
            {0.02354575, -0.04464164, -0.03638469, 6.750728e-05, 0.001182946, 0.03469820, -0.04340085, 0.03430886, -0.03324879, 0.06105391},
            {0.03807591, 0.05068012, 0.0164281, 0.02187235, 0.03970963, 0.04503209, -0.04340085, 0.07120998, 0.04976866, 0.01549073},
            {-0.07816532, 0.05068012, 0.07786339, 0.05285819, 0.0782363, 0.0644473, 0.02655027, -0.002592262, 0.04067226, -0.009361911},
            {0.009015599, 0.05068012, -0.03961813, 0.02875810, 0.03833367, 0.0735286, -0.07285395, 0.1081111, 0.01556684, -0.04664087},
            {0.001750522, 0.05068012, 0.01103904, -0.01944209, -0.01670444, -0.003819065, -0.04708248, 0.03430886, 0.02405258, 0.02377494},
            {-0.07816532, -0.04464164, -0.04069594, -0.08141377, -0.1006376, -0.1127947, 0.02286863, -0.0763945, -0.02028875, -0.05078298},
            {0.03081083, 0.05068012, -0.03422907, 0.0436772, 0.05759701, 0.06883138, -0.03235593, 0.05755657, 0.03546194, 0.08590655},
            {-0.03457486, 0.05068012, 0.005649979, -0.00567061, -0.07311851, -0.06269098, -0.006584468, -0.03949338, -0.04542096, 0.03205916},
            {0.04897352, 0.05068012, 0.0886415, 0.0872869, 0.03558177, 0.02154596, -0.02499266, 0.03430886, 0.0660482, 0.1314697},
            {-0.04183994, -0.04464164, -0.03315126, -0.02288496, 0.04658939, 0.04158746, 0.05600338, -0.02473293, -0.02595242, -0.03835666},
            {-0.009147093, -0.04464164, -0.05686312, -0.05042793, 0.02182224, 0.04534524, -0.02867429, 0.03430886, -0.009918957, -0.01764613},
            {0.07076875, 0.05068012, -0.03099563, 0.02187235, -0.03734373, -0.04703355, 0.03391355, -0.03949338, -0.01495648, -0.001077698},
            {0.009015599, -0.04464164, 0.05522933, -0.00567061, 0.05759701, 0.04471895, -0.00290283, 0.02323852, 0.05568355, 0.1066171},
            {-0.02730979, -0.04464164, -0.06009656, -0.02977071, 0.04658939, 0.01998022, 0.1222729, -0.03949338, -0.05140054, -0.009361911},
            {0.01628068, -0.04464164, 0.001338730, 0.008100872, 0.005310804, 0.01089891, 0.03023191, -0.03949338, -0.04542096, 0.03205916},
            {-0.01277963, -0.04464164, -0.02345095, -0.04009932, -0.01670444, 0.004635943, -0.01762938, -0.002592262, -0.03845911, -0.03835666},
            {-0.05637009, -0.04464164, -0.07410811, -0.05042793, -0.02496016, -0.04703355, 0.09281975, -0.0763945, -0.0611766, -0.04664087},
            {0.04170844, 0.05068012, 0.01966154, 0.05974393, -0.005696818, -0.002566471, -0.02867429, -0.002592262, 0.03119299, 0.007206516},
            {-0.005514555, 0.05068012, -0.01590626, -0.06764228, 0.0493413, 0.07916528, -0.02867429, 0.03430886, -0.01811827, 0.04448548},
            {0.04170844, 0.05068012, -0.01590626, 0.01728186, -0.03734373, -0.01383982, -0.02499266, -0.01107952, -0.04687948, 0.01549073},
            {-0.04547248, -0.04464164, 0.03906215, 0.001215131, 0.01631843, 0.01528299, -0.02867429, 0.02655962, 0.04452837, -0.02593034},
            {-0.04547248, -0.04464164, -0.0730303, -0.08141377, 0.08374012, 0.02780893, 0.1738158, -0.03949338, -0.00421986, 0.003064409}});

    public static Vector diabetes_y = new DenseVector(
            new double[]{151, 75, 141, 206, 135, 97, 138, 63, 110, 310, 101, 69, 179, 185, 118, 171, 166, 144, 97, 168, 68, 49, 68, 245, 184, 202, 137, 85, 131, 283, 129, 59, 341, 87, 65, 102, 265, 276, 252, 90, 100, 55, 61, 92, 259, 53, 190, 142, 75, 142, 155, 225, 59, 104, 182, 128, 52, 37, 170, 170, 61, 144, 52, 128, 71, 163, 150, 97, 160, 178, 48, 270, 202, 111, 85, 42, 170, 200, 252, 113, 143, 51, 52, 210, 65, 141, 55, 134, 42, 111, 98, 164, 48, 96, 90, 162, 150, 279, 92, 83, 128, 102, 302, 198, 95, 53, 134, 144, 232, 81, 104, 59, 246, 297, 258, 229, 275, 281, 179, 200, 200, 173, 180, 84, 121, 161, 99, 109, 115, 268, 274, 158, 107, 83, 103, 272, 85, 280, 336, 281, 118, 317, 235, 60, 174, 259, 178, 128, 96, 126, 288, 88, 292, 71, 197, 186, 25, 84, 96, 195, 53, 217, 172, 131, 214, 59, 70, 220, 268, 152, 47, 74, 295, 101, 151, 127, 237, 225, 81, 151, 107, 64, 138, 185, 265, 101, 137, 143, 141, 79, 292, 178, 91, 116, 86, 122, 72, 129, 142, 90, 158, 39, 196, 222, 277, 99, 196, 202, 155, 77, 191, 70, 73, 49, 65, 263, 248, 296, 214, 185, 78, 93, 252, 150, 77, 208, 77, 108, 160, 53, 220, 154, 259, 90, 246, 124, 67, 72, 257, 262, 275, 177, 71, 47, 187, 125, 78, 51, 258, 215, 303, 243, 91, 150, 310, 153, 346, 63, 89, 50, 39, 103, 308, 116, 145, 74, 45, 115, 264, 87, 202, 127, 182, 241, 66, 94, 283, 64, 102, 200, 265, 94, 230, 181, 156, 233, 60, 219, 80, 68, 332, 248, 84, 200, 55, 85, 89, 31, 129, 83, 275, 65, 198, 236, 253, 124, 44, 172, 114, 142, 109, 180, 144, 163, 147, 97, 220, 190, 109, 191, 122, 230, 242, 248, 249, 192, 131, 237, 78, 135, 244, 199, 270, 164, 72, 96, 306, 91, 214, 95, 216, 263, 178, 113, 200, 139, 139, 88, 148, 88, 243, 71, 77, 109, 272, 60, 54, 221, 90, 311, 281, 182, 321, 58, 262, 206, 233, 242, 123, 167, 63, 197, 71, 168, 140, 217, 121, 235, 245, 40, 52, 104, 132, 88, 69, 219, 72, 201, 110, 51, 277, 63, 118, 69, 273, 258, 43, 198, 242, 232, 175, 93, 168, 275, 293, 281, 72, 140, 189, 181, 209, 136, 261, 113, 131, 174, 257, 55, 84, 42, 146, 212, 233, 91, 111, 152, 120, 67, 310, 94, 183, 66, 173, 72, 49, 64, 48, 178, 104, 132, 220, 57});

}
