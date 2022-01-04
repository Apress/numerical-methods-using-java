/*
 * Copyright (c) Numerical Method Inc.
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
import dev.nm.algebra.linear.matrix.doubles.MatrixPropertyUtils;
import dev.nm.algebra.linear.matrix.doubles.factorization.diagonalization.TriDiagonalization;
import dev.nm.algebra.linear.matrix.doubles.factorization.eigen.Eigen;
import dev.nm.algebra.linear.matrix.doubles.factorization.eigen.EigenDecomposition;
import dev.nm.algebra.linear.matrix.doubles.factorization.eigen.EigenProperty;
import dev.nm.algebra.linear.matrix.doubles.factorization.eigen.qr.Hessenberg;
import dev.nm.algebra.linear.matrix.doubles.factorization.eigen.qr.HessenbergDecomposition;
import dev.nm.algebra.linear.matrix.doubles.factorization.gaussianelimination.GaussJordanElimination;
import dev.nm.algebra.linear.matrix.doubles.factorization.gaussianelimination.GaussianElimination;
import dev.nm.algebra.linear.matrix.doubles.factorization.qr.GramSchmidt;
import dev.nm.algebra.linear.matrix.doubles.factorization.qr.HouseholderQR;
import dev.nm.algebra.linear.matrix.doubles.factorization.qr.QR;
import dev.nm.algebra.linear.matrix.doubles.factorization.qr.QRDecomposition;
import dev.nm.algebra.linear.matrix.doubles.factorization.svd.SVD;
import dev.nm.algebra.linear.matrix.doubles.factorization.triangle.LU;
import dev.nm.algebra.linear.matrix.doubles.factorization.triangle.cholesky.Chol;
import dev.nm.algebra.linear.matrix.doubles.factorization.triangle.cholesky.Cholesky;
import dev.nm.algebra.linear.matrix.doubles.linearsystem.BackwardSubstitution;
import dev.nm.algebra.linear.matrix.doubles.linearsystem.ForwardSubstitution;
import dev.nm.algebra.linear.matrix.doubles.linearsystem.LSProblem;
import dev.nm.algebra.linear.matrix.doubles.linearsystem.LUSolver;
import dev.nm.algebra.linear.matrix.doubles.linearsystem.LinearSystemSolver;
import dev.nm.algebra.linear.matrix.doubles.linearsystem.OLSSolverByQR;
import dev.nm.algebra.linear.matrix.doubles.linearsystem.OLSSolverBySVD;
import dev.nm.algebra.linear.matrix.doubles.matrixtype.PermutationMatrix;
import dev.nm.algebra.linear.matrix.doubles.matrixtype.dense.DenseMatrix;
import dev.nm.algebra.linear.matrix.doubles.matrixtype.dense.diagonal.DiagonalMatrix;
import dev.nm.algebra.linear.matrix.doubles.matrixtype.dense.triangle.LowerTriangularMatrix;
import dev.nm.algebra.linear.matrix.doubles.matrixtype.dense.triangle.SymmetricMatrix;
import dev.nm.algebra.linear.matrix.doubles.matrixtype.dense.triangle.UpperTriangularMatrix;
import dev.nm.algebra.linear.matrix.doubles.matrixtype.sparse.CSRSparseMatrix;
import dev.nm.algebra.linear.matrix.doubles.matrixtype.sparse.DOKSparseMatrix;
import dev.nm.algebra.linear.matrix.doubles.matrixtype.sparse.LILSparseMatrix;
import dev.nm.algebra.linear.matrix.doubles.matrixtype.sparse.MatrixCoordinate;
import dev.nm.algebra.linear.matrix.doubles.matrixtype.sparse.SparseMatrix;
import dev.nm.algebra.linear.matrix.doubles.matrixtype.sparse.SparseVector;
import dev.nm.algebra.linear.matrix.doubles.matrixtype.sparse.solver.iterative.ConvergenceFailure;
import dev.nm.algebra.linear.matrix.doubles.matrixtype.sparse.solver.iterative.IterativeLinearSystemSolver;
import dev.nm.algebra.linear.matrix.doubles.matrixtype.sparse.solver.iterative.nonstationary.BiconjugateGradientSolver;
import dev.nm.algebra.linear.matrix.doubles.matrixtype.sparse.solver.iterative.nonstationary.BiconjugateGradientStabilizedSolver;
import dev.nm.algebra.linear.matrix.doubles.matrixtype.sparse.solver.iterative.nonstationary.ConjugateGradientNormalErrorSolver;
import dev.nm.algebra.linear.matrix.doubles.matrixtype.sparse.solver.iterative.nonstationary.ConjugateGradientNormalResidualSolver;
import dev.nm.algebra.linear.matrix.doubles.matrixtype.sparse.solver.iterative.nonstationary.ConjugateGradientSolver;
import dev.nm.algebra.linear.matrix.doubles.matrixtype.sparse.solver.iterative.nonstationary.GeneralizedConjugateResidualSolver;
import dev.nm.algebra.linear.matrix.doubles.matrixtype.sparse.solver.iterative.nonstationary.GeneralizedMinimalResidualSolver;
import dev.nm.algebra.linear.matrix.doubles.matrixtype.sparse.solver.iterative.nonstationary.MinimalResidualSolver;
import dev.nm.algebra.linear.matrix.doubles.matrixtype.sparse.solver.iterative.nonstationary.QuasiMinimalResidualSolver;
import dev.nm.algebra.linear.matrix.doubles.matrixtype.sparse.solver.iterative.stationary.GaussSeidelSolver;
import dev.nm.algebra.linear.matrix.doubles.matrixtype.sparse.solver.iterative.stationary.JacobiSolver;
import dev.nm.algebra.linear.matrix.doubles.matrixtype.sparse.solver.iterative.stationary.SuccessiveOverrelaxationSolver;
import dev.nm.algebra.linear.matrix.doubles.matrixtype.sparse.solver.iterative.stationary.SymmetricSuccessiveOverrelaxationSolver;
import dev.nm.algebra.linear.matrix.doubles.operation.Inverse;
import dev.nm.algebra.linear.matrix.doubles.operation.KroneckerProduct;
import dev.nm.algebra.linear.matrix.doubles.operation.MatrixMeasure;
import dev.nm.algebra.linear.matrix.doubles.operation.PseudoInverse;
import dev.nm.algebra.linear.vector.doubles.Vector;
import dev.nm.algebra.linear.vector.doubles.dense.DenseVector;
import dev.nm.algebra.linear.vector.doubles.operation.RealVectorSpace;
import dev.nm.misc.algorithm.iterative.monitor.CountMonitor;
import dev.nm.misc.algorithm.iterative.tolerance.AbsoluteTolerance;
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
public class Chapter2 {

    public static void main(String[] args) throws ConvergenceFailure {
        System.out.println("Chapter 2 demos");

        Chapter2 chapter2 = new Chapter2();
        chapter2.vectors();
        chapter2.inverse();
        chapter2.matrices_0010();
        chapter2.matrices_0020();
        chapter2.transpose();
        chapter2.multiplication();
        chapter2.rank();
        chapter2.determinant();
        chapter2.kronecker();
        chapter2.lu();
        chapter2.cholesky();
        chapter2.hessenberg();
        chapter2.tridiagonalization();
        chapter2.qr_0010();
        chapter2.qr_0020();
        chapter2.eigen_decomposition_0010();
        chapter2.eigen_decomposition_0020();
        chapter2.svd();
        chapter2.backward_substitution();
        chapter2.forward_substitution();
        chapter2.LU_solver();
        chapter2.gaussian_elimination();
        chapter2.gauss_jordan_elimination();
        chapter2.linear_system_solver();
        chapter2.overdetermined_system();
        chapter2.sparse_matrices();
        chapter2.sparse_linear_system_nonstationary();
        chapter2.sparse_linear_system_stationary();
    }

    public void sparse_linear_system_stationary() throws ConvergenceFailure {
        System.out.println("solving sparse linear system using stationary iterative solvers");

        Matrix A = new SymmetricMatrix(
                new double[][]{
                    {4},
                    {1, 3}
                });
        Vector b = new DenseVector(
                new double[]{
                    1, 2
                });
        // construct a linear system problem to be solved
        LSProblem problem = new LSProblem(A, b);

        // construct a sparse matrix linear system solver
        GaussSeidelSolver gauss_seidel
                = new GaussSeidelSolver(
                        10,
                        new AbsoluteTolerance(1e-4));
        IterativeLinearSystemSolver.Solution soln1 = gauss_seidel.solve(problem);
        Vector x1 = soln1.search(new SparseVector(A.nCols())); // use 0 as the initial guess
        System.out.println("x = " + x1);
        Vector Ax1_b = A.multiply(x1).minus(b); // verify that Ax = b
        System.out.println("||Ax - b|| = " + Ax1_b.norm()); // should be (close to) 0

        // construct a sparse matrix linear system solver
        JacobiSolver jacobi
                = new JacobiSolver(
                        10,
                        new AbsoluteTolerance(1e-4));
        IterativeLinearSystemSolver.Solution soln2 = jacobi.solve(problem);
        Vector x2 = soln2.search(new SparseVector(A.nCols()));
        System.out.println("x = " + x2);
        Vector Ax2_b = A.multiply(x1).minus(b); // verify that Ax = b
        System.out.println("||Ax - b|| = " + Ax2_b.norm()); // should be (close to) 0

        SuccessiveOverrelaxationSolver SOR
                = new SuccessiveOverrelaxationSolver(
                        1.5,
                        20, // need more iterations
                        new AbsoluteTolerance(1e-4));
        IterativeLinearSystemSolver.Solution soln3 = SOR.solve(problem);
        Vector x3 = soln3.search(new SparseVector(A.nCols())); // use 0 as the initial guess
        System.out.println("x = " + x3);
        Vector Ax3_b = A.multiply(x3).minus(b); // verify that Ax = b
        System.out.println("||Ax - b|| = " + Ax3_b.norm()); // should be (close to) 0

        SymmetricSuccessiveOverrelaxationSolver SSOR
                = new SymmetricSuccessiveOverrelaxationSolver(
                        1.5,
                        20, // need more iterations
                        new AbsoluteTolerance(1e-4));
        IterativeLinearSystemSolver.Solution soln4 = SSOR.solve(problem);
        Vector x4 = soln4.search(new SparseVector(A.nCols())); // use 0 as the initial guess
        System.out.println("x = " + x4);
        Vector Ax4_b = A.multiply(x4).minus(b); // verify that Ax = b
        System.out.println("||Ax - b|| = " + Ax4_b.norm()); // should be (close to) 0
    }

    public void sparse_linear_system_nonstationary() throws ConvergenceFailure {
        System.out.println("solving sparse linear system using non-stationary iterative solvers");

        /* Symmetric matrix:
         * 8x8
         * [,1] [,2] [,3] [,4] [,5] [,6] [,7] [,8]
         * [1,] 7.000000, 0.000000, 1.000000, 0.000000, 0.000000, 2.000000, 7.000000, 0.000000,
         * [2,] 0.000000, -4.000000, 8.000000, 0.000000, 2.000000, 0.000000, 0.000000, 0.000000,
         * [3,] 1.000000, 8.000000, 1.000000, 0.000000, 0.000000, 0.000000, 0.000000, 5.000000,
         * [4,] 0.000000, 0.000000, 0.000000, 7.000000, 0.000000, 0.000000, 9.000000, 0.000000,
         * [5,] 0.000000, 2.000000, 0.000000, 0.000000, 5.000000, 1.000000, 5.000000, 0.000000,
         * [6,] 2.000000, 0.000000, 0.000000, 0.000000, 1.000000, -1.000000, 0.000000, 5.000000,
         * [7,] 7.000000, 0.000000, 0.000000, 9.000000, 5.000000, 0.000000, 11.000000, 0.000000,
         * [8,] 0.000000, 0.000000, 5.000000, 0.000000, 0.000000, 5.000000, 0.000000, 5.000000,
         */
        Matrix A = new CSRSparseMatrix(8, 8, // matrix dimension
                new int[]{1, 1, 1, 1, 2, 2, 2, 3, 3, 3, 3, 4, 4, 5, 5, 5, 5, 6, 6, 6, 6, 7, 7, 7, 7, 8, 8, 8}, // row indices
                new int[]{1, 3, 6, 7, 2, 3, 5, 1, 2, 3, 8, 4, 7, 2, 5, 6, 7, 1, 5, 6, 8, 1, 4, 5, 7, 3, 6, 8}, // column indices
                new double[]{7, 1, 2, 7, -4, 8, 2, 1, 8, 1, 5, 7, 9, 2, 5, 1, 5, 2, 1, -1, 5, 7, 9, 5, 11, 5, 5, 5} // entries/values
        );
        Vector b = new DenseVector( // note that we can still use dense data structure
                new double[]{
                    1, 1, 1, 1, 1, 1, 1, 1
                });
        // construct a linear system problem to be solved
        LSProblem problem = new LSProblem(A, b);

        // construct a sparse matrix linear system solver
        BiconjugateGradientSolver BiCG
                = new BiconjugateGradientSolver(
                        10, // maximum number of iterations
                        new AbsoluteTolerance(1e-8) // precision
                );
        IterativeLinearSystemSolver.Solution soln1 = BiCG.solve(problem);
        Vector x1 = soln1.search(new SparseVector(A.nCols())); // use 0 as the initial guess
        System.out.println("x = " + x1);
        Vector Ax1_b = A.multiply(x1).minus(b); // verify that Ax = b
        System.out.println("||Ax - b|| = " + Ax1_b.norm()); // should be (close to) 0

        // construct a sparse matrix linear system solver
        BiconjugateGradientStabilizedSolver BiCGSTAB
                = new BiconjugateGradientStabilizedSolver(
                        10, // maximum number of iterations
                        new AbsoluteTolerance(1e-7) // less precision
                );
        IterativeLinearSystemSolver.Solution soln2 = BiCGSTAB.solve(problem);
        Vector x2 = soln2.search(new SparseVector(A.nCols())); // use 0 as the initial guess
        System.out.println("x = " + x2);
        Vector Ax2_b = A.multiply(x2).minus(b); // verify that Ax = b
        System.out.println("||Ax - b|| = " + Ax2_b.norm()); // should be (close to) 0

        ConjugateGradientNormalErrorSolver CGNE
                = new ConjugateGradientNormalErrorSolver(
                        10, // maximum number of iterations
                        new AbsoluteTolerance(1e-8) // precision
                );
        IterativeLinearSystemSolver.Solution soln3 = CGNE.solve(problem);
        Vector x3 = soln3.search(new SparseVector(A.nCols())); // use 0 as the initial guess
        System.out.println("x = " + x3);
        Vector Ax3_b = A.multiply(x3).minus(b); // verify that Ax = b
        System.out.println("||Ax - b|| = " + Ax3_b.norm()); // should be (close to) 0

        ConjugateGradientNormalResidualSolver CGNR
                = new ConjugateGradientNormalResidualSolver(
                        10, // maximum number of iterations
                        new AbsoluteTolerance(1e-8) // precision
                );
        IterativeLinearSystemSolver.Solution soln4 = CGNR.solve(problem);
        Vector x4 = soln4.search(new SparseVector(A.nCols())); // use 0 as the initial guess
        System.out.println("x = " + x4);
        Vector Ax4_b = A.multiply(x4).minus(b); // verify that Ax = b
        System.out.println("||Ax - b|| = " + Ax4_b.norm()); // should be (close to) 0

        ConjugateGradientSolver CG
                = new ConjugateGradientSolver(
                        10, // maximum number of iterations
                        new AbsoluteTolerance(1e-8) // precision
                );
        IterativeLinearSystemSolver.Solution soln5 = CG.solve(problem);
        Vector x5 = soln5.search(new SparseVector(A.nCols())); // use 0 as the initial guess
        System.out.println("x = " + x5);
        Vector Ax5_b = A.multiply(x5).minus(b); // verify that Ax = b
        System.out.println("||Ax - b|| = " + Ax5_b.norm()); // should be (close to) 0

        ConjugateGradientNormalResidualSolver CGS
                = new ConjugateGradientNormalResidualSolver(
                        10, // maximum number of iterations
                        new AbsoluteTolerance(1e-8) // precision
                );
        IterativeLinearSystemSolver.Solution soln6 = CGS.solve(problem);
        Vector x6 = soln6.search(new SparseVector(A.nCols())); // use 0 as the initial guess
        System.out.println("x = " + x6);
        Vector Ax6_b = A.multiply(x6).minus(b); // verify that Ax = b
        System.out.println("||Ax - b|| = " + Ax6_b.norm()); // should be (close to) 0

        GeneralizedConjugateResidualSolver GRES
                = new GeneralizedConjugateResidualSolver(
                        10, // maximum number of iterations
                        new AbsoluteTolerance(1e-8) // precision
                );
        IterativeLinearSystemSolver.Solution soln7 = GRES.solve(problem);
        Vector x7 = soln7.search(new SparseVector(A.nCols())); // use 0 as the initial guess
        System.out.println("x = " + x7);
        Vector Ax7_b = A.multiply(x7).minus(b); // verify that Ax = b
        System.out.println("||Ax - b|| = " + Ax7_b.norm()); // should be (close to) 0

        GeneralizedMinimalResidualSolver GMRES
                = new GeneralizedMinimalResidualSolver(
                        10, // maximum number of iterations
                        new AbsoluteTolerance(1e-8) // precision
                );
        IterativeLinearSystemSolver.Solution soln8 = GMRES.solve(problem);
        Vector x8 = soln8.search(new SparseVector(A.nCols())); // use 0 as the initial guess
        System.out.println("x = " + x8);
        Vector Ax8_b = A.multiply(x8).minus(b); // verify that Ax = b
        System.out.println("||Ax - b|| = " + Ax8_b.norm()); // should be (close to) 0

        MinimalResidualSolver MINRES = new MinimalResidualSolver(
                10, // maximum number of iterations
                new AbsoluteTolerance(1e-8) // precision
        );
        CountMonitor<Vector> monitor = new CountMonitor<Vector>();
        IterativeLinearSystemSolver.Solution soln9 = MINRES.solve(problem, monitor);
        Vector x9 = soln9.search(new SparseVector(A.nCols())); // use 0 as the initial guess
        System.out.println("x = " + x9);
        Vector Ax9_b = A.multiply(x9).minus(b); // verify that Ax = b
        System.out.println("||Ax - b|| = " + Ax9_b.norm()); // should be (close to) 0

        QuasiMinimalResidualSolver QMR
                = new QuasiMinimalResidualSolver(
                        10, // maximum number of iterations
                        new AbsoluteTolerance(1e-8) // precision
                );
        IterativeLinearSystemSolver.Solution soln10 = QMR.solve(problem);
        Vector x10 = soln10.search(new SparseVector(A.nCols())); // use 0 as the initial guess
        System.out.println("x = " + x10);
        Vector Ax10_b = A.multiply(x10).minus(b); // verify that Ax = b
        System.out.println("||Ax - b|| = " + Ax10_b.norm()); // should be (close to) 0
    }

    public void sparse_matrices() {
        System.out.println("sparse matrices");

        // the target matrix in dense representation
        Matrix A = new DenseMatrix(new double[][]{
            {1, 2, 0, 0},
            {0, 3, 9, 0},
            {0, 1, 4, 0}
        });

        // DOK
        SparseMatrix B1 = new DOKSparseMatrix(3, 4, // 3x4 dimension
                new int[]{3, 2, 1, 3, 2, 1}, // row indices
                new int[]{3, 2, 1, 2, 3, 2}, // column indices
                new double[]{4, 3, 1, 1, 9, 2} // matrix entries/values
        );
        //verify that B1 = A
        System.out.println(String.format(
                "B1 = A, %b",
                MatrixPropertyUtils.areEqual(B1, A, 1e-15)));
        SparseMatrix B2 = new DOKSparseMatrix(3, 4, // 3x4 dimension
                Arrays.<SparseMatrix.Entry>asList( // specify only the non-zero entries
                        new SparseMatrix.Entry(new MatrixCoordinate(3, 3), 4),
                        new SparseMatrix.Entry(new MatrixCoordinate(2, 2), 3),
                        new SparseMatrix.Entry(new MatrixCoordinate(1, 1), 1),
                        new SparseMatrix.Entry(new MatrixCoordinate(3, 2), 1),
                        new SparseMatrix.Entry(new MatrixCoordinate(2, 3), 9),
                        new SparseMatrix.Entry(new MatrixCoordinate(1, 2), 2)));
        //verify that B2 = A
        System.out.println(String.format(
                "B2 = A, %b",
                MatrixPropertyUtils.areEqual(B2, A, 1e-15)));

        // LIL
        SparseMatrix C1 = new LILSparseMatrix(3, 4, // 3x4 dimension
                new int[]{3, 2, 1, 3, 2, 1}, // row indices
                new int[]{3, 2, 1, 2, 3, 2}, // column indices
                new double[]{4, 3, 1, 1, 9, 2} // matrix entries/values
        );
        //verify that C1 = A
        System.out.println(String.format(
                "C1 = A, %b",
                MatrixPropertyUtils.areEqual(C1, A, 1e-15)));
        SparseMatrix C2 = new LILSparseMatrix(3, 4, // 3x4 dimension
                Arrays.<SparseMatrix.Entry>asList( // specify only the non-zero entries
                        new SparseMatrix.Entry(new MatrixCoordinate(3, 3), 4),
                        new SparseMatrix.Entry(new MatrixCoordinate(2, 2), 3),
                        new SparseMatrix.Entry(new MatrixCoordinate(1, 1), 1),
                        new SparseMatrix.Entry(new MatrixCoordinate(3, 2), 1),
                        new SparseMatrix.Entry(new MatrixCoordinate(2, 3), 9),
                        new SparseMatrix.Entry(new MatrixCoordinate(1, 2), 2)));
        //verify that C2 = A
        System.out.println(String.format(
                "C2 = A, %b",
                MatrixPropertyUtils.areEqual(C2, A, 1e-15)));

        // CSR
        SparseMatrix D1 = new CSRSparseMatrix(3, 4, // 3x4 dimension
                new int[]{3, 2, 1, 3, 2, 1}, // row indices
                new int[]{3, 2, 1, 2, 3, 2}, // column indices
                new double[]{4, 3, 1, 1, 9, 2} // matrix entries/values
        );
        //verify that D1 = A
        System.out.println(String.format(
                "D1 = A, %b",
                MatrixPropertyUtils.areEqual(D1, A, 1e-15)));
        SparseMatrix D2 = new CSRSparseMatrix(3, 4, // 3x4 dimension
                Arrays.<SparseMatrix.Entry>asList( // specify only the non-zero entries
                        new SparseMatrix.Entry(new MatrixCoordinate(3, 3), 4),
                        new SparseMatrix.Entry(new MatrixCoordinate(2, 2), 3),
                        new SparseMatrix.Entry(new MatrixCoordinate(1, 1), 1),
                        new SparseMatrix.Entry(new MatrixCoordinate(3, 2), 1),
                        new SparseMatrix.Entry(new MatrixCoordinate(2, 3), 9),
                        new SparseMatrix.Entry(new MatrixCoordinate(1, 2), 2)));
        //verify that D2 = A
        System.out.println(String.format(
                "D2 = A, %b",
                MatrixPropertyUtils.areEqual(D2, A, 1e-15)));

        // sparse vector construction
        SparseVector v1 = new SparseVector(
                99, // vector size
                new int[]{1, 3, 53, 79, 99}, // indices
                new double[]{11, 22, 33, 44, 55} // values
        );
        System.out.println("v = " + v1);

        // addition
        Matrix M1 = B1.add(A);
        System.out.println("M1 = " + M1);

        DOKSparseMatrix M2 = (DOKSparseMatrix) B1.add(B2);
        System.out.println("M2 = " + M2);

        LILSparseMatrix M3 = (LILSparseMatrix) C1.minus(C2);
        System.out.println("M3 = " + M3);

        CSRSparseMatrix M4 = (CSRSparseMatrix) D1.multiply(D2.t());
        System.out.println("M4 = " + M4);

        SparseVector v2 = new SparseVector(
                4, // vector size
                new int[]{1}, // indices
                new double[]{11} // values
        );
        System.out.println("v2 = " + v2);
        Vector v3 = B1.multiply(v2);
        System.out.println("ve = " + v3);
    }

    public void overdetermined_system() {
        System.out.println("solving an over-determined system");

        // define an overdetermined system of linear equations
        Matrix A = new DenseMatrix(new double[][]{
            {1, 1},
            {1, 2},
            {1, 3},
            {1, 4}
        });
        Vector b = new DenseVector(new double[]{6, 5, 7, 10});
        LSProblem problem = new LSProblem(A, b);

        // solve the system using QR method
        OLSSolverByQR solver1 = new OLSSolverByQR(0); // precision
        // compute the OLS solution
        Vector x1 = solver1.solve(problem);
        System.out.println("the OLS solution = " + x1);

        // solve the system using SVD method
        OLSSolverBySVD solver2 = new OLSSolverBySVD(0); // precision
        // compute the OLS solution
        Vector x2 = solver2.solve(problem);
        // verify that Ax2 = b
        Vector Ax2 = A.multiply(x2);
        System.out.println("the OLS solution = " + x2);
    }

    public void linear_system_solver() {
        System.out.println("solving system of linear equations ");

        Matrix A = new DenseMatrix(new double[][]{
            {0, 1, 2, -1},
            {1, 0, 1, 1},
            {-1, 1, 0, -1},
            {0, 2, 3, -1}
        });
        Vector b = new DenseVector(new double[]{1, 4, 2, 7});

        // construct a linear system solver
        LinearSystemSolver solver = new LinearSystemSolver(1e-15); // precision
        // solve the homogenous linear system
        LinearSystemSolver.Solution soln = solver.solve(A);
        // get a particular solution
        Vector p = soln.getParticularSolution(b);
        System.out.println("p = \n" + p);

        // verify that Ap = b
        Vector Ap = A.multiply(p);
        System.out.println(String.format("%s = \n%s is %b",
                Ap,
                b,
                MatrixPropertyUtils.areEqual(Ap, b, 1e-15)));

        // get the basis for the null-space
        List<Vector> kernel = soln.getHomogeneousSoln();
        System.out.println("kernel size = " + kernel.size());

        // verify that A * kernel = 0
        Vector k = kernel.get(0);
        System.out.println("kernel basis = " + k);
        Vector Ak = A.multiply(k);
        System.out.println("Ak = 0, " + Ak);

    }

    public void gauss_jordan_elimination() {
        Matrix A = new DenseMatrix(new double[][]{
            {2, 1, 1},
            {2, 2, -1},
            {4, -1, 6}
        });
        GaussJordanElimination ops = new GaussJordanElimination(A, true, 0);

        Matrix U = ops.U();
        System.out.println(
                String.format("U = %s is in reduced row echelon form, %b",
                        U,
                        MatrixPropertyUtils.isReducedRowEchelonForm(U, 0)));

        Matrix T = ops.T();
        System.out.println("T = \n" + T);

        // verify that TA = U
        Matrix TA = T.multiply(A);
        System.out.println(String.format("%s = \n%s is %b",
                TA,
                U,
                MatrixPropertyUtils.areEqual(TA, U, 1e-15)));
    }

    public void gaussian_elimination() {
        System.out.println("Gaussian Elimination");

        Matrix A = new DenseMatrix(new double[][]{
            {2, 1, 1},
            {2, 2, -1},
            {4, -1, 6}
        });
        GaussianElimination ops = new GaussianElimination(A, true, 0);

        Matrix T = ops.T();
        System.out.println("T = \n" + T);
        PermutationMatrix P = ops.P();
        System.out.println("P = \n" + P);

        Matrix U = ops.U();
        System.out.println(
                String.format("U = %s is upper triangular, %b",
                        U,
                        MatrixPropertyUtils.isUpperTriangular(U, 0)));

        Matrix L = ops.L();
        System.out.println(
                String.format("L = %s is lower triangular, %b",
                        L,
                        MatrixPropertyUtils.isLowerTriangular(L, 0)));

        // verify that TA = U
        Matrix TA = T.multiply(A);
        System.out.println(String.format("%s = \n%s is %b",
                TA,
                U,
                MatrixPropertyUtils.areEqual(TA, U, 1e-15)));

        // verify that PA = LU
        Matrix PA = P.multiply(A);
        Matrix LU = L.multiply(U);
        System.out.println(String.format("%s = \n%s is %b",
                PA,
                LU,
                MatrixPropertyUtils.areEqual(PA, LU, 0)));
    }

    public void LU_solver() {
        System.out.println("LU solver");

        // an LSProblem
        LowerTriangularMatrix L = new LowerTriangularMatrix(new double[][]{
            {1},
            {2, 3},
            {4, 5, 6}
        });
        Vector b1 = new DenseVector(new double[]{10, 20, 30});

        LUSolver solver1 = new LUSolver();
        Vector x1 = solver1.solve(
                // construct a Linear System Problem: Lx = b1
                new LSProblem(L, b1)
        );

        System.out.println("x1 = " + x1);

        // verify that Ux = b
        Vector Lx = L.multiply(x1);
        System.out.println(String.format("%s = \n%s is %b",
                Lx,
                b1,
                MatrixPropertyUtils.areEqual(Lx, b1, 1e-14))); // MatrixPropertyUtils.areEqual works for vectors too

        // an other LSProblem
        UpperTriangularMatrix U = new UpperTriangularMatrix(new double[][]{
            {1, 2, 3},
            {0, 5},
            {0}
        });
        Vector b2 = new DenseVector(new double[]{10, 0, 0});

        LUSolver solver2 = new LUSolver();
        Vector x2 = solver2.solve(
                // construct a Linear System Problem: Ux = b2
                new LSProblem(U, b2)
        );
        System.out.println("x2 = " + x2);

        // verify that Ux = b
        Vector Ux = U.multiply(x2);
        System.out.println(String.format("%s = \n%s is %b",
                Ux,
                b2,
                MatrixPropertyUtils.areEqual(Ux, b2, 1e-14))); // MatrixPropertyUtils.areEqual works for vectors too
    }

    public void forward_substitution() {
        System.out.println("forward substitution");

        LowerTriangularMatrix L = new LowerTriangularMatrix(new double[][]{
            {1},
            {2, 3},
            {4, 5, 6}
        });
        Vector b = new DenseVector(new double[]{10, 20, 30});

        ForwardSubstitution solver = new ForwardSubstitution();
        Vector x = solver.solve(L, b);
        System.out.println("x = " + x);

        // verify that Ux = b
        Vector Lx = L.multiply(x);
        System.out.println(String.format("%s = \n%s is %b",
                Lx,
                b,
                MatrixPropertyUtils.areEqual(Lx, b, 1e-14))); // MatrixPropertyUtils.areEqual works for vectors too
    }

    public void backward_substitution() {
        System.out.println("backward substitution");

        UpperTriangularMatrix U = new UpperTriangularMatrix(new double[][]{
            {1, 2, 3},
            {0, 5},
            {0}
        });
        Vector b = new DenseVector(new double[]{10, 0, 0});

        BackwardSubstitution solver = new BackwardSubstitution();
        Vector x = solver.solve(U, b);
        System.out.println("x = " + x);

        // verify that Ux = b
        Vector Ux = U.multiply(x);
        System.out.println(String.format("%s = \n%s is %b",
                Ux,
                b,
                MatrixPropertyUtils.areEqual(Ux, b, 1e-14))); // MatrixPropertyUtils.areEqual works for vectors too
    }

    public void svd() {
        System.out.println("singular value decomposition");

        Matrix A = new DenseMatrix(new double[][]{
            {1, 0, 0, 0, 2},
            {0, 0, 3, 0, 0,},
            {0, 0, 0, 0, 0,},
            {0, 2, 0, 0, 0,}
        });

        // perform SVD
        SVD svd = new SVD(A, true, 1e-15);

        Matrix U = svd.U();
        System.out.println("U = \n" + U);
        DiagonalMatrix D = svd.D();
        System.out.println("D = \n" + D);
        Matrix V = svd.V();
        System.out.println("Vt = \n" + V.t());
        Matrix Ut = svd.Ut();
        System.out.println("Ut = \n" + Ut);

        // verify that UDVt = A
        Matrix UDVt = U.multiply(D).multiply(V.t());
        System.out.println(String.format("%s = \n%s is %b",
                A,
                UDVt,
                MatrixPropertyUtils.areEqual(UDVt, A, 1e-14)));

        // verify that UtAV = D
        Matrix UtAV = Ut.multiply(A).multiply(V);
        System.out.println(String.format("%s = \n%s is %b",
                A,
                UtAV,
                MatrixPropertyUtils.areEqual(UtAV, D, 1e-14)));
    }

    public void eigen_decomposition_0020() {
        System.out.println("eigen decomposition");

        Matrix A = new DenseMatrix(new double[][]{
            {1, -3, 3},
            {3, -5, 3},
            {6, -6, 4}
        });

        // perform an eigen decomposition
        Eigen eigen = new Eigen(A);
        eigen.getEigenvalues().forEach(eigenvalue -> {
            System.out.println("eigen value = " + eigenvalue);
        });

        // the first eigenvalue
        Number eigenvalue0 = eigen.getEigenvalue(0); // count from 0
        System.out.println("eigenvalue0 = " + eigenvalue0);
        // get the properties associated with this eigenvalue
        EigenProperty prop0 = eigen.getProperty(eigenvalue0);
        System.out.println("algebraic multiplicity = " + prop0.algebraicMultiplicity());
        System.out.println("geometric multiplicity = " + prop0.geometricMultiplicity());
        List<Vector> basis0 = prop0.eigenbasis();
        basis0.forEach(v -> {
            System.out.println("basis vector = " + v);
        });
        RealVectorSpace vs0 = new RealVectorSpace(basis0, 1e-15);
        // check if this vector belongs to the vector space, i.e., a linear combination of the basis

        boolean in0 = vs0.isSpanned(
                new DenseVector(new double[]{-0.4, -0.4, -0.8}));
        System.out.println("is in the vector space = " + in0);

        // the second eigenvalue
        Number eigenvalue1 = eigen.getEigenvalue(1);
        System.out.println("eigenvalue1 = " + eigenvalue1.doubleValue());
        EigenProperty prop1 = eigen.getProperty(eigenvalue1);
        System.out.println("algebraic multiplicity = " + prop1.algebraicMultiplicity());
        System.out.println("geometric multiplicity = " + prop1.geometricMultiplicity());
        List<Vector> basis1 = prop1.eigenbasis();
        basis1.forEach(v -> {
            System.out.println("basis vector = " + v);
        });
        RealVectorSpace vs1 = new RealVectorSpace(basis1, 1e-15);
        boolean in1 = vs1.isSpanned(
                new DenseVector(new double[]{-0.4, 0.4, 0.8}));
        System.out.println("is in the vector space = " + in1);
        boolean in2 = vs1.isSpanned(
                new DenseVector(new double[]{-0.5, 0.5, 1.0}));
        System.out.println("is in the vector space = " + in2);
    }

    public void eigen_decomposition_0010() {
        System.out.println("eigen decomposition");

        Matrix A = new DenseMatrix(new double[][]{
            {5, 2},
            {2, 5}
        });

        // doing eigen decomposition
        EigenDecomposition eigen = new EigenDecomposition(A);

        Matrix D = eigen.D();
        System.out.println("D = \n" + D);
        Matrix Q = eigen.Q();
        System.out.println("Q = \n" + Q);
        Matrix Qt = eigen.Qt();
        System.out.println("Qt = \n" + Qt);

        // verify that QDQt = A
        Matrix QDQt = Q.multiply(D).multiply(Qt);
        System.out.println(String.format("%s = \n%s is %b",
                A,
                QDQt,
                MatrixPropertyUtils.areEqual(A, QDQt, 1e-14)));
    }

    public void qr_0020() {
        System.out.println("QR decomposition of a tall matrix");

        Matrix A = new DenseMatrix(new double[][]{
            {1, 2, 3},
            {6, 7, 8},
            {11, 12, 13},
            {16, 17, 18},
            {21, 22, 23}
        });

        QRDecomposition qr = new QR(A, 1e-14);
        System.out.println("rank = " + qr.rank());

        Matrix Q1 = qr.Q();
        System.out.println("Q1 = \n" + Q1);
        UpperTriangularMatrix R1 = qr.R();
        System.out.println("R1 = \n" + R1);
        // verify that Q1R1 = A
        Matrix Q1R1 = Q1.multiply(R1);
        System.out.println(String.format("%s = \n%s is %b",
                A,
                Q1R1,
                MatrixPropertyUtils.areEqual(Q1R1, A, 1e-13)));

        Matrix Q = qr.squareQ();
        System.out.println("Q = \n" + Q);
        Matrix R = qr.tallR();
        System.out.println("R = \n" + R);
        // verify that QR = A
        Matrix QR = Q.multiply(R);
        System.out.println(String.format("%s = \n%s is %b",
                A,
                QR,
                MatrixPropertyUtils.areEqual(QR, A, 1e-13)));
    }

    public void qr_0010() {
        System.out.println("QR decomposition of a square matrix");

        Matrix A = new DenseMatrix(new double[][]{
            {3, 2},
            {1, 2}
        });

        // use the Householder QR algorithm
        QRDecomposition qr1 = new HouseholderQR(A, 0);
        System.out.println("rank = " + qr1.rank());

        Matrix Q1 = qr1.Q();
        System.out.println(String.format(
                "Q = \n%s is orthogonal, %b",
                Q1,
                MatrixPropertyUtils.isOrthogonal(Q1, 1e-15)));

        UpperTriangularMatrix R1 = qr1.R();
        System.out.println("R = \n" + R1);

        // verify that Q1R1 = A
        Matrix Q1R1 = Q1.multiply(R1);
        System.out.println(String.format("%s = \n%s is %b",
                A,
                Q1R1,
                MatrixPropertyUtils.areEqual(Q1R1, A, 1e-13)));

        // use the Gram-Schmidt QR algorithm
        QRDecomposition qr2 = new GramSchmidt(A);
        System.out.println("rank = " + qr2.rank());

        Matrix Q2 = qr2.Q();
        System.out.println(String.format(
                "Q = \n%s is orthogonal, %b",
                Q2,
                MatrixPropertyUtils.isOrthogonal(Q2, 1e-15)));

        UpperTriangularMatrix R2 = qr2.R();
        System.out.println("R = \n" + R2);

        // verify that Q2R2 = A
        Matrix Q2R2 = Q2.multiply(R2);
        System.out.println(String.format("%s = \n%s is %b",
                A,
                Q2R2,
                MatrixPropertyUtils.areEqual(Q2R2, A, 1e-13)));
    }

    public void tridiagonalization() {
        System.out.println("tri-diagonalization");

        // define a symmetric matrix
        Matrix S = new SymmetricMatrix(new double[][]{
            {1},
            {5, 0},
            {7, 3, 1},
            {9, 13, 2, 10}});
//        Matrix S = new DenseMatrix(new double[][]{
//            {1, 5, 7, 9},
//            {5, 0, 3, 13},
//            {7, 3, 1, 2},
//            {9, 13, 2, 10}
//        });

        TriDiagonalization triDiagonalization
                = new TriDiagonalization(S);

        Matrix T = triDiagonalization.T();
        System.out.println(String.format(
                "T = \n%s is tri-diagonal, %b",
                T,
                MatrixPropertyUtils.isTridiagonal(T, 1e-14)));

        Matrix Q = triDiagonalization.Q();
        System.out.println(String.format(
                "Q = \n%s is tri-diagonal, %b",
                Q,
                MatrixPropertyUtils.isOrthogonal(Q, 1e-14)));

        // verify that Qt * A * Q = T
        Matrix QtSQ = Q.t().multiply(S).multiply(Q);
        System.out.println(String.format("%s = \n%s is %b",
                T,
                QtSQ,
                MatrixPropertyUtils.areEqual(QtSQ, T, 1e-13)));
    }

    public void hessenberg() {
        System.out.println("Hessenberg decomposition");

        Matrix A = new DenseMatrix(new double[][]{
            {1, 5, 7, 9},
            {3, 0, 6, 3},
            {4, 3, 1, 0},
            {7, 13, 2, 10}
        });
        HessenbergDecomposition hessenberg
                = new HessenbergDecomposition(A);

        Matrix H = hessenberg.H();
        System.out.println(String.format(
                "H = \n%s is Hessenberg, %b",
                H,
                Hessenberg.isHessenberg(H, 0)));

        Matrix Q = hessenberg.Q();
        System.out.println(String.format(
                "Q = \n%s is orthogonal, %b",
                Q,
                MatrixPropertyUtils.isOrthogonal(Q, 1e-14)));

        // verify that Qt * A * Q = H
        Matrix QtAQ = Q.t().multiply(A).multiply(Q);
        System.out.println(String.format("%s = \n%s is %b",
                H,
                QtAQ,
                MatrixPropertyUtils.isOrthogonal(Q, 1e-14)));
    }

    public void cholesky() {
        System.out.println("Cholesky decomposition");

        Matrix A = new DenseMatrix(new double[][]{
            {4, 12, -16},
            {12, 37, -43},
            {-16, -43, 98}
        });
        Cholesky chol = new Chol(A);
        LowerTriangularMatrix L = chol.L();
        UpperTriangularMatrix Lt = chol.L().t();
        System.out.println(String.format("L = %s", L));
        System.out.println(String.format("Lt = %s", Lt));

        Matrix LLt = L.multiply(Lt);
        // verify that A = LLt
        System.out.println(String.format("%s = \n%s is %b",
                A,
                LLt,
                MatrixPropertyUtils.areEqual(A, LLt, 1e-14)));
    }

    public void lu() {
        System.out.println("LU decomposition1");

        Matrix A = new DenseMatrix(new double[][]{
            {1, 2, 3},
            {4, 5, 6},
            {7, 8, 9}
        });
        // perform LU decomposition
        LU lu = new LU(A);
        LowerTriangularMatrix L = lu.L();
        UpperTriangularMatrix U = lu.U();
        PermutationMatrix P = lu.P();

        System.out.println(String.format("P = %s", P));
        System.out.println(String.format("L = %s", L));
        System.out.println(String.format("U = %s", U));

        Matrix PA = P.multiply(A);
        Matrix LU = L.multiply(U);
        // verify that PA = LU
        System.out.println(String.format("%s = \n%s is %b",
                PA,
                LU,
                MatrixPropertyUtils.areEqual(PA, LU, 1e-14)));
    }

    public void kronecker() {
        System.out.println("Kronecker products");

        Matrix A1 = new DenseMatrix(new double[][]{
            {1, 2},
            {3, 4}
        });
        Matrix B1 = new DenseMatrix(new double[][]{
            {0, 5},
            {6, 7}
        });
        Matrix C1 = new KroneckerProduct(A1, B1);
        System.out.println(String.format("%s ⊗ \n%s = \n%s", A1, B1, C1));

        Matrix A2 = new DenseMatrix(new double[][]{
            {1, -4, 7},
            {-2, 3, 3}
        });
        Matrix B2 = new DenseMatrix(new double[][]{
            {8, -9, -6, 5},
            {1, -3, -4, 7},
            {2, 8, -8, -3},
            {1, 2, -5, -1}
        });
        Matrix C2 = new KroneckerProduct(A2, B2);
        System.out.println(String.format("%s ⊗ \n%s = \n%s", A2, B2, C2));
    }

    public void determinant() {
        System.out.println("computing determinants");

        Matrix M1 = new DenseMatrix(new double[][]{
            {2, 1, 2},
            {3, 2, 2},
            {1, 2, 3}
        });

        // calculate the determinant of matrix M1
        double det = MatrixMeasure.det(M1);
        System.out.println(det);
    }

    public void rank() {
        System.out.println("computing ranks");

        final double PRECISION = 1e-15;
        Matrix M1 = new DenseMatrix(new double[][]{
            {1, 0, 1},
            {-2, -3, 1},
            {3, 3, 0}
        });
        // calculate the rank of M1, treating numbers smaller than PRECISION as 0
        int rank1 = MatrixMeasure.rank(M1, PRECISION);
        System.out.println(rank1);
        Matrix M2 = new DenseMatrix(new double[][]{
            {1, 1, 0, 2},
            {-1, -1, 0, -2}
        });
        // calculate the rank of M2, treating numbers smaller than PRECISION as 0
        int rank2 = MatrixMeasure.rank(M2, PRECISION);
        System.out.println(rank2);
    }

    public void multiplication() {
        System.out.println("matrix multiplication");

        Matrix m1 = new DenseMatrix(
                new double[][]{
                    {1, 2, 3},
                    {4, 5, 6},
                    {7, 8, 9}
                });
        Matrix m1t = m1.t();
        Matrix m1tm1 = m1t.multiply(m1);
        System.out.println(String.format("%s * \n%s = \n%s", m1t, m1, m1tm1));
    }

    public void vectors() {
        System.out.println("vectors");

        Vector v1 = new DenseVector(new double[]{1.0, 2.0}); // define a vector
        System.out.println(String.format("v1 = %s", v1)); // print out v1

        int size = v1.size(); // 2
        System.out.println(String.format("v1 size = %d", size));
        double v1_1 = v1.get(1); // 1.0
        System.out.println(String.format("v1_1 = %f", v1_1));
        v1.set(2, -2.0); // v1 becomes (1.0, -2.0)
        System.out.println(String.format("v1 = %s", v1)); // print out v1

        Vector v2 = new DenseVector(new double[]{-2.0, 1.0}); // define another vector

        // addition
        Vector a1 = v1.add(0.1); // add 0.1 to all entries
        System.out.println(String.format("a1 = %s", a1)); // print out a1
        Vector a2 = v1.add(v2); // a2 = v1 + v2, entry by entry
        System.out.println(String.format("a2 = %s", a2)); // print out a2

        // subtraction
        Vector m1 = v1.minus(0.1); // subtract 0.1 from all entries
        System.out.println(String.format("m1 = %s", m1)); // print out m1
        Vector m2 = v1.minus(v2); // v1 – v2, entry by entry
        System.out.println(String.format("m2 = %s", m2)); // print out m2

        // scaling, multiplication, division
        Vector s1 = v1.scaled(0.5); // multiply all entries by 0.5
        System.out.println(String.format("s1 = %s", s1)); // print out s1

        // multiplication
        Vector t1 = v1.multiply(v2); // multiply v1 by v2, entry by entry
        System.out.println(String.format("t1 = %s", t1)); // print out t1

        // division
        Vector d1 = v1.divide(v2); // divide v1 by v2, entry by entry
        System.out.println(String.format("d1 = %s", d1)); // print out d1

        // power
        Vector p1 = v1.pow(2.0); // take to the square, entry-wise
        System.out.println(String.format("p1 = %s", p1)); // print out p1

        // norm
        Vector v3 = new DenseVector(new double[]{3., 4.});
        double norm1 = v3.norm(1); // l1 norm = (3. + 4.) = 7.
        System.out.println(String.format("L1 norm = %f", norm1));
        double norm2 = v3.norm(); // l2 norm = sqrt(3^2 + 4^2) = 5, Pythagorean theorem
        System.out.println(String.format("L2 norm = %f", norm2));

        // inner product and angle
        Vector v4 = new DenseVector(new double[]{1., 2.});
        Vector v5 = new DenseVector(new double[]{3., 4.});
        double dot = v4.innerProduct(v5); // (1.*3. + 2.*4.) = 11.
        System.out.println(String.format("<%s,%s> = %f", v4, v5, dot));
        double angle = v4.angle(v5);
        System.out.println(String.format("angle between %s and %s = %f", v4, v5, angle));
    }

    public void inverse() {
        // compute the inverse of an invertible matrix
        Matrix A = new DenseMatrix(
                new double[][]{
                    {1, 2, 3},
                    {6, 5, 4},
                    {8, 7, 9}
                });
        Matrix Ainv = new Inverse(A);
        Matrix I = A.multiply(Ainv);
        System.out.println(String.format("%s * \n%s = \n%s", A, Ainv, I));

        // compute the inverse of an invertible matrix
        Matrix A1 = new DenseMatrix(
                new double[][]{
                    {-1, 3. / 2},
                    {1, -1}
                });
        double det1 = MatrixMeasure.det(A1);
        System.out.println("det of A1 = " + det1);
        Matrix Ainv1 = new Inverse(A1);
        Matrix I1 = A1.multiply(Ainv1);
        System.out.println(String.format("%s * \n%s = \n%s", A1, Ainv1, I1));

        // compute the pseudo-inverse of a non-invertible matrix
        Matrix A2 = new DenseMatrix(new double[][]{
            {2, 4},
            {5, 10}
        });
        double det2 = MatrixMeasure.det(A2);
        System.out.println("det of A2 = " + det2);
        PseudoInverse A2p = new PseudoInverse(A2);
        System.out.println("the pseudo inverse is");
        System.out.println(A2p);
        // the first property of pseudo-inverse
        System.out.println("should be the same as the matrix");
        Matrix A2_copy = A2.multiply(A2p).multiply(A2);
        System.out.println(A2_copy);
        // the second property of pseudo-inverse
        System.out.println("should be the same as the pseudo inverse");
        Matrix A2p_copy = A2p.multiply(A2).multiply(A2p);
        System.out.println(A2p_copy);

        // compute the pseudo-inverse of a non-square matrix
        Matrix A3 = new DenseMatrix(new double[][]{
            {1, 0},
            {0, 1},
            {0, 1}
        });
        PseudoInverse A3p = new PseudoInverse(A3);
        System.out.println("the pseudo inverse is");
        System.out.println(A3p);
        // the first property of pseudo-inverse
        System.out.println("should be the same as the matrix");
        Matrix A3_copy = A3.multiply(A3p).multiply(A3);
        System.out.println(A3_copy);
        // the second property of pseudo-inverse
        System.out.println("should be the same as the pseudo inverse");
        Matrix A3p_copy = A3p.multiply(A3).multiply(A3p);
        System.out.println(A3p_copy);
    }

    public void transpose() {
        System.out.println("matrix transpose");

        Matrix m1 = new DenseMatrix(
                new double[][]{
                    {1, 2, 3},
                    {4, 5, 6},
                    {7, 8, 9}
                });
        System.out.println(String.format("m1 = %s", m1)); // print out the matrix m1
        Matrix m1t = m1.t(); // doing a transpose
        System.out.println(String.format("m1t = %s", m1t)); // print out the transpose matrix
    }

    public void matrices_0020() {
        System.out.println("matrix arithmetic");

        Matrix m1 = new DenseMatrix(
                new double[][]{
                    {1, 2, 3},
                    {4, 5, 6},
                    {7, 8, 9}
                });

        // m2 = -m1
        Matrix m2 = m1.scaled(-1);

        // m3 = m1 + m2 = m1 - m1 = 0
        Matrix m3 = m1.add(m2);

        // not a recommended usage
        boolean isEqual1 = m3.equals(m3.ZERO());
        System.out.println(isEqual1);

        // recommended usage
        boolean isEqual2 = MatrixPropertyUtils.areEqual(m3, m3.ZERO(), 1e-16);
        System.out.println(isEqual2);
    }

    public void matrices_0010() {
        System.out.println("matrices");

        // matrix construction
        Matrix m1 = new DenseMatrix( // define a matrix
                new double[][]{
                    {1, 2, 3},
                    {4, 5, 6},
                    {7, 8, 9}
                });
        Matrix m2 = new DenseMatrix(m1); // copy the matrix m1
        System.out.println(String.format("m1 = %s", m1)); // print out the matrix m1
        System.out.println(String.format("m2 = %s", m2)); // print out the matrix m2

        // getters and setters
        double m1_11 = m1.get(1, 1);
        double m1_22 = m1.get(2, 2);
        double m1_33 = m1.get(3, 3);
        System.out.println(String.format("the diagonal entries are: %f, %f, %f", m1_11, m1_22, m1_33));

        // changeing the diagonal to 0
        System.out.println("changing the diagonal to 0");
        m1.set(1, 1, 0.);
        m1.set(2, 2, 0.);
        m1.set(3, 3, 0.);
        System.out.println(String.format("m1 = %s", m1)); // print out the matrix m1

        // create a vector
        Vector v = new DenseVector(new double[]{1, 2, 3});
        // create a column matrix from a vector
        Matrix m3 = new DenseMatrix(v);
        System.out.println(String.format("m3 = %s", m3)); // print out the matrix m3
    }

}
