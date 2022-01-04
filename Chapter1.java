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
import dev.nm.algebra.linear.matrix.doubles.matrixtype.dense.DenseMatrix;
import dev.nm.algebra.linear.matrix.doubles.operation.Inverse;
import dev.nm.algebra.linear.vector.doubles.Vector;
import dev.nm.algebra.linear.vector.doubles.dense.DenseVector;

/**
 * Numerical Methods Using Java: For Data Science, Analysis, and Engineering
 *
 * @author haksunli
 * @see
 * https://www.amazon.com/Numerical-Methods-Using-Java-Engineering/dp/1484267966
 * https://nm.dev/
 */
public class Chapter1 {

    public static void main(String[] args) {
        System.out.println("Chapter 1 demos");

        Chapter1 chapter1 = new Chapter1();

        chapter1.invert_matrix();
        chapter1.innerProductDemo();
        chapter1.a_vector();
    }

    public void a_vector() {
        Vector v = new DenseVector(1., 2., 3.);
        System.out.println(v);
    }

    public void innerProductDemo() {
        System.out.println("dot product of two vectors");

        Vector v1 = new DenseVector(new double[]{1., 2.});
        Vector v2 = new DenseVector(new double[]{3., 4.});
        double product = v1.innerProduct(v2);
        System.out.println(product);
    }

    public void invert_matrix() {
        System.out.println("invert a matrix");

        Matrix A = new DenseMatrix(
                new double[][]{
                    {1, 2, 3},
                    {6, 5, 4},
                    {8, 7, 9}
                });

        Matrix Ainv = new Inverse(A);
        System.out.println(Ainv);
    }
}
