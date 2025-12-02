package utils;

/**
 * Matrix operations.
 */
public class MatrixUtils {
    
    /**
     * Transpose a matrix (rows become columns)
     * Input: m x n matrix
     * Output: n x m matrix
     */
    public static double[][] transpose(double[][] matrix) {
        int rows = matrix.length;
        int cols = matrix[0].length;
        double[][] result = new double[cols][rows];
        
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                result[j][i] = matrix[i][j];
            }
        }
        return result;
    }
    
    /**
     * Multiply two matrices: A * B
     */
    public static double[][] multiply(double[][] A, double[][] B) {
        int m = A.length;
        int n = A[0].length;
        int p = B[0].length;
        
        if (n != B.length) {
            throw new IllegalArgumentException("Matrix dimensions don't match for multiplication");
        }
        
        double[][] result = new double[m][p];
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < p; j++) {
                result[i][j] = 0;
                for (int k = 0; k < n; k++) {
                    result[i][j] += A[i][k] * B[k][j];
                }
            }
        }
        return result;
    }
    
    /**
     * Multiply matrix by vector: A * v
     */
    public static double[] multiplyVector(double[][] A, double[] v) {
        int m = A.length;
        int n = A[0].length;
        
        if (n != v.length) {
            throw new IllegalArgumentException("Matrix and vector dimensions don't match");
        }
        
        double[] result = new double[m];
        for (int i = 0; i < m; i++) {
            result[i] = 0;
            for (int j = 0; j < n; j++) {
                result[i] += A[i][j] * v[j];
            }
        }
        return result;
    }
    
    /**
     * Add a scalar to each element of a vector
     */
    public static double[] addScalar(double[] v, double scalar) {
        double[] result = new double[v.length];
        for (int i = 0; i < v.length; i++) {
            result[i] = v[i] + scalar;
        }
        return result;
    }
    
    /**
     * Invert a matrix
     */
    public static double[][] invert(double[][] matrix) {
        int n = matrix.length;
        
        // Create augmented matrix [A | I]
        double[][] augmented = new double[n][2 * n];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                augmented[i][j] = matrix[i][j];
            }
            augmented[i][n + i] = 1.0; 
        }
        
        // Forward elimination
        for (int i = 0; i < n; i++) {
            // Find pivot
            int maxRow = i;
            for (int k = i + 1; k < n; k++) {
                if (Math.abs(augmented[k][i]) > Math.abs(augmented[maxRow][i])) {
                    maxRow = k;
                }
            }
            
            // Swap rows
            double[] temp = augmented[i];
            augmented[i] = augmented[maxRow];
            augmented[maxRow] = temp;
            
            // Check for singular matrix
            if (Math.abs(augmented[i][i]) < 1e-10) {
                throw new RuntimeException("Matrix is singular and cannot be inverted");
            }
            
            // Scale pivot row
            double pivot = augmented[i][i];
            for (int j = 0; j < 2 * n; j++) {
                augmented[i][j] /= pivot;
            }
            
            // Eliminate column
            for (int k = 0; k < n; k++) {
                if (k != i) {
                    double factor = augmented[k][i];
                    for (int j = 0; j < 2 * n; j++) {
                        augmented[k][j] -= factor * augmented[i][j];
                    }
                }
            }
        }
        
        // Extract the inverse matrix from the right side
        double[][] inverse = new double[n][n];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                inverse[i][j] = augmented[i][n + j];
            }
        }
        
        return inverse;
    }
    
    /**
     * Add an intercept column to the left of the matrix
     */
    public static double[][] addInterceptColumn(double[][] X) {
        int rows = X.length;
        int cols = X[0].length;
        double[][] result = new double[rows][cols + 1];
        
        for (int i = 0; i < rows; i++) {
            result[i][0] = 1.0; 
            for (int j = 0; j < cols; j++) {
                result[i][j + 1] = X[i][j];
            }
        }
        return result;
    }
    
    /**
     * Print matrix dimensions 
     */
    public static void printShape(double[][] matrix, String name) {
        System.out.println(name + " shape: [" + matrix.length + " x " + matrix[0].length + "]");
    }
}
