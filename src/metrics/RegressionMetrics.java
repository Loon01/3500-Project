package metrics;

/**
 * Metrics for evaluating regression models.
 */
public class RegressionMetrics {
    
    /**
     * Calculate Root Mean Squared Error (RMSE)
     * RMSE = sqrt(mean((y_true - y_pred)^2))
     */
    public static double rmse(double[] yTrue, double[] yPred) {
        if (yTrue.length != yPred.length) {
            throw new IllegalArgumentException("Arrays must have the same length");
        }
        
        double sumSquaredError = 0.0;
        for (int i = 0; i < yTrue.length; i++) {
            double error = yTrue[i] - yPred[i];
            sumSquaredError += error * error;
        }
        
        return Math.sqrt(sumSquaredError / yTrue.length);
    }
    
    /**
     * Calculate R^2 (R-squared, coefficient of determination)
     * R^2 = 1 - (SS_res / SS_tot)
     * Range: (-infinity, 1], where 1 = perfect predictions, 0 = mean baseline
     */
    public static double r2Score(double[] yTrue, double[] yPred) {
        if (yTrue.length != yPred.length) {
            throw new IllegalArgumentException("Arrays must have the same length");
        }
        
        // Calculate mean of true values
        double mean = 0.0;
        for (double val : yTrue) {
            mean += val;
        }
        mean /= yTrue.length;
        
        // Calculate SS_tot (total sum of squares)
        double ssTot = 0.0;
        for (double val : yTrue) {
            double diff = val - mean;
            ssTot += diff * diff;
        }
        
        // Calculate SS_res (residual sum of squares)
        double ssRes = 0.0;
        for (int i = 0; i < yTrue.length; i++) {
            double diff = yTrue[i] - yPred[i];
            ssRes += diff * diff;
        }
        
        // Avoid division by zero
        if (ssTot < 1e-10) {
            return 0.0;
        }
        
        return 1.0 - (ssRes / ssTot);
    }
    
    /**
     * Calculate Mean Absolute Error (MAE) - bonus metric
     * MAE = mean(|y_true - y_pred|)
     */
    public static double mae(double[] yTrue, double[] yPred) {
        if (yTrue.length != yPred.length) {
            throw new IllegalArgumentException("Arrays must have the same length");
        }
        
        double sumAbsoluteError = 0.0;
        for (int i = 0; i < yTrue.length; i++) {
            sumAbsoluteError += Math.abs(yTrue[i] - yPred[i]);
        }
        
        return sumAbsoluteError / yTrue.length;
    }
}
