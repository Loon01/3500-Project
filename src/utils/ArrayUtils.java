package utils;

/**
 * Utility class for common array operations.
 */
public class ArrayUtils {
    
    /**
     * Calculate the mean (average) of an array
     */
    public static double mean(double[] arr) {
        if (arr.length == 0) return 0.0;
        double sum = 0.0;
        for (double val : arr) {
            sum += val;
        }
        return sum / arr.length;
    }
    
    /**
     * Calculate the standard deviation of an array
     */
    public static double standardDeviation(double[] arr) {
        if (arr.length == 0) return 0.0;
        double avg = mean(arr);
        double sumSquaredDiff = 0.0;
        for (double val : arr) {
            double diff = val - avg;
            sumSquaredDiff += diff * diff;
        }
        return Math.sqrt(sumSquaredDiff / arr.length);
    }
    
    /**
     * Find the minimum value in an array
     */
    public static double min(double[] arr) {
        if (arr.length == 0) return Double.NaN;
        double minVal = arr[0];
        for (double val : arr) {
            if (val < minVal) minVal = val;
        }
        return minVal;
    }
    
    /**
     * Find the maximum value in an array
     */
    public static double max(double[] arr) {
        if (arr.length == 0) return Double.NaN;
        double maxVal = arr[0];
        for (double val : arr) {
            if (val > maxVal) maxVal = val;
        }
        return maxVal;
    }
    
    /**
     * Count occurrences of a specific value in an array
     */
    public static int count(double[] arr, double value) {
        int count = 0;
        for (double val : arr) {
            if (Math.abs(val - value) < 1e-9) count++;
        }
        return count;
    }
    
    /**
     * Get unique values from an array
     */
    public static double[] unique(double[] arr) {
        java.util.Set<Double> uniqueSet = new java.util.HashSet<>();
        for (double val : arr) {
            uniqueSet.add(val);
        }
        double[] result = new double[uniqueSet.size()];
        int i = 0;
        for (Double val : uniqueSet) {
            result[i++] = val;
        }
        return result;
    }
    
    /**
     * Print a 1D array (for debugging)
     */
    public static void print(double[] arr, String name) {
        System.out.print(name + ": [");
        for (int i = 0; i < Math.min(5, arr.length); i++) {
            System.out.printf("%.3f", arr[i]);
            if (i < arr.length - 1) System.out.print(", ");
        }
        if (arr.length > 5) System.out.print("...");
        System.out.println("]");
    }
}
