package data;

import java.util.*;

/**
 * Data preprocessing utilities.
 * Handles one-hot encoding, normalization, and train/test splitting.
 * 
 * Team Members: Solomon Anagha, Geneva Regpala, Adrian Rodriguez, Hermit Singh
 */
public class DataPreprocessor {
    
    /**
     * One-hot encode categorical columns
     * Each unique value becomes its own binary column
     */
    public static ProcessedData oneHotEncode(String[][] X, List<String> featureNames) {
        int numRows = X.length;
        int numCols = X[0].length;
        
        // Identify categorical vs numeric columns
        List<Integer> categoricalIndices = new ArrayList<>();
        List<Integer> numericIndices = new ArrayList<>();
        
        for (int col = 0; col < numCols; col++) {
            if (isNumeric(X, col)) {
                numericIndices.add(col);
            } else {
                categoricalIndices.add(col);
            }
        }
        
        // Build new column names and data
        List<String> newFeatureNames = new ArrayList<>();
        List<double[]> newColumns = new ArrayList<>();
        
        // Add numeric columns as-is
        for (int col : numericIndices) {
            newFeatureNames.add(featureNames.get(col));
            double[] column = new double[numRows];
            for (int row = 0; row < numRows; row++) {
                column[row] = Double.parseDouble(X[row][col]);
            }
            newColumns.add(column);
        }
        
        // One-hot encode categorical columns
        Map<String, Map<String, Integer>> encodingMaps = new HashMap<>();
        
        for (int col : categoricalIndices) {
            String colName = featureNames.get(col);
            
            // Get unique values
            Set<String> uniqueValues = new HashSet<>();
            for (int row = 0; row < numRows; row++) {
                uniqueValues.add(X[row][col]);
            }
            
            List<String> sortedValues = new ArrayList<>(uniqueValues);
            Collections.sort(sortedValues);
            
            // Create encoding map for this column
            Map<String, Integer> valueToIndex = new HashMap<>();
            for (int i = 0; i < sortedValues.size(); i++) {
                valueToIndex.put(sortedValues.get(i), i);
            }
            encodingMaps.put(colName, valueToIndex);
            
            // Create binary columns for each unique value
            for (String value : sortedValues) {
                String newColName = colName + "_" + value;
                newFeatureNames.add(newColName);
                
                double[] column = new double[numRows];
                for (int row = 0; row < numRows; row++) {
                    column[row] = X[row][col].equals(value) ? 1.0 : 0.0;
                }
                newColumns.add(column);
            }
        }
        
        // Convert to 2D array
        double[][] XEncoded = new double[numRows][newColumns.size()];
        for (int col = 0; col < newColumns.size(); col++) {
            double[] column = newColumns.get(col);
            for (int row = 0; row < numRows; row++) {
                XEncoded[row][col] = column[row];
            }
        }
        
        return new ProcessedData(XEncoded, newFeatureNames, encodingMaps);
    }
    
    /**
     * Check if a column contains only numeric values
     */
    private static boolean isNumeric(String[][] X, int colIndex) {
        for (int row = 0; row < X.length; row++) {
            try {
                Double.parseDouble(X[row][colIndex]);
            } catch (NumberFormatException e) {
                return false;
            }
        }
        return true;
    }
    
    /**
     * Z-score normalization: (x - mean) / std
     * Important: Fit on training data, apply to both train and test
     */
    public static NormalizedData normalize(double[][] X) {
        int numRows = X.length;
        int numCols = X[0].length;
        
        double[] means = new double[numCols];
        double[] stds = new double[numCols];
        
        // Calculate mean for each column
        for (int col = 0; col < numCols; col++) {
            double sum = 0.0;
            for (int row = 0; row < numRows; row++) {
                double value = X[row][col];
                sum = sum + value;
            }
            double mean = sum / numRows;
            means[col] = mean;
        }
        
        // Calculate standard deviation for each column
        for (int col = 0; col < numCols; col++) {
            double sumSquaredDiff = 0.0;
            for (int row = 0; row < numRows; row++) {
                double diff = X[row][col] - means[col];
                sumSquaredDiff += diff * diff;
            }
            stds[col] = Math.sqrt(sumSquaredDiff / numRows);
            
            // Avoid division by zero
            if (stds[col] < 1e-8) {
                stds[col] = 1.0;
            }
        }
        
        // Apply normalization
        double[][] XNormalized = new double[numRows][numCols];
        for (int row = 0; row < numRows; row++) {
            for (int col = 0; col < numCols; col++) {
                XNormalized[row][col] = (X[row][col] - means[col]) / stds[col];
            }
        }
        
        return new NormalizedData(XNormalized, means, stds);
    }
    
    /**
     * Apply existing normalization parameters (for test data)
     */
    public static double[][] applyNormalization(double[][] X, double[] means, double[] stds) {
        int numRows = X.length;
        int numCols = X[0].length;
        
        double[][] XNormalized = new double[numRows][numCols];
        for (int row = 0; row < numRows; row++) {
            for (int col = 0; col < numCols; col++) {
                XNormalized[row][col] = (X[row][col] - means[col]) / stds[col];
            }
        }
        
        return XNormalized;
    }
    
    /**
     * Train/test split with shuffling
     */
    public static TrainTestSplit trainTestSplit(double[][] X, double[] y, double testSize, long seed) {
        int numSamples = X.length;
        int trainSize = (int) ((1.0 - testSize) * numSamples);
        
        // Create indices and shuffle
        Integer[] indices = new Integer[numSamples];
        for (int i = 0; i < numSamples; i++) {
            indices[i] = i;
        }
        
        Random random = new Random(seed);
        List<Integer> indicesList = Arrays.asList(indices);
        Collections.shuffle(indicesList, random);
        
        // Split into train and test
        double[][] XTrain = new double[trainSize][];
        double[] yTrain = new double[trainSize];
        double[][] XTest = new double[numSamples - trainSize][];
        double[] yTest = new double[numSamples - trainSize];
        
        for (int i = 0; i < trainSize; i++) {
            int idx = indicesList.get(i);
            XTrain[i] = X[idx].clone();
            yTrain[i] = y[idx];
        }
        
        for (int i = trainSize; i < numSamples; i++) {
            int idx = indicesList.get(i);
            XTest[i - trainSize] = X[idx].clone();
            yTest[i - trainSize] = y[idx];
        }
        
        return new TrainTestSplit(XTrain, XTest, yTrain, yTest);
    }
    
    /**
     * Map income labels to binary (0/1)
     * "<=50K" -> 0, ">50K" -> 1
     */
    public static double[] mapIncomeToBinary(double[] y, String[] originalLabels) {
        double[] yBinary = new double[y.length];
        for (int i = 0; i < y.length; i++) {
            String label = originalLabels[i].trim();
            if (label.equals("<=50K") || label.equals("<=50K.")) {
                yBinary[i] = 0.0;
            } else {
                yBinary[i] = 1.0;
            }
        }
        return yBinary;
    }
    
    // Result containers
    
    public static class ProcessedData {
        public double[][] X;
        public List<String> featureNames;
        public Map<String, Map<String, Integer>> encodingMaps;
        
        public ProcessedData(double[][] X, List<String> featureNames, Map<String, Map<String, Integer>> encodingMaps) {
            this.X = X;
            this.featureNames = featureNames;
            this.encodingMaps = encodingMaps;
        }
    }
    
    public static class NormalizedData {
        public double[][] X;
        public double[] means;
        public double[] stds;
        
        public NormalizedData(double[][] X, double[] means, double[] stds) {
            this.X = X;
            this.means = means;
            this.stds = stds;
        }
    }
    
    public static class TrainTestSplit {
        public double[][] XTrain;
        public double[][] XTest;
        public double[] yTrain;
        public double[] yTest;
        
        public TrainTestSplit(double[][] XTrain, double[][] XTest, double[] yTrain, double[] yTest) {
            this.XTrain = XTrain;
            this.XTest = XTest;
            this.yTrain = yTrain;
            this.yTest = yTest;
        }
    }
}
