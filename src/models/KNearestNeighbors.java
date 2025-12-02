package models;

import java.util.Arrays;
import metrics.ClassificationMetrics;

/**
 * k-Nearest Neighbors classifier
 * Finds k closest training examples using Euclidean distance
 * 
 * Team Members: Solomon Anagha, Geneva Regpala, Adrian Rodriguez, Hermit Singh
 */
public class KNearestNeighbors implements Model {
    
    private double[][] XTrain;
    private double[] yTrain;
    
    // Hyperparameter
    private int k;  // Number of neighbors to consider
    
    /**
     * Constructor with default k=5
     */
    public KNearestNeighbors() {
        this(5);
    }
    
    /**
     * Constructor with custom k
     */
    public KNearestNeighbors(int k) {
        if (k <= 0) {
            throw new IllegalArgumentException("k must be positive");
        }
        this.k = k;
        this.XTrain = null;
        this.yTrain = null;
    }
    
    public void fit(double[][] XTrain, double[] yTrain) {
        this.XTrain = new double[XTrain.length][];
        for (int i = 0; i < XTrain.length; i++) {
            this.XTrain[i] = XTrain[i].clone();
        }
        this.yTrain = yTrain.clone();
    }
    
    /**
     * Compute Euclidean distance between two points
     */
    private double euclideanDistance(double[] point1, double[] point2) {
        double sum = 0.0;
        for (int i = 0; i < point1.length; i++) {
            double diff = point1[i] - point2[i];
            sum += diff * diff;
        }
        return Math.sqrt(sum);
    }
    
    /**
     * Predict the class for a single test point
     */
    private double predictOne(double[] testPoint) {
        int numTrainSamples = XTrain.length;
        
        // Compute distances to all training points
        double[] distances = new double[numTrainSamples];
        for (int i = 0; i < numTrainSamples; i++) {
            distances[i] = euclideanDistance(testPoint, XTrain[i]);
        }
        
        // Find indices of k nearest neighbors
        int[] nearestIndices = findKNearest(distances, k);
        
        // Get labels of k nearest neighbors
        double[] nearestLabels = new double[k];
        for (int i = 0; i < k; i++) {
            nearestLabels[i] = yTrain[nearestIndices[i]];
        }
        
        // Return majority class (most common label)
        return majorityVote(nearestLabels);
    }
    
    /**
     * Find indices of k smallest values in the array
     */
    private int[] findKNearest(double[] distances, int k) {
        // Create array of (distance, index) pairs
        DistanceIndexPair[] pairs = new DistanceIndexPair[distances.length];
        for (int i = 0; i < distances.length; i++) {
            pairs[i] = new DistanceIndexPair(distances[i], i);
        }
        
        // Sort by distance
        Arrays.sort(pairs);
        
        // Extract first k indices
        int[] indices = new int[k];
        for (int i = 0; i < k; i++) {
            indices[i] = pairs[i].index;
        }
        
        return indices;
    }
    
    /**
     * Stores distance and index for k-NN
     */
    private static class DistanceIndexPair implements Comparable<DistanceIndexPair> {
        double distance;
        int index;
        
        DistanceIndexPair(double distance, int index) {
            this.distance = distance;
            this.index = index;
        }
        
        public int compareTo(DistanceIndexPair other) {
            return Double.compare(this.distance, other.distance);
        }
    }
    
    /**
     * Return the most common value in an array (majority vote)
     */
    private double majorityVote(double[] labels) {
        // Count occurrences of each label
        java.util.Map<Double, Integer> counts = new java.util.HashMap<>();
        for (double label : labels) {
            int currentCount = counts.getOrDefault(label, 0);
            int newCount = currentCount + 1;
            counts.put(label, newCount);
        }
        
        // Find label with maximum count
        double majorityLabel = labels[0];
        int maxCount = 0;
        for (java.util.Map.Entry<Double, Integer> entry : counts.entrySet()) {
            if (entry.getValue() > maxCount) {
                maxCount = entry.getValue();
                majorityLabel = entry.getKey();
            }
        }
        
        return majorityLabel;
    }
    
    public double[] predict(double[][] XTest) {
        if (XTrain == null) {
            throw new IllegalStateException("Model must be fit before predicting");
        }
        
        int numSamples = XTest.length;
        double[] predictions = new double[numSamples];
        
        for (int i = 0; i < numSamples; i++) {
            predictions[i] = predictOne(XTest[i]);
        }
        
        return predictions;
    }
    
    public double score(double[][] XTest, double[] yTest) {
        double[] predictions = predict(XTest);
        return ClassificationMetrics.accuracy(yTest, predictions);
    }
    
    public String getName() {
        return "k-Nearest Neighbors (k=" + k + ")";
    }
    
    /**
     * Set k value 
     */
    public void setK(int k) {
        if (k <= 0) {
            throw new IllegalArgumentException("k must be positive");
        }
        this.k = k;
    }
    
    /**
     * Get current k value
     */
    public int getK() {
        return k;
    }
}
