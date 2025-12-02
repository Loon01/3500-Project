package models;

import java.util.*;
import metrics.ClassificationMetrics;
import utils.ArrayUtils;

/**
 * Decision Tree - ID3 algorithm with information gain
 * Numeric features discretized using binning
 * 
 * Team Members: Solomon Anagha, Geneva Regpala, Adrian Rodriguez, Hermit Singh
 */
public class DecisionTree implements Model {
    
    // The root of the decision tree
    private TreeNode root;
    
    // Hyperparameters
    private int maxDepth;
    private int minSamplesSplit;
    private int numBins;  // For discretizing numeric features
    
    /**
     * Default constructor
     */
    public DecisionTree() {
        this(10, 2, 5);  // max_depth=10, min_samples=2, bins=5
    }
    
    /**
     * Constructor with custom parameters
     */
    public DecisionTree(int maxDepth, int minSamplesSplit, int numBins) {
        this.maxDepth = maxDepth;
        this.minSamplesSplit = minSamplesSplit;
        this.numBins = numBins;
        this.root = null;
    }
    
    /**
     * Inner class representing a node in the decision tree
     */
    private static class TreeNode {
        boolean isLeaf;
        double prediction;  // For leaf nodes
        
        int splitFeature;   // For internal nodes
        double splitValue;  // For internal nodes
        TreeNode left;      // For internal nodes
        TreeNode right;     // For internal nodes
        
        // Constructor for leaf node
        TreeNode(double prediction) {
            this.isLeaf = true;
            this.prediction = prediction;
        }
        
        // Constructor for internal node
        TreeNode(int splitFeature, double splitValue, TreeNode left, TreeNode right) {
            this.isLeaf = false;
            this.splitFeature = splitFeature;
            this.splitValue = splitValue;
            this.left = left;
            this.right = right;
        }
    }
    
    public void fit(double[][] XTrain, double[] yTrain) {
        // Build the tree recursively
        root = buildTree(XTrain, yTrain, 0);
    }
    
    /**
     * Recursively build the decision tree
     */
    private TreeNode buildTree(double[][] X, double[] y, int depth) {
        int numSamples = X.length;
        
        // Stopping conditions
        if (numSamples < minSamplesSplit || depth >= maxDepth || isPure(y)) {
            return new TreeNode(majorityClass(y));
        }
        
        // Find the best split
        BestSplit bestSplit = findBestSplit(X, y);
        
        // If no good split found, make leaf node
        if (bestSplit == null || bestSplit.gain <= 0) {
            return new TreeNode(majorityClass(y));
        }
        
        // Split the data
        DataSplit split = splitData(X, y, bestSplit.feature, bestSplit.threshold);
        
        // Recursively build left and right subtrees
        TreeNode left = buildTree(split.XLeft, split.yLeft, depth + 1);
        TreeNode right = buildTree(split.XRight, split.yRight, depth + 1);
        
        return new TreeNode(bestSplit.feature, bestSplit.threshold, left, right);
    }
    
    /**
     * Check if all labels are the same (pure node)
     */
    private boolean isPure(double[] y) {
        if (y.length == 0) return true;
        double firstLabel = y[0];
        for (double label : y) {
            if (Math.abs(label - firstLabel) > 1e-9) {
                return false;
            }
        }
        return true;
    }
    
    /**
     * Find the most common class label
     */
    private double majorityClass(double[] y) {
        Map<Double, Integer> counts = new HashMap<>();
        for (double label : y) {
            counts.put(label, counts.getOrDefault(label, 0) + 1);
        }
        
        double majority = y[0];
        int maxCount = 0;
        for (Map.Entry<Double, Integer> entry : counts.entrySet()) {
            if (entry.getValue() > maxCount) {
                maxCount = entry.getValue();
                majority = entry.getKey();
            }
        }
        return majority;
    }
    

    private double entropy(double[] y) {
        Map<Double, Integer> counts = new HashMap<>();
        for (double label : y) {
            counts.put(label, counts.getOrDefault(label, 0) + 1);
        }
        
        double ent = 0.0;
        int total = y.length;
        for (int count : counts.values()) {
            if (count > 0) {
                double p = (double) count / total;
                double logP = Math.log(p) / Math.log(2);
                double term = p * logP;
                ent = ent - term;
            }
        }
        return ent;
    }
    
    /**
     * Stores split info
     */
    private static class BestSplit {
        int feature;
        double threshold;
        double gain;
        
        BestSplit(int feature, double threshold, double gain) {
            this.feature = feature;
            this.threshold = threshold;
            this.gain = gain;
        }
    }
    
    /**
     * Find the best feature and threshold to split on
     */
    private BestSplit findBestSplit(double[][] X, double[] y) {
        double parentEntropy = entropy(y);
        double bestGain = -1;
        int bestFeature = -1;
        double bestThreshold = 0;
        
        int numFeatures = X[0].length;
        
        // Try each feature
        for (int feature = 0; feature < numFeatures; feature++) {
            // Get thresholds for this feature (using binning)
            double[] thresholds = getThresholds(X, feature);
            
            // Try each threshold
            for (double threshold : thresholds) {
                DataSplit split = splitData(X, y, feature, threshold);
                
                if (split.XLeft.length == 0 || split.XRight.length == 0) {
                    continue;  // Skip if one side is empty
                }
                
                // Calculate weighted average entropy after split
                double leftWeight = (double) split.yLeft.length / y.length;
                double rightWeight = (double) split.yRight.length / y.length;
                double leftEntropy = entropy(split.yLeft);
                double rightEntropy = entropy(split.yRight);
                
                double weightedEntropy = leftWeight * leftEntropy + rightWeight * rightEntropy;
                double infoGain = parentEntropy - weightedEntropy;
                
                if (infoGain > bestGain) {
                    bestGain = infoGain;
                    bestFeature = feature;
                    bestThreshold = threshold;
                }
            }
        }
        
        if (bestFeature == -1) {
            return null;
        }
        
        return new BestSplit(bestFeature, bestThreshold, bestGain);
    }
    
    /**
     * Get threshold values for splitting (using histogram binning)
     */
    private double[] getThresholds(double[][] X, int feature) {
        // Extract the feature column
        double[] featureValues = new double[X.length];
        for (int i = 0; i < X.length; i++) {
            featureValues[i] = X[i][feature];
        }
        
        // Get min and max
        double min = ArrayUtils.min(featureValues);
        double max = ArrayUtils.max(featureValues);
        
        if (Math.abs(max - min) < 1e-9) {
            return new double[] {min};  
        }
        
        // Create bin edges
        double[] thresholds = new double[numBins - 1];
        double step = (max - min) / numBins;
        for (int i = 0; i < numBins - 1; i++) {
            thresholds[i] = min + (i + 1) * step;
        }
        
        return thresholds;
    }
    
    /**
     * Split data container
     */
    private static class DataSplit {
        double[][] XLeft, XRight;
        double[] yLeft, yRight;
        
        DataSplit(double[][] XLeft, double[] yLeft, double[][] XRight, double[] yRight) {
            this.XLeft = XLeft;
            this.yLeft = yLeft;
            this.XRight = XRight;
            this.yRight = yRight;
        }
    }
    
    /**
     * Split data based on feature and threshold
     */
    private DataSplit splitData(double[][] X, double[] y, int feature, double threshold) {
        List<double[]> XLeftList = new ArrayList<>();
        List<Double> yLeftList = new ArrayList<>();
        List<double[]> XRightList = new ArrayList<>();
        List<Double> yRightList = new ArrayList<>();
        
        for (int i = 0; i < X.length; i++) {
            if (X[i][feature] <= threshold) {
                XLeftList.add(X[i]);
                yLeftList.add(y[i]);
            } else {
                XRightList.add(X[i]);
                yRightList.add(y[i]);
            }
        }
        
        // Convert lists to arrays
        double[][] XLeft = XLeftList.toArray(new double[XLeftList.size()][]);
        double[] yLeft = yLeftList.stream().mapToDouble(Double::doubleValue).toArray();
        double[][] XRight = XRightList.toArray(new double[XRightList.size()][]);
        double[] yRight = yRightList.stream().mapToDouble(Double::doubleValue).toArray();
        
        return new DataSplit(XLeft, yLeft, XRight, yRight);
    }
    
    /**
     * Predict class for a single sample
     */
    private double predictOne(double[] x, TreeNode node) {
        if (node.isLeaf) {
            return node.prediction;
        }
        
        if (x[node.splitFeature] <= node.splitValue) {
            return predictOne(x, node.left);
        } else {
            return predictOne(x, node.right);
        }
    }
    
    public double[] predict(double[][] XTest) {
        if (root == null) {
            throw new IllegalStateException("Model must be fit before predicting");
        }
        
        double[] predictions = new double[XTest.length];
        for (int i = 0; i < XTest.length; i++) {
            predictions[i] = predictOne(XTest[i], root);
        }
        return predictions;
    }
    
    public double score(double[][] XTest, double[] yTest) {
        double[] predictions = predict(XTest);
        return ClassificationMetrics.accuracy(yTest, predictions);
    }
    
    public String getName() {
        return "Decision Tree (ID3)";
    }
}
