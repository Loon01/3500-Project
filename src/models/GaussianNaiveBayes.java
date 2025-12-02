package models;

import java.util.*;
import metrics.ClassificationMetrics;
import utils.ArrayUtils;

/**
 * Gaussian Naive Bayes classifier
 * Assumes normal distribution per class
 * 
 * Team Members: Solomon Anagha, Geneva Regpala, Adrian Rodriguez, Hermit Singh
 */
public class GaussianNaiveBayes implements Model {
    
    private double[] classPriors;
    private double[] classes;
    private double[][] featureMeans;      // Mean of each feature for each class
    private double[][] featureVars;       // Variance of each feature for each class
    
    private double varSmoothing = 1e-9;   // Smoothing parameter to avoid zero variance
    
    /**
     * Default constructor
     */
    public GaussianNaiveBayes() {
        this.classPriors = null;
        this.classes = null;
        this.featureMeans = null;
        this.featureVars = null;
    }
    
    /**
     * Constructor with custom variance smoothing
     */
    public GaussianNaiveBayes(double varSmoothing) {
        this();
        this.varSmoothing = varSmoothing;
    }
    
    public void fit(double[][] XTrain, double[] yTrain) {
        int numSamples = XTrain.length;
        int numFeatures = XTrain[0].length;
        
        // Get unique classes
        classes = ArrayUtils.unique(yTrain);
        int numClasses = classes.length;
        
        // Initialize arrays
        classPriors = new double[numClasses];
        featureMeans = new double[numClasses][numFeatures];
        featureVars = new double[numClasses][numFeatures];
        
        
        for (int c = 0; c < numClasses; c++) {
            double targetClass = classes[c];
            
            // Get samples belonging to this class
            List<double[]> classSamples = new ArrayList<>();
            for (int i = 0; i < numSamples; i++) {
                if (Math.abs(yTrain[i] - targetClass) < 1e-9) {
                    classSamples.add(XTrain[i]);
                }
            }
            
            int classCount = classSamples.size();
            
            // Calculate class prior: P(class) = count(class) / total
            classPriors[c] = (double) classCount / numSamples;
            
            // Calculate mean and variance for each feature
            for (int f = 0; f < numFeatures; f++) {
                // Extract feature values for this class
                double[] featureValues = new double[classCount];
                for (int i = 0; i < classCount; i++) {
                    featureValues[i] = classSamples.get(i)[f];
                }
                
                // Calculate mean
                featureMeans[c][f] = ArrayUtils.mean(featureValues);
                
                // Calculate variance
                double variance = 0.0;
                for (double value : featureValues) {
                    double diff = value - featureMeans[c][f];
                    variance += diff * diff;
                }
                variance /= classCount;
                
                // Add smoothing to avoid zero variance
                featureVars[c][f] = variance + varSmoothing;
            }
        }
    }
    
    /**
     * Calculate log probability of a sample belonging to a class
     * Uses log-space to avoid numerical underflow
     */
    private double logProbability(double[] x, int classIndex) {
        int numFeatures = x.length;
        
        // Start with log of class prior
        double logProb = Math.log(classPriors[classIndex]);
        
        // Add log probability for each feature
        for (int f = 0; f < numFeatures; f++) {
            double mean = featureMeans[classIndex][f];
            double var = featureVars[classIndex][f];
            
            // Log of Gaussian probability density function:
            double value = x[f];
            double diff = value - mean;
            double logCoeff = Math.log(2 * Math.PI * var);
            logProb = logProb - (0.5 * logCoeff);
            double diffSquared = diff * diff;
            logProb = logProb - (diffSquared / (2 * var));
        }
        
        return logProb;
    }
    
    /**
     * Predict class for a single sample
     */
    private double predictOne(double[] x) {
        double maxLogProb = Double.NEGATIVE_INFINITY;
        double predictedClass = classes[0];
        
        // Find class with highest log probability
        for (int c = 0; c < classes.length; c++) {
            double logProb = logProbability(x, c);
            
            if (logProb > maxLogProb) {
                maxLogProb = logProb;
                predictedClass = classes[c];
            }
        }
        
        return predictedClass;
    }
    
    public double[] predict(double[][] XTest) {
        if (classes == null) {
            throw new IllegalStateException("Model must be fit before predicting");
        }
        
        double[] predictions = new double[XTest.length];
        for (int i = 0; i < XTest.length; i++) {
            predictions[i] = predictOne(XTest[i]);
        }
        return predictions;
    }
    
    /**
     * Get probability estimates for each class (advanced feature)
     */
    public double[][] predictProba(double[][] XTest) {
        if (classes == null) {
            throw new IllegalStateException("Model must be fit before predicting");
        }
        
        int numSamples = XTest.length;
        int numClasses = classes.length;
        double[][] probabilities = new double[numSamples][numClasses];
        
        for (int i = 0; i < numSamples; i++) {
            // Get log probabilities for all classes
            double[] logProbs = new double[numClasses];
            double maxLogProb = Double.NEGATIVE_INFINITY;
            
            for (int c = 0; c < numClasses; c++) {
                logProbs[c] = logProbability(XTest[i], c);
                if (logProbs[c] > maxLogProb) {
                    maxLogProb = logProbs[c];
                }
            }
            
            // Convert log probabilities to probabilities (with numerical stability)
            double sumProbs = 0.0;
            for (int c = 0; c < numClasses; c++) {
                probabilities[i][c] = Math.exp(logProbs[c] - maxLogProb);
                sumProbs += probabilities[i][c];
            }
            
            // Normalize
            for (int c = 0; c < numClasses; c++) {
                probabilities[i][c] /= sumProbs;
            }
        }
        
        return probabilities;
    }
    
    public double score(double[][] XTest, double[] yTest) {
        double[] predictions = predict(XTest);
        return ClassificationMetrics.accuracy(yTest, predictions);
    }
    
    public String getName() {
        return "Gaussian Naive Bayes";
    }
}
