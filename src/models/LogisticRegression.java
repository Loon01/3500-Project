package models;

import metrics.ClassificationMetrics;

/**
 * Logistic Regression - binary classification with gradient descent
 * 
 * Team Members: Solomon Anagha, Geneva Regpala, Adrian Rodriguez, Hermit Singh
 */
public class LogisticRegression implements Model {
    
    private double[] weights;
    private double bias;
    private double learningRate;
    private int epochs;
    private double l2Lambda;
    private int seed;
    
    public LogisticRegression() {
        this(0.1, 100, 0.0, 42);
    }
    
    public LogisticRegression(double learningRate, int epochs, double l2Lambda, int seed) {
        this.learningRate = learningRate;
        this.epochs = epochs;
        this.l2Lambda = l2Lambda;
        this.seed = seed;
        this.weights = null;
        this.bias = 0.0;
    }
    
    /**
     * Sigmoid activation function
     */
    private double sigmoid(double z) {
        // Clip z to prevent overflow
        if (z > 500) return 1.0;
        if (z < -500) return 0.0;
        return 1.0 / (1.0 + Math.exp(-z));
    }
    
    public void fit(double[][] XTrain, double[] yTrain) {
        int numSamples = XTrain.length;
        int numFeatures = XTrain[0].length;
        // System.out.println("Training logistic regression on " + numSamples + " samples");
        
        // Initialize weights with small random values
        java.util.Random random = new java.util.Random(seed);
        weights = new double[numFeatures];
        for (int i = 0; i < numFeatures; i++) {
            weights[i] = random.nextGaussian() * 0.01; 
        }
        bias = 0.0;
        
        // Gradient descent
        for (int epoch = 0; epoch < epochs; epoch++) {
            // Compute predictions for all samples
            double[] predictions = new double[numSamples];
            for (int i = 0; i < numSamples; i++) {
                double z = bias;
                for (int j = 0; j < numFeatures; j++) {
                    z += weights[j] * XTrain[i][j];
                }
                predictions[i] = sigmoid(z);
            }
            
            // Compute gradients
            double[] weightGradients = new double[numFeatures];
            double biasGradient = 0.0;
            
            for (int i = 0; i < numSamples; i++) {
                double prediction = predictions[i];
                double actual = yTrain[i];
                double error = prediction - actual;
                biasGradient = biasGradient + error;
                for (int j = 0; j < numFeatures; j++) {
                    double grad = error * XTrain[i][j];
                    weightGradients[j] = weightGradients[j] + grad;
                }
            }
            
            // Average gradients
            biasGradient /= numSamples;
            for (int j = 0; j < numFeatures; j++) {
                weightGradients[j] /= numSamples;
                if (l2Lambda > 0) {
                    weightGradients[j] += l2Lambda * weights[j];
                }
            }
            
            // Update weights and bias
            bias -= learningRate * biasGradient;
            for (int j = 0; j < numFeatures; j++) {
                weights[j] -= learningRate * weightGradients[j];
            }
        }
    }
    
    private double computeLoss(double[][] X, double[] y) {
        double loss = 0.0;
        int numSamples = X.length;
        
        for (int i = 0; i < numSamples; i++) {
            double z = bias;
            for (int j = 0; j < weights.length; j++) {
                z += weights[j] * X[i][j];
            }
            double pred = sigmoid(z);
            
            // Clip predictions to avoid log(0)
            pred = Math.max(1e-15, Math.min(1 - 1e-15, pred));
            
            loss += -(y[i] * Math.log(pred) + (1 - y[i]) * Math.log(1 - pred));
        }
        
        loss /= numSamples;
        
        // Add L2 penalty
        if (l2Lambda > 0) {
            double l2Penalty = 0.0;
            for (double w : weights) {
                l2Penalty += w * w;
            }
            loss += 0.5 * l2Lambda * l2Penalty;
        }
        
        return loss;
    }
    
    public double[] predict(double[][] XTest) {
        if (weights == null) {
            throw new IllegalStateException("Model must be fit before predicting");
        }
        
        int numSamples = XTest.length;
        double[] predictions = new double[numSamples];
        
        for (int i = 0; i < numSamples; i++) {
            double z = bias;
            for (int j = 0; j < weights.length; j++) {
                z += weights[j] * XTest[i][j];
            }
            double prob = sigmoid(z);
            predictions[i] = (prob >= 0.5) ? 1.0 : 0.0; 
        }
        
        return predictions;
    }
    
    /**
     * Predict probabilities instead of class labels
     */
    public double[] predictProba(double[][] XTest) {
        if (weights == null) {
            throw new IllegalStateException("Model must be fit before predicting");
        }
        
        int numSamples = XTest.length;
        double[] probabilities = new double[numSamples];
        
        for (int i = 0; i < numSamples; i++) {
            double z = bias;
            for (int j = 0; j < weights.length; j++) {
                z += weights[j] * XTest[i][j];
            }
            probabilities[i] = sigmoid(z);
        }
        
        return probabilities;
    }
    
    public double score(double[][] XTest, double[] yTest) {
        double[] predictions = predict(XTest);
        return ClassificationMetrics.accuracy(yTest, predictions);
    }
    
    public String getName() {
        return "Logistic Regression";
    }
}
