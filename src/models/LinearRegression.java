package models;

import metrics.RegressionMetrics;
import utils.MatrixUtils;

/**
 * Linear Regression - Normal Equation approach
 * Uses formula: w = (X^T X)^(-1) X^T y
 * 
 * Team Members: Solomon Anagha, Geneva Regpala, Adrian Rodriguez, Hermit Singh
 */
public class LinearRegression implements Model {
    
    private double[] weights;
    private double bias;
    private double l2Lambda;
    
    public LinearRegression() {
        this(0.01);
    }
    
    public LinearRegression(double l2Lambda) {
        this.l2Lambda = l2Lambda;
        this.weights = null;
        this.bias = 0.0;
    }
    
    public void fit(double[][] XTrain, double[] yTrain) {
        int numSamples = XTrain.length;
        int numFeatures = XTrain[0].length;
        
        double[][] XWithIntercept = MatrixUtils.addInterceptColumn(XTrain);
        double[][] XT = MatrixUtils.transpose(XWithIntercept);
        double[][] XTX = MatrixUtils.multiply(XT, XWithIntercept);
        
        // regularization
        if (l2Lambda > 0) {
            for (int i = 1; i < XTX.length; i++) {
                XTX[i][i] += l2Lambda;
            }
        }
        
        double[][] XTXInv = MatrixUtils.invert(XTX);
        
        double[] XTy = new double[numFeatures + 1];
        for (int i = 0; i < numFeatures + 1; i++) {
            XTy[i] = 0.0;
            for (int j = 0; j < numSamples; j++) {
                XTy[i] += XT[i][j] * yTrain[j];
            }
        }
        
        double[] wFull = new double[numFeatures + 1];
        for (int i = 0; i < numFeatures + 1; i++) {
            wFull[i] = 0.0;
            for (int j = 0; j < numFeatures + 1; j++) {
                wFull[i] += XTXInv[i][j] * XTy[j];
            }
        }
        
        this.bias = wFull[0];
        this.weights = new double[numFeatures];
        for (int i = 0; i < numFeatures; i++) {
            this.weights[i] = wFull[i + 1];
        }
    }
    
    public double[] predict(double[][] XTest) {
        if (weights == null) {
            throw new IllegalStateException("Model must be fit before predicting");
        }
        
        int numSamples = XTest.length;
        double[] predictions = new double[numSamples];
        
        for (int i = 0; i < numSamples; i++) {
            predictions[i] = bias;
            for (int j = 0; j < weights.length; j++) {
                predictions[i] += XTest[i][j] * weights[j];
            }
        }
        return predictions;
    }
    
    public double score(double[][] XTest, double[] yTest) {
        double[] predictions = predict(XTest);
        return RegressionMetrics.r2Score(yTest, predictions);
    }
    
    public String getName() {
        return "Linear Regression";
    }
    
    public double[] getWeights() {
        return weights;
    }
    
    public double getBias() {
        return bias;
    }
}
