package metrics;

import utils.ArrayUtils;

/**
 * Metrics for classification: Accuracy and Macro-F1
 */
public class ClassificationMetrics {
    
    /**
     * Calculate accuracy: (correct predictions) / (total predictions)
     */
    public static double accuracy(double[] yTrue, double[] yPred) {
        if (yTrue.length != yPred.length) {
            throw new IllegalArgumentException("Arrays must have the same length");
        }
        
        int correct = 0;
        for (int i = 0; i < yTrue.length; i++) {
            if (Math.abs(yTrue[i] - yPred[i]) < 1e-9) {
                correct++;
            }
        }
        return (double) correct / yTrue.length;
    }
    
    /**
     * Calculate F1 score for a single class
     * F1 = 2 * (precision * recall) / (precision + recall)
     */
    private static double f1Score(double[] yTrue, double[] yPred, double targetClass) {
        int truePositives = 0;
        int falsePositives = 0;
        int falseNegatives = 0;
        
        for (int i = 0; i < yTrue.length; i++) {
            boolean actualIsTarget = Math.abs(yTrue[i] - targetClass) < 1e-9;
            boolean predIsTarget = Math.abs(yPred[i] - targetClass) < 1e-9;
            
            if (actualIsTarget && predIsTarget) {
                truePositives++;
            } else if (!actualIsTarget && predIsTarget) {
                falsePositives++;
            } else if (actualIsTarget && !predIsTarget) {
                falseNegatives++;
            }
        }
        
        if (truePositives == 0) {
            return 0.0;  // No correct predictions for this class
        }
        
        double precision = (double) truePositives / (truePositives + falsePositives);
        double recall = (double) truePositives / (truePositives + falseNegatives);
        
        if (precision + recall == 0) {
            return 0.0;
        }
        
        return 2 * (precision * recall) / (precision + recall);
    }
    
    /**
     * Calculate Macro-F1: average F1 score across all classes
     * This treats all classes equally (good for imbalanced datasets)
     */
    public static double macroF1(double[] yTrue, double[] yPred) {
        // Get unique classes
        double[] classes = ArrayUtils.unique(yTrue);
        
        double totalF1 = 0.0;
        for (double targetClass : classes) {
            totalF1 += f1Score(yTrue, yPred, targetClass);
        }
        
        return totalF1 / classes.length;
    }
    
    /**
     * Print confusion matrix (helpful for debugging)
     */
    public static void printConfusionMatrix(double[] yTrue, double[] yPred) {
        double[] classes = ArrayUtils.unique(yTrue);
        int numClasses = classes.length;
        int[][] matrix = new int[numClasses][numClasses];
        
        for (int i = 0; i < yTrue.length; i++) {
            int trueIdx = -1, predIdx = -1;
            for (int j = 0; j < numClasses; j++) {
                if (Math.abs(yTrue[i] - classes[j]) < 1e-9) trueIdx = j;
                if (Math.abs(yPred[i] - classes[j]) < 1e-9) predIdx = j;
            }
            if (trueIdx >= 0 && predIdx >= 0) {
                matrix[trueIdx][predIdx]++;
            }
        }
        
        System.out.println("\nConfusion Matrix:");
        System.out.print("     ");
        for (int j = 0; j < numClasses; j++) {
            System.out.printf("Pred %.0f  ", classes[j]);
        }
        System.out.println();
        
        for (int i = 0; i < numClasses; i++) {
            System.out.printf("True %.0f: ", classes[i]);
            for (int j = 0; j < numClasses; j++) {
                System.out.printf("%6d  ", matrix[i][j]);
            }
            System.out.println();
        }
    }
}
