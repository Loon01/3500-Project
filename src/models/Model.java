package models;

/**
 * Interface for all machine learning models.
 * All algorithms implement this to work with the menu system.
 */
public interface Model {
    
    /**
     * Train the model
     */
    void fit(double[][] XTrain, double[] yTrain);
    
    /**
     * Make predictions
     */
    double[] predict(double[][] XTest);
    
    /**
     * Evaluate model performance
     */
    double score(double[][] XTest, double[] yTest);
    
    /**
     * Get model name
     */
    String getName();
}
