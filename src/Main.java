import data.*;
import java.io.File;
import java.io.IOException;
import java.util.*;
import models.*;

/**
 * Main application for the AI/ML Library (Java OOP Implementation).
 * 
 * Provides a menu-driven interface to:
 * - Load datasets
 * - Train and evaluate 5 machine learning algorithms
 * - Compare results
 * 
 * Team Members: Solomon Anagha, Geneva Regpala, Adrian Rodriguez, Hermit Singh
 * Course: CMPS 3500 - Artificial Intelligence
 * Project: AI Library Design Comparison
 */
public class Main {
    
    // Loaded data
    private static Dataset dataset = null;
    
    // Available dataset files
    private static List<String> availableDatasets = new ArrayList<>();
    private static double[][] XTrain = null;
    private static double[][] XTest = null;
    private static double[] yTrain = null;
    private static double[] yTest = null;
    private static String targetColumn = null;
    private static boolean isClassification = false;
    
    // Results storage
    private static Map<String, ModelResult> results = new HashMap<>();
    
    private static Scanner scanner = new Scanner(System.in);
    
    public static void main(String[] args) {
        System.out.println("******************************************************");
        System.out.println("Welcome to the AI/ML Library Implementation (Java OOP)");
        System.out.println("******************************************************\n");
        
        mainMenu();
    }
    
    /**
     * Main menu loop
     */
    private static void mainMenu() {
        while (true) {
            System.out.println("\n=== Java OOP Implementation Menu ===");
            System.out.println("(1) Load data");
            System.out.println("(2) Linear Regression (closed-form)");
            System.out.println("(3) Logistic Regression (binary)");
            System.out.println("(4) k-Nearest Neighbors");
            System.out.println("(5) Decision Tree (ID3)");
            System.out.println("(6) Gaussian Naive Bayes");
            System.out.println("(7) Print results");
            System.out.println("(8) Quit");
            System.out.print("\nEnter your choice: ");
            
            String choice = scanner.nextLine().trim();
            
            switch (choice) {
                case "1":
                    loadData();
                    break;
                case "2":
                    runLinearRegression();
                    break;
                case "3":
                    runLogisticRegression();
                    break;
                case "4":
                    runKNN();
                    break;
                case "5":
                    runDecisionTree();
                    break;
                case "6":
                    runNaiveBayes();
                    break;
                case "7":
                    printResults();
                    break;
                case "8":
                    System.out.println("\nThank you for using the ML Library!");
                    System.exit(0);
                default:
                    System.out.println("Invalid choice. Please try again.");
            }
        }
    }
    
    /**
     * Load and preprocess data
     */
    private static void loadData() {
        System.out.println("\n*** Loading Data ***");
        
        // Scan current directory for CSV files
        scanForDatasets();
        
        if (availableDatasets.isEmpty()) {
            System.out.println("No CSV files found in current directory.");
            return;
        }
        
        // Display available datasets
        System.out.println("Pick a dataset:");
        for (int i = 0; i < availableDatasets.size(); i++) {
            System.out.println("  " + (i + 1) + ". " + availableDatasets.get(i));
        }
        System.out.print("Enter option: ");
        
        String option = scanner.nextLine().trim();
        int choice;
        
        try {
            choice = Integer.parseInt(option);
            if (choice < 1 || choice > availableDatasets.size()) {
                System.out.println("Invalid option. Please try again.");
                return;
            }
        } catch (NumberFormatException e) {
            System.out.println("Invalid input. Please enter a number.");
            return;
        }
        
        String filePath = availableDatasets.get(choice - 1);
        
        try {
            long startTime = System.currentTimeMillis();
            
            // Load CSV
            DataLoader loader = new DataLoader(filePath);
            dataset = loader.load();
            
            long loadTime = System.currentTimeMillis() - startTime;
            
            System.out.println("\nLoading and cleaning input data set:");
            System.out.println("************************************");
            System.out.println("Total Columns Read: " + dataset.getNumColumns());
            System.out.println("Total Rows Read: " + dataset.getNumRows());
            System.out.println("Time to load: " + loadTime + " ms");
            
            System.out.println("\nData loaded successfully!");
            
        } catch (IOException e) {
            System.out.println("Error loading file: " + e.getMessage());
            System.out.println("Make sure 'adult_income_cleaned.csv' is in the current directory.");
        }
    }
    
    /**
     * Prepare data for a specific target
     */
    private static void prepareData(String target, boolean normalize) {
        if (dataset == null) {
            System.out.println("Please load data first (option 1)");
            return;
        }
        
        try {
            targetColumn = target;
            
            // Check if classification or regression
            isClassification = target.equals("income");
            
            // Extract X and y
            Dataset.DataSplit split = dataset.extractXY(target);
            
            // One-hot encode categorical features
            System.out.println("Applying one-hot encoding...");
            DataPreprocessor.ProcessedData processed = DataPreprocessor.oneHotEncode(
                split.X, split.featureNames
            );
            
            double[][] X = processed.X;
            double[] y = split.y;
            
            // Map income to binary if needed
            if (isClassification) {
                String[] originalLabels = dataset.getColumn(target);
                y = DataPreprocessor.mapIncomeToBinary(y, originalLabels);
            }
            
            // Train/test split
            System.out.println("Splitting data (80% train, 20% test)...");
            DataPreprocessor.TrainTestSplit trainTest = DataPreprocessor.trainTestSplit(
                X, y, 0.2, 42
            );
            
            XTrain = trainTest.XTrain;
            XTest = trainTest.XTest;
            yTrain = trainTest.yTrain;
            yTest = trainTest.yTest;
            
            // Normalize if requested
            if (normalize) {
                System.out.println("Applying z-score normalization...");
                DataPreprocessor.NormalizedData normalized = DataPreprocessor.normalize(XTrain);
                XTrain = normalized.X;
                XTest = DataPreprocessor.applyNormalization(XTest, normalized.means, normalized.stds);
            }
            
            System.out.println("Data prepared: " + XTrain.length + " training samples, " + 
                             XTest.length + " test samples, " + XTrain[0].length + " features");
            
        } catch (Exception e) {
            System.out.println("Error preparing data: " + e.getMessage());
            e.printStackTrace();
        }
    }
    
    /**
     * Run Linear Regression
     */
    private static void runLinearRegression() {
        System.out.println("\n*** Linear Regression (closed-form) ***");
        System.out.println("****************************************");
        
        System.out.println("Enter input options:");
        System.out.print("  Target variable (hours.per.week): ");
        String target = scanner.nextLine().trim();
        if (target.isEmpty()) target = "hours.per.week";
        
        System.out.print("  L2 regularization (default 0.01): ");
        String l2Str = scanner.nextLine().trim();
        double l2 = l2Str.isEmpty() ? 0.01 : Double.parseDouble(l2Str);
        
        // Prepare data 
        prepareData(target, false);
        if (XTrain == null) return;
        
        // Train model
        LinearRegression model = new LinearRegression(l2);
        
        long startTime = System.currentTimeMillis();
        model.fit(XTrain, yTrain);
        long trainTime = System.currentTimeMillis() - startTime;
        
        // Predict
        double[] predictions = model.predict(XTest);
        
        // Calculate metrics
        double rmse = metrics.RegressionMetrics.rmse(yTest, predictions);
        double r2 = metrics.RegressionMetrics.r2Score(yTest, predictions);
        
        // Count SLOC 
        int sloc = countSLOC("src/models/LinearRegression.java");
        
        // Display results
        System.out.println("\nOutputs:");
        System.out.println("*******");
        System.out.println("Algorithm: " + model.getName());
        System.out.println("Train time: " + trainTime + " ms");
        System.out.println("Metric 1 - RMSE: " + String.format("%.4f", rmse));
        System.out.println("Metric 2 - R^2: " + String.format("%.4f", r2));
        System.out.println("Metric 3 - SLOC: " + sloc);
        
        // Store results
        results.put("Linear Regression", new ModelResult(
            model.getName(), trainTime, rmse, r2, sloc
        ));
    }
    
    /**
     * Run Logistic Regression
     */
    private static void runLogisticRegression() {
        System.out.println("\n*** Logistic Regression (binary) ***");
        System.out.println("************************************");
        
        System.out.println("Enter input options:");
        System.out.print("  Target variable (income): ");
        String target = scanner.nextLine().trim();
        if (target.isEmpty()) target = "income";
        
        System.out.print("  Learning rate (default 0.2): ");
        String lrStr = scanner.nextLine().trim();
        double lr = lrStr.isEmpty() ? 0.2 : Double.parseDouble(lrStr);
        
        System.out.print("  Epochs (default 400): ");
        String epochsStr = scanner.nextLine().trim();
        int epochs = epochsStr.isEmpty() ? 400 : Integer.parseInt(epochsStr);
        
        System.out.print("  L2 regularization (default 0.003): ");
        String l2Str = scanner.nextLine().trim();
        double l2 = l2Str.isEmpty() ? 0.003 : Double.parseDouble(l2Str);
        
        System.out.print("  Random seed (default 7): ");
        String seedStr = scanner.nextLine().trim();
        int seed = seedStr.isEmpty() ? 7 : Integer.parseInt(seedStr);
        
        // Prepare data 
        prepareData(target, true);
        if (XTrain == null) return;
        
        // Train model
        LogisticRegression model = new LogisticRegression(lr, epochs, l2, seed);
        
        long startTime = System.currentTimeMillis();
        model.fit(XTrain, yTrain);
        long trainTime = System.currentTimeMillis() - startTime;
        
        // Predict
        double[] predictions = model.predict(XTest);
        
        // Calculate metrics
        double accuracy = metrics.ClassificationMetrics.accuracy(yTest, predictions);
        double macroF1 = metrics.ClassificationMetrics.macroF1(yTest, predictions);
        
        // Count SLOC
        int sloc = countSLOC("src/models/LogisticRegression.java");
        
        // Display results
        System.out.println("\nOutputs:");
        System.out.println("*******");
        System.out.println("Algorithm: " + model.getName());
        System.out.println("Train time: " + trainTime + " ms");
        System.out.println("Metric 1 - Accuracy: " + String.format("%.4f", accuracy));
        System.out.println("Metric 2 - Macro-F1: " + String.format("%.4f", macroF1));
        System.out.println("Metric 3 - SLOC: " + sloc);
        
        // Store results
        results.put("Logistic Regression", new ModelResult(
            model.getName(), trainTime, accuracy, macroF1, sloc
        ));
    }
    
    /**
     * Run k-Nearest Neighbors
     */
    private static void runKNN() {
        System.out.println("\n*** k-Nearest Neighbors ***");
        System.out.println("***************************");
        
        System.out.println("Enter input options:");
        System.out.print("  Target variable (income): ");
        String target = scanner.nextLine().trim();
        if (target.isEmpty()) target = "income";
        
        System.out.print("  k (default 5): ");
        String kStr = scanner.nextLine().trim();
        int k = kStr.isEmpty() ? 5 : Integer.parseInt(kStr);
        
        // Prepare data 
        prepareData(target, true);
        if (XTrain == null) return;
        
        // Train model
        KNearestNeighbors model = new KNearestNeighbors(k);
        
        long startTime = System.currentTimeMillis();
        model.fit(XTrain, yTrain);
        long trainTime = System.currentTimeMillis() - startTime;
        
        // Predict
        double[] predictions = model.predict(XTest);
        
        // Calculate metrics
        double accuracy = metrics.ClassificationMetrics.accuracy(yTest, predictions);
        double macroF1 = metrics.ClassificationMetrics.macroF1(yTest, predictions);
        
        // Count SLOC
        int sloc = countSLOC("src/models/KNearestNeighbors.java");
        
        // Display results
        System.out.println("\nOutputs:");
        System.out.println("*******");
        System.out.println("Algorithm: " + model.getName());
        System.out.println("Train time: " + trainTime + " ms");
        System.out.println("Metric 1 - Accuracy: " + String.format("%.4f", accuracy));
        System.out.println("Metric 2 - Macro-F1: " + String.format("%.4f", macroF1));
        System.out.println("Metric 3 - SLOC: " + sloc);
        
        // Store results
        results.put("k-Nearest Neighbors", new ModelResult(
            model.getName(), trainTime, accuracy, macroF1, sloc
        ));
    }
    
    /**
     * Run Decision Tree
     */
    private static void runDecisionTree() {
        System.out.println("\n*** Decision Tree (ID3) ***");
        System.out.println("***************************");
        
        System.out.println("Enter input options:");
        System.out.print("  Target variable (income): ");
        String target = scanner.nextLine().trim();
        if (target.isEmpty()) target = "income";
        
        System.out.print("  Max depth (default 10): ");
        String depthStr = scanner.nextLine().trim();
        int maxDepth = depthStr.isEmpty() ? 10 : Integer.parseInt(depthStr);
        
        // Prepare data 
        prepareData(target, false);
        if (XTrain == null) return;
        
        // Train model
        DecisionTree model = new DecisionTree(maxDepth, 2, 5);
        
        long startTime = System.currentTimeMillis();
        model.fit(XTrain, yTrain);
        long trainTime = System.currentTimeMillis() - startTime;
        
        // Predict
        double[] predictions = model.predict(XTest);
        
        // Calculate metrics
        double accuracy = metrics.ClassificationMetrics.accuracy(yTest, predictions);
        double macroF1 = metrics.ClassificationMetrics.macroF1(yTest, predictions);
        
        // Count SLOC
        int sloc = countSLOC("src/models/DecisionTree.java");
        
        // Display results
        System.out.println("\nOutputs:");
        System.out.println("*******");
        System.out.println("Algorithm: " + model.getName());
        System.out.println("Train time: " + trainTime + " ms");
        System.out.println("Metric 1 - Accuracy: " + String.format("%.4f", accuracy));
        System.out.println("Metric 2 - Macro-F1: " + String.format("%.4f", macroF1));
        System.out.println("Metric 3 - SLOC: " + sloc);
        
        // Store results
        results.put("Decision Tree", new ModelResult(
            model.getName(), trainTime, accuracy, macroF1, sloc
        ));
    }
    
    /**
     * Run Gaussian Naive Bayes
     */
    private static void runNaiveBayes() {
        System.out.println("\n*** Gaussian Naive Bayes ***");
        System.out.println("****************************");
        
        System.out.println("Enter input options:");
        System.out.print("  Target variable (income): ");
        String target = scanner.nextLine().trim();
        if (target.isEmpty()) target = "income";
        
        // Prepare data
        prepareData(target, true);
        if (XTrain == null) return;
        
        // Train model
        GaussianNaiveBayes model = new GaussianNaiveBayes();
        
        long startTime = System.currentTimeMillis();
        model.fit(XTrain, yTrain);
        long trainTime = System.currentTimeMillis() - startTime;
        
        // Predict
        double[] predictions = model.predict(XTest);
        
        // Calculate metrics
        double accuracy = metrics.ClassificationMetrics.accuracy(yTest, predictions);
        double macroF1 = metrics.ClassificationMetrics.macroF1(yTest, predictions);
        
        // Count SLOC
        int sloc = countSLOC("src/models/GaussianNaiveBayes.java");
        
        // Display results
        System.out.println("\nOutputs:");
        System.out.println("*******");
        System.out.println("Algorithm: " + model.getName());
        System.out.println("Train time: " + trainTime + " ms");
        System.out.println("Metric 1 - Accuracy: " + String.format("%.4f", accuracy));
        System.out.println("Metric 2 - Macro-F1: " + String.format("%.4f", macroF1));
        System.out.println("Metric 3 - SLOC: " + sloc);
        
        // Store results
        results.put("Gaussian Naive Bayes", new ModelResult(
            model.getName(), trainTime, accuracy, macroF1, sloc
        ));
    }
    
    /**
     * Print all results
     */
    private static void printResults() {
        System.out.println("\n*** Java OOP Implementation Results ***");
        System.out.println("****************************************");
        
        if (results.isEmpty()) {
            System.out.println("No results yet. Run some algorithms first!");
            return;
        }
        
        System.out.println("\n" + String.format("%-25s %-15s %-15s %-15s %-10s", 
            "Algorithm", "TrainTime(ms)", "TestMetric1", "TestMetric2", "SLOC"));
        System.out.println("=================================================================================");
        
        for (ModelResult result : results.values()) {
            System.out.println(String.format("%-25s %-15d %-15.4f %-15.4f %-10d",
                result.name, result.trainTime, result.metric1, result.metric2, result.sloc));
        }
    }
    
    /**
     * Count Source Lines of Code 
     */
    private static int countSLOC(String filePath) {
        try {
            java.io.BufferedReader reader = new java.io.BufferedReader(
                new java.io.FileReader(filePath)
            );
            
            int count = 0;
            String line;
            boolean inBlockComment = false;
            
            while ((line = reader.readLine()) != null) {
                line = line.trim();
                
                // Skip empty lines
                if (line.isEmpty()) continue;
                
                // Handle block comments
                if (line.startsWith("/*")) {
                    inBlockComment = true;
                }
                if (inBlockComment) {
                    if (line.endsWith("*/")) {
                        inBlockComment = false;
                    }
                    continue;
                }
                
                // Skip single-line comments
                if (line.startsWith("//")) continue;
                
                // Count as source line
                count++;
            }
            
            reader.close();
            return count;
            
        } catch (Exception e) {
            return 0; 
        }
    }
    
    /**
     * Scan current directory for CSV files
     */
    private static void scanForDatasets() {
        availableDatasets.clear();
        File currentDir = new File(".");
        File[] files = currentDir.listFiles();
        
        if (files != null) {
            for (File file : files) {
                if (file.isFile() && file.getName().toLowerCase().endsWith(".csv")) {
                    availableDatasets.add(file.getName());
                }
            }
        }
        
        // Sort alphabetically for consistent ordering
        Collections.sort(availableDatasets);
    }
    
    /**
     * Store results for comparison
     */
    private static class ModelResult {
        String name;
        long trainTime;
        double metric1;
        double metric2;
        int sloc;
        
        ModelResult(String name, long trainTime, double metric1, double metric2, int sloc) {
            this.name = name;
            this.trainTime = trainTime;
            this.metric1 = metric1;
            this.metric2 = metric2;
            this.sloc = sloc;
        }
    }
}
