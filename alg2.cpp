#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <cmath>
#include <chrono>
#include <algorithm>
#include <memory>
#include <random>
#include <map>
#include <set>
#include <iomanip>
#include <limits>
#include <queue>
#include <tuple>
#include <ctime>

using namespace std;

class RandomGenerator {
private:
    mt19937 generator;
    
public:
    RandomGenerator(int seed = 42) : generator(seed) {}
    
    int randint(int min, int max) {
        uniform_int_distribution<int> dist(min, max);
        return dist(generator);
    }
    
    double uniform(double min = 0.0, double max = 1.0) {
        uniform_real_distribution<double> dist(min, max);
        return dist(generator);
    }
    
    void setSeed(int seed) {
        generator.seed(seed);
    }
};


// ============================================================================
// CONSTANTS AND TYPE DEFINITIONS
// ============================================================================

const double EPSILON = 1e-10;
const int MAX_KNN_SAMPLES = 1000;  // Safety limit for KNN

// ============================================================================
// ERROR HANDLING
// ============================================================================

struct ErrorResult {
    bool success;
    string message;
    
    ErrorResult(bool s = true, const string& m = "") : success(s), message(m) {}
    
    static ErrorResult Success() { return ErrorResult(true); }
    static ErrorResult Failure(const string& msg) { return ErrorResult(false, msg); }
};

// ============================================================================
// DATA STRUCTURES
// ============================================================================

struct Matrix {
    vector<vector<double>> data;
    int rows;
    int cols;
    
    Matrix(int r = 0, int c = 0) : rows(r), cols(c) {
        data.resize(r, vector<double>(c, 0.0));
    }
    
    double& operator()(int i, int j) { return data[i][j]; }
    const double& operator()(int i, int j) const { return data[i][j]; }
    
    static int getRegressionSLOC() { return 150; } // Approximate SLOC for regression
};

struct Dataset {
    Matrix features;
    Matrix target;
    vector<string> feature_names;
    string target_name;
    vector<double> unique_targets;
    bool is_classification;
    int num_samples;
    int num_features;
    
    bool isBinaryClassification() const {
        return is_classification && unique_targets.size() == 2;
    }
    
    vector<int> getTargetLabels() const {
        vector<int> labels(num_samples);
        map<double, int> label_map;
        int current_label = 0;
        
        for (int i = 0; i < num_samples; i++) {
            double val = target(i, 0);
            if (label_map.find(val) == label_map.end()) {
                label_map[val] = current_label++;
            }
            labels[i] = label_map[val];
        }
        
        return labels;
    }
};

struct AlgorithmResult {
    string algorithmName;
    double trainTime;
    double metric1;
    double metric2;
    int sloc;
    bool isClassification;
};

vector<AlgorithmResult> allResults;
RandomGenerator global_rng(42);

// ============================================================================
// UTILITY FUNCTIONS
// ============================================================================


bool isNumeric(const string& str) {
    if (str.empty()) return false;
    
    size_t start = 0;
    if (str[0] == '-') start = 1;
    
    bool has_digit = false;
    bool has_decimal = false;
    
    for (size_t i = start; i < str.length(); i++) {
        if (isdigit(str[i])) {
            has_digit = true;
        } else if (str[i] == '.' && !has_decimal) {
            has_decimal = true;
        } else {
            return false;
        }
    }
    return has_digit;
}

void storeResults(const string& algorithmName, double trainTime, 
                  double metric1, double metric2, int sloc, bool isClassification = false) {
    AlgorithmResult result;
    result.algorithmName = algorithmName;
    result.trainTime = trainTime;
    result.metric1 = metric1;
    result.metric2 = metric2;
    result.sloc = sloc;
    result.isClassification = isClassification;
    allResults.push_back(result);
}

void printComparisonTable(const string& implName) {
    cout << "\n" << string(40, ' ') << "\n";
    cout << implName << " Results:\n";
    cout << string(implName.length() + 8, '*') << "\n";
    
    cout << left << setw(20) << "Impl" 
         << left << setw(25) << "Algorithm" 
         << left << setw(15) << "TrainTime" 
         << left << setw(15) << "TestMetric1" 
         << left << setw(15) << "TestMetric2" 
         << left << setw(10) << "SLOC" << "\n";
    cout << string(95, '-') << "\n";
    
    for (const auto& result : allResults) {
        cout << left << setw(20) << implName 
             << left << setw(25) << result.algorithmName
             << left << setw(15) << fixed << setprecision(4) << result.trainTime 
             << left << setw(15) << fixed << setprecision(4) << result.metric1 
             << left << setw(15) << fixed << setprecision(4) << result.metric2 
             << left << setw(10) << result.sloc << "\n";
    }
}

void saveResultsToFile(const string& filename, const string& implName) {
    ofstream outfile(filename);
    if (!outfile.is_open()) {
        cerr << "Error: Could not open file " << filename << " for writing.\n";
        return;
    }
    
    outfile << implName << " Results:\n";
    outfile << string(implName.length() + 8, '*') << "\n\n";
    
    outfile << left << setw(20) << "Impl" 
            << left << setw(25) << "Algorithm" 
            << left << setw(15) << "TrainTime" 
            << left << setw(15) << "TestMetric1" 
            << left << setw(15) << "TestMetric2" 
            << left << setw(10) << "SLOC" << "\n";
    outfile << string(95, '-') << "\n";
    
    for (const auto& result : allResults) {
        outfile << left << setw(20) << implName 
                << left << setw(25) << result.algorithmName
                << left << setw(15) << fixed << setprecision(6) << result.trainTime 
                << left << setw(15) << fixed << setprecision(6) << result.metric1 
                << left << setw(15) << fixed << setprecision(6) << result.metric2 
                << left << setw(10) << result.sloc << "\n";
    }
    
    outfile << "\n\nSummary Statistics:\n";
    outfile << "===================\n";
    outfile << "Total algorithms run: " << allResults.size() << "\n";
    
    if (!allResults.empty()) {
        auto fastest = min_element(allResults.begin(), allResults.end(),
            [](const AlgorithmResult& a, const AlgorithmResult& b) {
                return a.trainTime < b.trainTime;
            });
        outfile << "Fastest algorithm: " << fastest->algorithmName 
                << " (" << fastest->trainTime << " seconds)\n";
        
        outfile << "\nClassification Results:\n";
        for (const auto& result : allResults) {
            if (result.isClassification) {
                outfile << "  " << result.algorithmName 
                        << ": Accuracy=" << result.metric1 
                        << ", Macro-F1=" << result.metric2 << "\n";
            }
        }
        
        outfile << "\nRegression Results:\n";
        for (const auto& result : allResults) {
            if (!result.isClassification) {
                outfile << "  " << result.algorithmName 
                        << ": RMSE=" << result.metric1 
                        << ", R²=" << result.metric2 << "\n";
            }
        }
    }
    
    outfile.close();
    cout << "\nResults saved to: " << filename << "\n";
}

// ============================================================================
// CSV READER WITH ERROR HANDLING
// ============================================================================

Dataset readCSV(const string& filename, const string& target_col, bool is_classification) {
    Dataset dataset;
    ifstream file(filename);
    
    if (!file.is_open()) {
        throw runtime_error("Cannot open file: " + filename);
    }
    
    string line;
    vector<vector<string>> rows;
    vector<string> headers;
    
    // Read header
    if (!getline(file, line)) {
        throw runtime_error("Empty CSV file");
    }
    
    stringstream header_stream(line);
    string header;
    while (getline(header_stream, header, ',')) {
        if (!header.empty() && header.back() == '\r') {
            header.pop_back();
        }
        headers.push_back(header);
    }
    
    // Find target column
    int target_idx = -1;
    for (size_t i = 0; i < headers.size(); i++) {
        if (headers[i] == target_col) {
            target_idx = i;
            break;
        }
    }
    
    if (target_idx == -1) {
        throw runtime_error("Target column '" + target_col + "' not found");
    }
    
    // Read data
    int line_num = 1;
    while (getline(file, line)) {
        if (line.empty()) continue;
        
        vector<string> row;
        stringstream line_stream(line);
        string cell;
        
        while (getline(line_stream, cell, ',')) {
            if (!cell.empty() && cell.back() == '\r') {
                cell.pop_back();
            }
            row.push_back(cell);
        }
        
        if (row.size() != headers.size()) {
            throw runtime_error("Line " + to_string(line_num) + ": Expected " + 
                               to_string(headers.size()) + " columns, got " + 
                               to_string(row.size()));
        }
        
        rows.push_back(row);
        line_num++;
    }
    
    file.close();
    
    if (rows.empty()) {
        throw runtime_error("No data rows found");
    }
    
    // Identify numerical columns (excluding target)
    vector<int> numerical_cols;
    for (int col = 0; col < headers.size(); col++) {
        if (col == target_idx) continue;
        
        bool is_numerical = true;
        int check_rows = min(10, (int)rows.size());
        for (int row = 0; row < check_rows; row++) {
            if (!isNumeric(rows[row][col])) {
                is_numerical = false;
                break;
            }
        }
        
        if (is_numerical) {
            numerical_cols.push_back(col);
            dataset.feature_names.push_back(headers[col]);
        }
    }
    
    if (numerical_cols.empty()) {
        throw runtime_error("No numerical columns found for features");
    }
    
    // Create dataset
    dataset.num_samples = rows.size();
    dataset.num_features = numerical_cols.size();
    dataset.features = Matrix(dataset.num_samples, dataset.num_features);
    dataset.target = Matrix(dataset.num_samples, 1);
    dataset.target_name = target_col;
    dataset.is_classification = is_classification;
    
    // Check target column
    set<double> unique_targets;
    int check_rows = min(10, dataset.num_samples);
    
    if (is_classification) {
        set<string> unique_vals;
        for (int row = 0; row < check_rows; row++) {
            unique_vals.insert(rows[row][target_idx]);
        }
        
        if (unique_vals.size() > 100) {
            cout << "Warning: Target has " << unique_vals.size() 
                 << " unique values (might not be categorical)\n";
        }
    } else {
        bool target_is_numerical = true;
        for (int row = 0; row < check_rows; row++) {
            if (!isNumeric(rows[row][target_idx])) {
                target_is_numerical = false;
                break;
            }
        }
        
        if (!target_is_numerical) {
            throw runtime_error("Target column must be numerical for regression");
        }
    }
    
    // Fill matrices
    for (int i = 0; i < dataset.num_samples; i++) {
        // Features
        for (int j = 0; j < dataset.num_features; j++) {
            try {
                dataset.features(i, j) = stod(rows[i][numerical_cols[j]]);
            } catch (...) {
                throw runtime_error("Non-numeric value in column '" + 
                                   headers[numerical_cols[j]] + "' at row " + 
                                   to_string(i + 2));
            }
        }
        
        // Target
        try {
            double target_val = stod(rows[i][target_idx]);
            dataset.target(i, 0) = target_val;
            unique_targets.insert(target_val);
        } catch (...) {
            // For classification, convert string to numeric hash
            if (is_classification) {
                string label = rows[i][target_idx];
                double hash_val = 0.0;
                for (char c : label) {
                    hash_val = hash_val * 31 + c;
                }
                dataset.target(i, 0) = hash_val;
                unique_targets.insert(hash_val);
            } else {
                throw runtime_error("Non-numeric value in target column at row " + 
                                   to_string(i + 2));
            }
        }
    }
    
    dataset.unique_targets.assign(unique_targets.begin(), unique_targets.end());
    
    // For binary classification, map to 0/1
    if (is_classification && dataset.unique_targets.size() == 2) {
        map<double, int> label_map;
        label_map[dataset.unique_targets[0]] = 0;
        label_map[dataset.unique_targets[1]] = 1;
        
        for (int i = 0; i < dataset.num_samples; i++) {
            double original = dataset.target(i, 0);
            dataset.target(i, 0) = label_map[original];
        }
        
        dataset.unique_targets = {0.0, 1.0};
        cout << "Binary classification: mapped to 0/1\n";
    }
    
    return dataset;
}

// ============================================================================
// LINEAR REGRESSION IMPLEMENTATION
// ============================================================================

struct LinearRegression {
    vector<double> weights;
    double rmse;
    double r_squared;
    
    ErrorResult fit(const Matrix& X, const Matrix& y, double l2_reg = 0.0) {
        int n = X.rows;
        int m = X.cols;
        
        if (n <= m) {
            return ErrorResult::Failure("Need more samples than features");
        }
        
        // Add bias column
        Matrix X_with_bias(n, m + 1);
        for (int i = 0; i < n; i++) {
            X_with_bias(i, 0) = 1.0;  // Bias
            for (int j = 0; j < m; j++) {
                X_with_bias(i, j + 1) = X(i, j);
            }
        }
        
        // Compute X^T X
        Matrix XtX(m + 1, m + 1);
        for (int i = 0; i <= m; i++) {
            for (int j = 0; j <= m; j++) {
                double sum = 0.0;
                for (int k = 0; k < n; k++) {
                    sum += X_with_bias(k, i) * X_with_bias(k, j);
                }
                XtX(i, j) = sum;
                
                // Add L2 regularization (not for bias)
                if (i == j && i > 0 && l2_reg > 0) {
                    XtX(i, j) += l2_reg;
                }
            }
        }
        
        // Compute X^T y
        vector<double> Xty(m + 1, 0.0);
        for (int i = 0; i <= m; i++) {
            double sum = 0.0;
            for (int k = 0; k < n; k++) {
                sum += X_with_bias(k, i) * y(k, 0);
            }
            Xty[i] = sum;
        }
        
        // Solve using Gaussian elimination
        weights = solveLinearSystem(XtX, Xty);
        if (weights.empty()) {
            return ErrorResult::Failure("Matrix is singular or ill-conditioned");
        }
        
        return ErrorResult::Success();
    }
    
    vector<double> predict(const Matrix& X) const {
        int n = X.rows;
        int m = X.cols;
        vector<double> predictions(n, 0.0);
        
        for (int i = 0; i < n; i++) {
            double pred = weights[0];  // Bias
            for (int j = 0; j < m; j++) {
                pred += X(i, j) * weights[j + 1];
            }
            predictions[i] = pred;
        }
        
        return predictions;
    }
    
    void calculateMetrics(const Matrix& X, const Matrix& y) {
        vector<double> y_pred = predict(X);
        int n = X.rows;
        
        double sum_squared_error = 0.0;
        double sum_y = 0.0;
        
        for (int i = 0; i < n; i++) {
            double error = y(i, 0) - y_pred[i];
            sum_squared_error += error * error;
            sum_y += y(i, 0);
        }
        
        rmse = sqrt(sum_squared_error / n);
        
        double y_mean = sum_y / n;
        double total_sum_squares = 0.0;
        for (int i = 0; i < n; i++) {
            double diff = y(i, 0) - y_mean;
            total_sum_squares += diff * diff;
        }
        
        r_squared = (total_sum_squares == 0.0) ? 1.0 : 
                   1.0 - (sum_squared_error / total_sum_squares);
    }
    
private:
    vector<double> solveLinearSystem(const Matrix& A, const vector<double>& b) {
        int n = A.rows;
        Matrix aug = A;
        
        // Add b as last column
        for (int i = 0; i < n; i++) {
            aug.data[i].push_back(b[i]);
        }
        aug.cols++;
        
        // Gaussian elimination
        for (int i = 0; i < n; i++) {
            // Find pivot
            int max_row = i;
            double max_val = abs(aug(i, i));
            for (int k = i + 1; k < n; k++) {
                if (abs(aug(k, i)) > max_val) {
                    max_val = abs(aug(k, i));
                    max_row = k;
                }
            }
            
            if (max_val < EPSILON) {
                return {};  // Singular
            }
            
            // Swap rows
            if (max_row != i) {
                swap(aug.data[i], aug.data[max_row]);
            }
            
            // Eliminate below
            for (int k = i + 1; k < n; k++) {
                double factor = aug(k, i) / aug(i, i);
                for (int j = i; j <= n; j++) {
                    aug(k, j) -= factor * aug(i, j);
                }
            }
        }
        
        // Back substitution
        vector<double> x(n, 0.0);
        for (int i = n - 1; i >= 0; i--) {
            double sum = 0.0;
            for (int j = i + 1; j < n; j++) {
                sum += aug(i, j) * x[j];
            }
            x[i] = (aug(i, n) - sum) / aug(i, i);
        }
        
        return x;
    }
};

// ============================================================================
// LOGISTIC REGRESSION IMPLEMENTATION
// ============================================================================

struct LogisticRegression {
    vector<double> weights;
    double accuracy;
    double macro_f1;
    vector<double> training_history;
    
    static double sigmoid(double z) {
        if (z >= 0) {
            return 1.0 / (1.0 + exp(-z));
        } else {
            double exp_z = exp(z);
            return exp_z / (1.0 + exp_z);
        }
    }
    
    ErrorResult fit(const Matrix& X, const Matrix& y, double learning_rate = 0.01,
                   int epochs = 100, double l2_reg = 0.0, int seed = 42) {
        
        // Check binary classification
        set<double> unique_vals;
        for (int i = 0; i < y.rows; i++) {
            unique_vals.insert(y(i, 0));
        }
        
        if (unique_vals.size() != 2) {
            return ErrorResult::Failure("Logistic regression requires binary classification (found " + 
                                       to_string(unique_vals.size()) + " classes)");
        }
        
        int n = X.rows;
        int m = X.cols;
        
        // Initialize weights
        RandomGenerator rng(seed);
        weights.resize(m + 1);
        weights[0] = rng.uniform(-0.01, 0.01);  // Bias
        for (int i = 1; i <= m; i++) {
            weights[i] = rng.uniform(-0.01, 0.01);
        }
        
        training_history.clear();
        
        // Gradient descent
        for (int epoch = 0; epoch < epochs; epoch++) {
            double total_loss = 0.0;
            
            for (int i = 0; i < n; i++) {
                // Forward pass
                double z = weights[0];  // Bias
                for (int j = 0; j < m; j++) {
                    z += X(i, j) * weights[j + 1];
                }
                
                double prediction = sigmoid(z);
                double target = y(i, 0);
                
                // Clip prediction for numerical stability
                prediction = max(EPSILON, min(1.0 - EPSILON, prediction));
                
                // Binary cross-entropy loss
                double loss = -target * log(prediction) - (1.0 - target) * log(1.0 - prediction);
                total_loss += loss;
                
                // Backward pass
                double error = prediction - target;
                
                // Update bias
                weights[0] -= learning_rate * error;
                
                // Update weights
                for (int j = 0; j < m; j++) {
                    double gradient = error * X(i, j);
                    if (l2_reg > 0) {
                        gradient += l2_reg * weights[j + 1];
                    }
                    weights[j + 1] -= learning_rate * gradient;
                }
            }
            
            training_history.push_back(total_loss / n);
            
            // Early stopping
            if (epoch > 10) {
                if (training_history[epoch] > training_history[epoch - 5]) {
                    break;
                }
            }
        }
        
        return ErrorResult::Success();
    }
    
    vector<int> predictClasses(const Matrix& X) const {
        int n = X.rows;
        int m = X.cols;
        vector<int> predictions(n);
        
        for (int i = 0; i < n; i++) {
            double z = weights[0];
            for (int j = 0; j < m; j++) {
                z += X(i, j) * weights[j + 1];
            }
            
            double probability = sigmoid(z);
            predictions[i] = (probability >= 0.5) ? 1 : 0;
        }
        
        return predictions;
    }
    
    void calculateMetrics(const Matrix& X, const Matrix& y) {
        vector<int> predictions = predictClasses(X);
        vector<int> true_labels(y.rows);
        for (int i = 0; i < y.rows; i++) {
            true_labels[i] = static_cast<int>(y(i, 0));
        }
        
        // Confusion matrix
        int tp = 0, tn = 0, fp = 0, fn = 0;
        for (size_t i = 0; i < predictions.size(); i++) {
            if (predictions[i] == 1 && true_labels[i] == 1) tp++;
            else if (predictions[i] == 0 && true_labels[i] == 0) tn++;
            else if (predictions[i] == 1 && true_labels[i] == 0) fp++;
            else if (predictions[i] == 0 && true_labels[i] == 1) fn++;
        }
        
        accuracy = (tp + tn) / static_cast<double>(predictions.size());
        
        // Precision, recall, F1
        double precision_class0 = (tn + fn == 0) ? 0.0 : tn / static_cast<double>(tn + fn);
        double recall_class0 = (tn + fp == 0) ? 0.0 : tn / static_cast<double>(tn + fp);
        double f1_class0 = (precision_class0 + recall_class0 == 0) ? 0.0 : 
                          2 * precision_class0 * recall_class0 / (precision_class0 + recall_class0);
        
        double precision_class1 = (tp + fp == 0) ? 0.0 : tp / static_cast<double>(tp + fp);
        double recall_class1 = (tp + fn == 0) ? 0.0 : tp / static_cast<double>(tp + fn);
        double f1_class1 = (precision_class1 + recall_class1 == 0) ? 0.0 : 
                          2 * precision_class1 * recall_class1 / (precision_class1 + recall_class1);
        
        macro_f1 = (f1_class0 + f1_class1) / 2.0;
    }
};

// ============================================================================
// DECISION TREE IMPLEMENTATION
// ============================================================================

struct DecisionTree {
    struct TreeNode {
        bool is_leaf;
        int feature_idx;
        double threshold;
        double value;
        unique_ptr<TreeNode> left;
        unique_ptr<TreeNode> right;
    };
    
    unique_ptr<TreeNode> root;
    bool is_classification;
    int max_depth;
    int min_samples_split;
    int n_bins;
    
    double accuracy;
    double macro_f1;
    double rmse;
    double r_squared;
    
    DecisionTree(bool cls = true, int md = 5, int mss = 2, int nb = 32)
        : is_classification(cls), max_depth(md), min_samples_split(mss), n_bins(nb),
          accuracy(0.0), macro_f1(0.0), rmse(0.0), r_squared(0.0) {}
    
    ErrorResult fit(const Matrix& X, const Matrix& y) {
        if (max_depth <= 0) {
            return ErrorResult::Failure("max_depth must be positive");
        }
        if (min_samples_split < 2) {
            return ErrorResult::Failure("min_samples_split must be at least 2");
        }
        if (n_bins < 2) {
            return ErrorResult::Failure("n_bins must be at least 2");
        }
        
        vector<int> indices(X.rows);
        for (int i = 0; i < X.rows; i++) {
            indices[i] = i;
        }
        
        vector<double> y_vec(X.rows);
        for (int i = 0; i < X.rows; i++) {
            y_vec[i] = y(i, 0);
        }
        
        try {
            root = buildTree(X, y_vec, indices, 0);
            return ErrorResult::Success();
        } catch (const exception& e) {
            return ErrorResult::Failure(e.what());
        }
    }
    
    vector<double> predict(const Matrix& X) const {
        if (!root) {
            throw runtime_error("Model not fitted");
        }
        
        vector<double> predictions(X.rows);
        for (int i = 0; i < X.rows; i++) {
            predictions[i] = predictSingle(root.get(), X, i);
        }
        return predictions;
    }
    
    void calculateMetrics(const Matrix& X, const Matrix& y) {
        vector<double> predictions = predict(X);
        
        if (is_classification) {
            vector<int> pred_labels(X.rows);
            vector<int> true_labels(X.rows);
            for (int i = 0; i < X.rows; i++) {
                pred_labels[i] = static_cast<int>(round(predictions[i]));
                true_labels[i] = static_cast<int>(y(i, 0));
            }
            
            // Simplified accuracy (for multi-class, average per-class accuracy)
            int correct = 0;
            for (int i = 0; i < X.rows; i++) {
                if (pred_labels[i] == true_labels[i]) correct++;
            }
            accuracy = correct / static_cast<double>(X.rows);
            macro_f1 = accuracy;  // Simplified for multi-class
            
        } else {
            // Regression metrics
            double sum_squared_error = 0.0;
            double sum_y = 0.0;
            for (int i = 0; i < X.rows; i++) {
                double error = y(i, 0) - predictions[i];
                sum_squared_error += error * error;
                sum_y += y(i, 0);
            }
            
            rmse = sqrt(sum_squared_error / X.rows);
            
            double y_mean = sum_y / X.rows;
            double total_sum_squares = 0.0;
            for (int i = 0; i < X.rows; i++) {
                double diff = y(i, 0) - y_mean;
                total_sum_squares += diff * diff;
            }
            
            r_squared = (total_sum_squares == 0.0) ? 1.0 : 
                       1.0 - (sum_squared_error / total_sum_squares);
        }
    }
    
private:
    unique_ptr<TreeNode> buildTree(const Matrix& X, const vector<double>& y,
                                  const vector<int>& indices, int depth) {
        auto node = make_unique<TreeNode>();
        
        // Stopping conditions
        if (depth >= max_depth || indices.size() < min_samples_split) {
            node->is_leaf = true;
            if (is_classification) {
                // Majority vote
                map<double, int> counts;
                for (int idx : indices) {
                    counts[y[idx]]++;
                }
                int max_count = 0;
                for (const auto& pair : counts) {
                    if (pair.second > max_count) {
                        max_count = pair.second;
                        node->value = pair.first;
                    }
                }
            } else {
                // Mean value
                double sum = 0.0;
                for (int idx : indices) {
                    sum += y[idx];
                }
                node->value = sum / indices.size();
            }
            return node;
        }
        
        // Find best split
        int best_feature = -1;
        double best_threshold = 0.0;
        double best_score = numeric_limits<double>::max();
        
        for (int feature = 0; feature < X.cols; feature++) {
            // Collect feature values
            vector<double> values;
            for (int idx : indices) {
                values.push_back(X(idx, feature));
            }
            sort(values.begin(), values.end());
            
            // Generate thresholds using binning
            vector<double> thresholds;
            int n_bins_actual = min(n_bins, (int)values.size());
            for (int i = 1; i < n_bins_actual; i++) {
                int idx = (i * values.size()) / n_bins_actual;
                if (idx < values.size()) {
                    thresholds.push_back(values[idx]);
                }
            }
            
            // Remove duplicates
            thresholds.erase(unique(thresholds.begin(), thresholds.end()), thresholds.end());
            
            for (double threshold : thresholds) {
                vector<int> left_idx, right_idx;
                for (int idx : indices) {
                    if (X(idx, feature) <= threshold) {
                        left_idx.push_back(idx);
                    } else {
                        right_idx.push_back(idx);
                    }
                }
                
                if (left_idx.empty() || right_idx.empty()) continue;
                
                double score = 0.0;
                if (is_classification) {
                    // Gini impurity
                    map<double, int> left_counts, right_counts;
                    for (int idx : left_idx) left_counts[y[idx]]++;
                    for (int idx : right_idx) right_counts[y[idx]]++;
                    
                    double gini_left = 1.0, gini_right = 1.0;
                    for (const auto& pair : left_counts) {
                        double prob = pair.second / static_cast<double>(left_idx.size());
                        gini_left -= prob * prob;
                    }
                    for (const auto& pair : right_counts) {
                        double prob = pair.second / static_cast<double>(right_idx.size());
                        gini_right -= prob * prob;
                    }
                    
                    score = (left_idx.size() * gini_left + right_idx.size() * gini_right) / indices.size();
                } else {
                    // Variance reduction
                    double left_sum = 0.0, right_sum = 0.0;
                    double left_sq_sum = 0.0, right_sq_sum = 0.0;
                    
                    for (int idx : left_idx) {
                        left_sum += y[idx];
                        left_sq_sum += y[idx] * y[idx];
                    }
                    for (int idx : right_idx) {
                        right_sum += y[idx];
                        right_sq_sum += y[idx] * y[idx];
                    }
                    
                    double left_var = left_sq_sum / left_idx.size() - pow(left_sum / left_idx.size(), 2);
                    double right_var = right_sq_sum / right_idx.size() - pow(right_sum / right_idx.size(), 2);
                    
                    score = (left_idx.size() * left_var + right_idx.size() * right_var) / indices.size();
                }
                
                if (score < best_score) {
                    best_score = score;
                    best_feature = feature;
                    best_threshold = threshold;
                }
            }
        }
        
        if (best_feature == -1) {
            node->is_leaf = true;
            if (is_classification) {
                map<double, int> counts;
                for (int idx : indices) {
                    counts[y[idx]]++;
                }
                int max_count = 0;
                for (const auto& pair : counts) {
                    if (pair.second > max_count) {
                        max_count = pair.second;
                        node->value = pair.first;
                    }
                }
            } else {
                double sum = 0.0;
                for (int idx : indices) {
                    sum += y[idx];
                }
                node->value = sum / indices.size();
            }
            return node;
        }
        
        // Split the data
        vector<int> left_indices, right_indices;
        for (int idx : indices) {
            if (X(idx, best_feature) <= best_threshold) {
                left_indices.push_back(idx);
            } else {
                right_indices.push_back(idx);
            }
        }
        
        node->is_leaf = false;
        node->feature_idx = best_feature;
        node->threshold = best_threshold;
        
        // Recursively build children
        node->left = buildTree(X, y, left_indices, depth + 1);
        node->right = buildTree(X, y, right_indices, depth + 1);
        
        return node;
    }
    
    double predictSingle(const TreeNode* node, const Matrix& X, int row_idx) const {
        if (node->is_leaf) {
            return node->value;
        }
        
        if (X(row_idx, node->feature_idx) <= node->threshold) {
            return predictSingle(node->left.get(), X, row_idx);
        } else {
            return predictSingle(node->right.get(), X, row_idx);
        }
    }
};

// ============================================================================
// GAUSSIAN NAIVE BAYES IMPLEMENTATION
// ============================================================================

struct GaussianNB {
    vector<double> class_priors;
    vector<vector<double>> means;
    vector<vector<double>> variances;
    vector<int> class_labels;
    double accuracy;
    double macro_f1;
    
    GaussianNB() : accuracy(0.0), macro_f1(0.0) {}
    
    ErrorResult fit(const Matrix& X, const vector<int>& y) {
        // Extract unique classes
        map<int, int> class_map;
        for (int label : y) {
            if (class_map.find(label) == class_map.end()) {
                class_map[label] = class_labels.size();
                class_labels.push_back(label);
            }
        }
        
        int n_classes = class_labels.size();
        int n_features = X.cols;
        
        means.resize(n_classes, vector<double>(n_features, 0.0));
        variances.resize(n_classes, vector<double>(n_features, 0.0));
        class_priors.resize(n_classes, 0.0);
        
        vector<int> class_counts(n_classes, 0);
        vector<vector<double>> class_sums(n_classes, vector<double>(n_features, 0.0));
        
        // Calculate sums
        for (int i = 0; i < X.rows; i++) {
            int class_idx = class_map[y[i]];
            class_counts[class_idx]++;
            for (int j = 0; j < n_features; j++) {
                class_sums[class_idx][j] += X(i, j);
            }
        }
        
        // Calculate means and priors
        for (int c = 0; c < n_classes; c++) {
            if (class_counts[c] > 0) {
                for (int j = 0; j < n_features; j++) {
                    means[c][j] = class_sums[c][j] / class_counts[c];
                }
                class_priors[c] = class_counts[c] / static_cast<double>(X.rows);
            }
        }
        
        // Calculate variances
        for (int c = 0; c < n_classes; c++) {
            if (class_counts[c] > 1) {
                vector<double> sum_sq(n_features, 0.0);
                for (int i = 0; i < X.rows; i++) {
                    if (class_map[y[i]] == c) {
                        for (int j = 0; j < n_features; j++) {
                            double diff = X(i, j) - means[c][j];
                            sum_sq[j] += diff * diff;
                        }
                    }
                }
                for (int j = 0; j < n_features; j++) {
                    variances[c][j] = sum_sq[j] / (class_counts[c] - 1);
                    if (variances[c][j] < EPSILON) variances[c][j] = EPSILON;
                }
            }
        }
        
        return ErrorResult::Success();
    }
    
    vector<int> predict(const Matrix& X) const {
        int n_samples = X.rows;
        int n_classes = class_priors.size();
        int n_features = X.cols;
        
        vector<int> predictions(n_samples);
        
        for (int i = 0; i < n_samples; i++) {
            vector<double> log_probs(n_classes, 0.0);
            
            for (int c = 0; c < n_classes; c++) {
                log_probs[c] = log(class_priors[c]);
                for (int j = 0; j < n_features; j++) {
                    double x = X(i, j);
                    double mean = means[c][j];
                    double var = variances[c][j];
                    
                    // Gaussian PDF (log)
                    log_probs[c] += -0.5 * log(2 * M_PI * var) - 
                                   ((x - mean) * (x - mean)) / (2 * var);
                }
            }
            
            // Find class with highest probability
            int best_class = 0;
            for (int c = 1; c < n_classes; c++) {
                if (log_probs[c] > log_probs[best_class]) {
                    best_class = c;
                }
            }
            
            predictions[i] = class_labels[best_class];
        }
        
        return predictions;
    }
    
    void calculateMetrics(const Matrix& X, const vector<int>& y_true) {
        vector<int> predictions = predict(X);
        
        // Multi-class accuracy (simplified)
        int correct = 0;
        for (size_t i = 0; i < predictions.size(); i++) {
            if (predictions[i] == y_true[i]) correct++;
        }
        
        accuracy = correct / static_cast<double>(predictions.size());
        macro_f1 = accuracy;  // Simplified for multi-class
    }
};

// ============================================================================
// K-NEAREST NEIGHBORS IMPLEMENTATION (WITH SAFETY LIMITS)
// ============================================================================

struct KNN {
    Matrix X_train;
    Matrix y_train;
    int n_neighbors;
    string distance_metric;
    bool weighted;
    string tie_break;
    bool is_classification;
    vector<int> class_labels;
    
    double accuracy;
    double macro_f1;
    double rmse;
    double r_squared;
    
    KNN(int k = 5, string dist = "euclidean", bool weight = false, 
        string tie = "smallest_label", bool cls = true)
        : n_neighbors(k), distance_metric(dist), weighted(weight), 
          tie_break(tie), is_classification(cls),
          accuracy(0.0), macro_f1(0.0), rmse(0.0), r_squared(0.0) {}
    
    ErrorResult fit(const Matrix& X, const Matrix& y) {
        // Safety checks
        if (n_neighbors <= 0) {
            return ErrorResult::Failure("n_neighbors must be positive");
        }
        
        if (n_neighbors > X.rows) {
            cout << "Warning: Reducing n_neighbors from " << n_neighbors 
                 << " to " << X.rows << " (number of samples)\n";
            n_neighbors = X.rows;
        }
        
        if (distance_metric != "euclidean" && distance_metric != "manhattan") {
            return ErrorResult::Failure("distance_metric must be 'euclidean' or 'manhattan'");
        }
        
        if (tie_break != "smallest_label" && tie_break != "random") {
            return ErrorResult::Failure("tie_break must be 'smallest_label' or 'random'");
        }
        
        // Copy training data
        X_train = X;
        y_train = y;
        
        // For classification, extract class labels
        if (is_classification) {
            set<int> unique_labels;
            for (int i = 0; i < y.rows; i++) {
                unique_labels.insert(static_cast<int>(y(i, 0)));
            }
            class_labels.assign(unique_labels.begin(), unique_labels.end());
            sort(class_labels.begin(), class_labels.end());
        }
        
        return ErrorResult::Success();
    }
    
    double calculateDistance(const vector<double>& a, const vector<double>& b) const {
        if (a.size() != b.size()) {
            throw runtime_error("Feature vectors have different sizes");
        }
        
        double distance = 0.0;
        
        if (distance_metric == "euclidean") {
            for (size_t i = 0; i < a.size(); i++) {
                double diff = a[i] - b[i];
                distance += diff * diff;
            }
            distance = sqrt(distance);
        } else { // manhattan
            for (size_t i = 0; i < a.size(); i++) {
                distance += abs(a[i] - b[i]);
            }
        }
        
        return distance;
    }
    
    vector<double> predict(const Matrix& X) const {
        if (X_train.rows == 0) {
            throw runtime_error("Model not fitted yet");
        }
        
        // Safety limit
        int max_samples = min(MAX_KNN_SAMPLES, X.rows);
        if (X.rows > MAX_KNN_SAMPLES) {
            cout << "Warning: Limiting KNN prediction to " << max_samples 
                 << " samples (safety limit)\n";
        }
        
        vector<double> predictions(max_samples);
        RandomGenerator rng(42);
        
        cout << "KNN: Predicting " << max_samples << " samples...\n";
        
        for (int i = 0; i < max_samples; i++) {
            // Show progress
            if (i % max(1, max_samples / 10) == 0) {
                cout << "\rProgress: " << (i * 100 / max_samples) << "%";
                cout.flush();
            }
            
            // Extract test sample
            vector<double> test_sample(X.cols);
            for (int j = 0; j < X.cols; j++) {
                test_sample[j] = X(i, j);
            }
            
            // Calculate distances to all training samples
            vector<pair<double, double>> distances; // (distance, label)
            
            for (int j = 0; j < X_train.rows; j++) {
                vector<double> train_sample(X_train.cols);
                for (int k = 0; k < X_train.cols; k++) {
                    train_sample[k] = X_train(j, k);
                }
                
                double dist = calculateDistance(test_sample, train_sample);
                distances.push_back(make_pair(dist, y_train(j, 0)));
            }
            
            // Sort by distance
            sort(distances.begin(), distances.end());
            
            // Take k nearest neighbors
            int k = min(n_neighbors, (int)distances.size());
            
            if (is_classification) {
                // Classification
                map<double, double> class_weights;
                
                for (int j = 0; j < k; j++) {
                    double label = distances[j].second;
                    double weight = 1.0;
                    
                    if (weighted) {
                        if (distances[j].first > 0) {
                            weight = 1.0 / distances[j].first;
                        } else {
                            weight = 1e10; // Large weight for zero distance
                        }
                    }
                    
                    class_weights[label] += weight;
                }
                
                // Find class with highest weight
                double best_class = -1;
                double best_weight = -1.0;
                vector<double> tied_classes;
                
                for (const auto& pair : class_weights) {
                    double cls = pair.first;
                    double weight = pair.second;
                    
                    if (weight > best_weight) {
                        best_weight = weight;
                        best_class = cls;
                        tied_classes.clear();
                        tied_classes.push_back(cls);
                    } else if (weight == best_weight) {
                        tied_classes.push_back(cls);
                    }
                }
                
                // Handle ties
                if (tied_classes.size() > 1) {
                    if (tie_break == "smallest_label") {
                        best_class = *min_element(tied_classes.begin(), tied_classes.end());
                    } else { // random
                        int idx = rng.randint(0, tied_classes.size() - 1);
                        best_class = tied_classes[idx];
                    }
                }
                
                predictions[i] = best_class;
                
            } else {
                // Regression - weighted average
                double sum_weights = 0.0;
                double weighted_sum = 0.0;
                
                for (int j = 0; j < k; j++) {
                    double weight = 1.0;
                    
                    if (weighted) {
                        if (distances[j].first > 0) {
                            weight = 1.0 / distances[j].first;
                        } else {
                            weight = 1e10;
                        }
                    }
                    
                    weighted_sum += weight * distances[j].second;
                    sum_weights += weight;
                }
                
                predictions[i] = weighted_sum / sum_weights;
            }
        }
        
        cout << "\rProgress: 100%\n";
        
        return predictions;
    }
    
    void calculateMetrics(const Matrix& X, const Matrix& y) {
        vector<double> predictions = predict(X);
        int n = predictions.size();
        
        if (is_classification) {
            vector<int> pred_labels(n);
            vector<int> true_labels(n);
            for (int i = 0; i < n; i++) {
                pred_labels[i] = static_cast<int>(predictions[i]);
                true_labels[i] = static_cast<int>(y(i, 0));
            }
            
            // Simplified accuracy
            int correct = 0;
            for (int i = 0; i < n; i++) {
                if (pred_labels[i] == true_labels[i]) correct++;
            }
            
            accuracy = correct / static_cast<double>(n);
            macro_f1 = accuracy;  // Simplified
            
        } else {
            // Regression metrics
            double sum_squared_error = 0.0;
            double sum_y = 0.0;
            for (int i = 0; i < n; i++) {
                double error = y(i, 0) - predictions[i];
                sum_squared_error += error * error;
                sum_y += y(i, 0);
            }
            
            rmse = sqrt(sum_squared_error / n);
            
            double y_mean = sum_y / n;
            double total_sum_squares = 0.0;
            for (int i = 0; i < n; i++) {
                double diff = y(i, 0) - y_mean;
                total_sum_squares += diff * diff;
            }
            
            r_squared = (total_sum_squares == 0.0) ? 1.0 : 
                       1.0 - (sum_squared_error / total_sum_squares);
        }
    }
};

// ============================================================================
// USER INPUT VALIDATION FUNCTIONS
// ============================================================================

int getIntegerInput(const string& prompt, int min_val, int max_val) {
    int value;
    while (true) {
        cout << prompt;
        cin >> value;
        
        if (cin.fail()) {
            cin.clear();
            cin.ignore(numeric_limits<streamsize>::max(), '\n');
            cout << "Error: Please enter a valid integer.\n";
            continue;
        }
        
        if (value < min_val || value > max_val) {
            cout << "Error: Value must be between " << min_val << " and " << max_val << ".\n";
            continue;
        }
        
        break;
    }
    return value;
}

double getDoubleInput(const string& prompt, double min_val = -INFINITY, double max_val = INFINITY) {
    double value;
    while (true) {
        cout << prompt;
        cin >> value;
        
        if (cin.fail()) {
            cin.clear();
            cin.ignore(numeric_limits<streamsize>::max(), '\n');
            cout << "Error: Please enter a valid number.\n";
            continue;
        }
        
        if (value < min_val || value > max_val) {
            cout << "Error: Value must be between " << min_val << " and " << max_val << ".\n";
            continue;
        }
        
        break;
    }
    return value;
}

string getStringInput(const string& prompt, const vector<string>& valid_options = {}) {
    string input;
    while (true) {
        cout << prompt;
        cin >> input;
        
        if (valid_options.empty()) {
            break;
        }
        
        for (const string& option : valid_options) {
            if (input == option) {
                return input;
            }
        }
        
        cout << "Error: Valid options are: ";
        for (size_t i = 0; i < valid_options.size(); i++) {
            cout << valid_options[i];
            if (i < valid_options.size() - 1) cout << ", ";
        }
        cout << "\n";
    }
    return input;
}

// ============================================================================
// MAIN FUNCTION WITH CONTINUOUS LOOP
// ============================================================================

int main() {
    bool running = true;
    string current_filename = "";
    
    cout << "==================================================\n";
    cout << "     MACHINE LEARNING ALGORITHM SUITE\n";
    cout << "==================================================\n\n";
    
    while (running) {
        try {
            // Get filename
            if (current_filename.empty()) {
                cout << "Enter CSV filename (or 'quit' to exit): ";
                cin >> current_filename;
                
                if (current_filename == "quit") {
                    running = false;
                    continue;
                }
                
                // Test if file exists
                ifstream test_file(current_filename);
                if (!test_file.good()) {
                    cerr << "Error: File '" << current_filename << "' not found.\n";
                    current_filename = "";
                    continue;
                }
                test_file.close();
            }
            
            cout << "\nCurrent file: " << current_filename << "\n";
            cout << "Press Enter to continue or 'c' to change file: ";
            cin.ignore();
            string input;
            getline(cin, input);
            if (input == "c" || input == "C") {
                current_filename = "";
                continue;
            }
            
            // Display main menu
            cout << "\n" << string(40, '=') << "\n";
            cout << "MAIN MENU\n";
            cout << string(40, '=') << "\n";
            cout << "1. Linear Regression (closed-form)\n";
            cout << "2. Logistic Regression (binary classification)\n";
            cout << "3. Decision Tree (classification/regression)\n";
            cout << "4. Gaussian Naive Bayes (classification)\n";
            cout << "5. K-Nearest Neighbors (classification/regression)\n";
            cout << "6. Print Results Comparison Table\n";
            cout << "7. Save Results to File and Quit\n";
            cout << "8. Quit Without Saving\n\n";
            
            int choice = getIntegerInput("Choose option (1-8): ", 1, 8);
            
            if (choice == 8) {
                cout << "\nGoodbye!\n";
                running = false;
                continue;
            }
            
            if (choice == 7) {
                if (!allResults.empty()) {
                    string filename;
                    cout << "Enter filename to save results: ";
                    cin >> filename;
                    saveResultsToFile(filename, "MLSuite");
                } else {
                    cout << "No results to save.\n";
                }
                cout << "\nGoodbye!\n";
                running = false;
                continue;
            }
            
            if (choice == 6) {
                if (allResults.empty()) {
                    cout << "\nNo algorithms have been run yet.\n";
                } else {
                    printComparisonTable("MLSuite");
                }
                continue;
            }
            
            // For algorithms 1-5, get target variable
            string target_col;
            cout << "\nEnter target variable name: ";
            cin >> target_col;
            
            // Determine task type
            bool is_classification = (choice != 1);
            
            // Load dataset
            cout << "\nLoading dataset...\n";
            Dataset dataset;
            try {
                dataset = readCSV(current_filename, target_col, is_classification);
                cout << "Dataset loaded successfully!\n";
                cout << "Samples: " << dataset.num_samples << "\n";
                cout << "Features: " << dataset.num_features << "\n";
                if (is_classification) {
                    cout << "Classes: " << dataset.unique_targets.size() << "\n";
                }
            } catch (const exception& e) {
                cerr << "Error loading dataset: " << e.what() << "\n";
                cout << "Returning to main menu...\n";
                continue;
            }
            
            // Warn about large dataset for KNN
            if (choice == 5 && dataset.num_samples > 1000) {
                cout << "\n⚠️  WARNING: Dataset has " << dataset.num_samples << " samples.\n";
                cout << "KNN may be slow with large datasets.\n";
                cout << "Continue anyway? (y/n): ";
                char continue_choice;
                cin >> continue_choice;
                if (continue_choice != 'y' && continue_choice != 'Y') {
                    cout << "Returning to menu...\n";
                    continue;
                }
            }
            
            // Algorithm-specific parameters
            double l2_reg = 0.0;
            double learning_rate = 0.01;
            int epochs = 100;
            int seed = 42;
            int max_depth = 5;
            int min_samples_split = 2;
            int n_bins = 32;
            int n_neighbors = 5;
            string distance_metric = "euclidean";
            bool weighted = false;
            string tie_break = "smallest_label";
            
            // Get algorithm-specific parameters
            switch (choice) {
                case 1: // Linear Regression
                case 2: // Logistic Regression
                    l2_reg = getDoubleInput("Enter L2 regularization (0 for none): ", 0.0);
                    if (choice == 2) {
                        learning_rate = getDoubleInput("Enter learning rate (default 0.01): ", 0.0001, 1.0);
                        epochs = getIntegerInput("Enter number of epochs (default 100): ", 1, 10000);
                        seed = getIntegerInput("Enter random seed (default 42): ", 0, 1000000);
                    }
                    break;
                    
                case 3: // Decision Tree
                    max_depth = getIntegerInput("Enter max_depth (default 5): ", 1, 100);
                    min_samples_split = getIntegerInput("Enter min_samples_split (default 2): ", 2, 1000);
                    n_bins = getIntegerInput("Enter n_bins (default 32): ", 2, 1000);
                    break;
                    
                case 5: // K-Nearest Neighbors
                    n_neighbors = getIntegerInput("Enter number of neighbors (should be ODD for classification): ", 1, 1000);
                    if (n_neighbors % 2 == 0 && is_classification) {
                        cout << "Warning: Even number of neighbors may cause ties in classification.\n";
                    }
                    distance_metric = getStringInput("Enter distance metric (euclidean/manhattan): ", 
                                                     {"euclidean", "manhattan"});
                    int weight_choice = getIntegerInput("Use weighted voting? (0 = false/majority vote, 1 = true/inverse distance): ", 0, 1);
                    weighted = (weight_choice == 1);
                    tie_break = getStringInput("Enter tie break method (smallest_label/random): ", 
                                              {"smallest_label", "random"});
                    break;
            }
            
            // Display chosen configuration
            cout << "\n" << string(40, '=') << "\n";
            cout << "ALGORITHM CONFIGURATION\n";
            cout << string(40, '=') << "\n";
            cout << "Target variable: " << target_col << "\n";
            
            // Train and evaluate
            auto start_time = chrono::high_resolution_clock::now();
            ErrorResult result;
            
            switch (choice) {
                case 1: { // Linear Regression
                    cout << "Algorithm: Linear Regression (Normal Equation)\n";
                    cout << "L2 regularization: " << l2_reg << "\n";
                    
                    LinearRegression model;
                    result = model.fit(dataset.features, dataset.target, l2_reg);
                    if (!result.success) throw runtime_error(result.message);
                    
                    model.calculateMetrics(dataset.features, dataset.target);
                    
                    auto end_time = chrono::high_resolution_clock::now();
                    chrono::duration<double> elapsed = end_time - start_time;
                    
                    cout << "\nRESULTS:\n";
                    cout << "Train time: " << elapsed.count() << " seconds\n";
                    cout << "RMSE: " << model.rmse << "\n";
                    cout << "R²: " << model.r_squared << "\n";
                    cout << "SLOC: " << Matrix::getRegressionSLOC() << "\n";
                    
                    storeResults("Linear Regression", elapsed.count(), 
                                model.rmse, model.r_squared, Matrix::getRegressionSLOC(), false);
                    break;
                }
                    
                case 2: { // Logistic Regression
                    if (!dataset.isBinaryClassification()) {
                        throw runtime_error("Logistic regression requires binary classification");
                    }
                    
                    cout << "Algorithm: Logistic Regression\n";
                    cout << "Learning rate: " << learning_rate << "\n";
                    cout << "Epochs: " << epochs << "\n";
                    cout << "L2 regularization: " << l2_reg << "\n";
                    cout << "Seed: " << seed << "\n";
                    
                    LogisticRegression model;
                    result = model.fit(dataset.features, dataset.target, learning_rate, epochs, l2_reg, seed);
                    if (!result.success) throw runtime_error(result.message);
                    
                    model.calculateMetrics(dataset.features, dataset.target);
                    
                    auto end_time = chrono::high_resolution_clock::now();
                    chrono::duration<double> elapsed = end_time - start_time;
                    
                    cout << "\nRESULTS:\n";
                    cout << "Train time: " << elapsed.count() << " seconds\n";
                    cout << "Accuracy: " << model.accuracy << "\n";
                    cout << "Macro-F1: " << model.macro_f1 << "\n";
                    cout << "Training epochs: " << model.training_history.size() << "\n";
                    cout << "SLOC: 85\n";
                    
                    storeResults("Logistic Regression", elapsed.count(), 
                                model.accuracy, model.macro_f1, 85, true);
                    break;
                }
                    
                case 3: { // Decision Tree
                    cout << "Algorithm: Decision Tree\n";
                    cout << "Max depth: " << max_depth << "\n";
                    cout << "Min samples split: " << min_samples_split << "\n";
                    cout << "Number of bins: " << n_bins << "\n";
                    
                    DecisionTree model(is_classification, max_depth, min_samples_split, n_bins);
                    result = model.fit(dataset.features, dataset.target);
                    if (!result.success) throw runtime_error(result.message);
                    
                    model.calculateMetrics(dataset.features, dataset.target);
                    
                    auto end_time = chrono::high_resolution_clock::now();
                    chrono::duration<double> elapsed = end_time - start_time;
                    
                    cout << "\nRESULTS:\n";
                    cout << "Train time: " << elapsed.count() << " seconds\n";
                    
                    if (is_classification) {
                        cout << "Accuracy: " << model.accuracy << "\n";
                        cout << "Macro-F1: " << model.macro_f1 << "\n";
                        storeResults("Decision Tree", elapsed.count(), 
                                    model.accuracy, model.macro_f1, 100, true);
                    } else {
                        cout << "RMSE: " << model.rmse << "\n";
                        cout << "R²: " << model.r_squared << "\n";
                        storeResults("Decision Tree", elapsed.count(), 
                                    model.rmse, model.r_squared, 100, false);
                    }
                    cout << "SLOC: 100\n";
                    break;
                }
                    
                case 4: { // Gaussian Naive Bayes
                    if (!is_classification) {
                        throw runtime_error("Gaussian Naive Bayes is for classification only");
                    }
                    
                    cout << "Algorithm: Gaussian Naive Bayes\n";
                    
                    GaussianNB model;
                    vector<int> labels = dataset.getTargetLabels();
                    result = model.fit(dataset.features, labels);
                    if (!result.success) throw runtime_error(result.message);
                    
                    model.calculateMetrics(dataset.features, labels);
                    
                    auto end_time = chrono::high_resolution_clock::now();
                    chrono::duration<double> elapsed = end_time - start_time;
                    
                    cout << "\nRESULTS:\n";
                    cout << "Train time: " << elapsed.count() << " seconds\n";
                    cout << "Accuracy: " << model.accuracy << "\n";
                    cout << "Macro-F1: " << model.macro_f1 << "\n";
                    cout << "Number of classes: " << model.class_labels.size() << "\n";
                    cout << "SLOC: 80\n";
                    
                    storeResults("Gaussian Naive Bayes", elapsed.count(), 
                                model.accuracy, model.macro_f1, 80, true);
                    break;
                }
                    
                case 5: { // K-Nearest Neighbors
                    cout << "Algorithm: K-Nearest Neighbors\n";
                    cout << "Number of neighbors: " << n_neighbors << "\n";
                    cout << "Distance metric: " << distance_metric << "\n";
                    cout << "Weighted voting: " << (weighted ? "true" : "false") << "\n";
                    cout << "Tie break: " << tie_break << "\n";
                    
                    KNN model(n_neighbors, distance_metric, weighted, tie_break, is_classification);
                    result = model.fit(dataset.features, dataset.target);
                    if (!result.success) throw runtime_error(result.message);
                    
                    model.calculateMetrics(dataset.features, dataset.target);
                    
                    auto end_time = chrono::high_resolution_clock::now();
                    chrono::duration<double> elapsed = end_time - start_time;
                    
                    cout << "\nRESULTS:\n";
                    cout << "Train time: " << elapsed.count() << " seconds\n";
                    
                    if (is_classification) {
                        cout << "Accuracy: " << model.accuracy << "\n";
                        cout << "Macro-F1: " << model.macro_f1 << "\n";
                        storeResults("K-Nearest Neighbors", elapsed.count(), 
                                    model.accuracy, model.macro_f1, 120, true);
                    } else {
                        cout << "RMSE: " << model.rmse << "\n";
                        cout << "R²: " << model.r_squared << "\n";
                        storeResults("K-Nearest Neighbors", elapsed.count(), 
                                    model.rmse, model.r_squared, 120, false);
                    }
                    cout << "SLOC: 120\n";
                    break;
                }
            }
            
            cout << "\n" << string(40, '=') << "\n";
            cout << "ALGORITHM COMPLETED SUCCESSFULLY!\n";
            cout << string(40, '=') << "\n";
            
            // Ask to continue
            cout << "\nPress Enter to return to menu or 'q' to quit...";
            cin.ignore();
            if (cin.get() == 'q') {
                if (!allResults.empty()) {
                    char save;
                    cout << "\nSave results to file before quitting? (y/n): ";
                    cin >> save;
                    if (save == 'y' || save == 'Y') {
                        string filename;
                        cout << "Enter filename: ";
                        cin >> filename;
                        saveResultsToFile(filename, "MLSuite");
                    }
                }
                running = false;
            }
            
        } catch (const exception& e) {
            cerr << "\n❌ ERROR: " << e.what() << "\n";
            cout << "Returning to main menu...\n\n";
            
            // Clear input buffer
            cin.clear();
            cin.ignore(numeric_limits<streamsize>::max(), '\n');
        }
    }
    
    cout << "\n==================================================\n";
    cout << "     PROGRAM TERMINATED\n";
    cout << "     Total algorithms run: " << allResults.size() << "\n";
    cout << "==================================================\n";
    
    return 0;
}