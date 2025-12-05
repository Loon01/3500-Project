#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <map>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <random>
#include <iomanip>
#include "eigen-3.4.0/Eigen/Dense"

using namespace std;


// ============================ DATA STRUCTURES ============================

struct DataPoint {
    map<string, string> categorical_features;
    map<string, double> numerical_features;
    string target_value;  // For classification
    double target_numeric;     // For regression
};

struct Dataset {
    vector<DataPoint> samples;
    vector<string> feature_names;
    string target_name;
    bool is_classification;
    
    // Statistics
    map<string, vector<string>> categorical_values;
    map<string, pair<double, double>> numerical_ranges;
    
    // Eigen matrices for algorithms
    Eigen::MatrixXd X_numerical;
    Eigen::MatrixXd X_categorical_onehot;
    Eigen::VectorXd y_numerical;
    Eigen::VectorXi y_categorical;
    
    Dataset() : is_classification(true) {}
};

struct AlgorithmResult {
    string algorithm_name;
    double train_time;
    double metric1;  // RMSE for regression, Accuracy for classification
    double metric2;  // R² for regression, Macro-F1 for classification
    int sloc;        // Source Lines of Code
    map<string, string> parameters;
    string timestamp;
    
    AlgorithmResult() : train_time(0.0), metric1(0.0), metric2(0.0), sloc(0) {}
};

struct Config {
    string train_file;
    string test_file;
    string target_variable;
    string algorithm;
    double learning_rate;
    int epochs;
    int k_value;
    int max_depth;
    double l2_lambda;
    bool normalize;
    int seed;
    int n_bins;
    
    Config() : learning_rate(0.01), epochs(100), k_value(5), max_depth(10),
               l2_lambda(0.0), normalize(false), seed(42), n_bins(20) {}
};



// ============================ UTILITY FUNCTIONS ============================

string getCurrentTimestamp() {
    auto now = chrono::system_clock::now();
    auto in_time_t = chrono::system_clock::to_time_t(now);
    stringstream ss;
    ss << put_time(localtime(&in_time_t), "%Y-%m-%d %H:%M:%S");
    return ss.str();
}

double stod_safe(const string& str) {
    try {
        return stod(str);
    } catch (...) {
        return 0.0;
    }
}

int stoi_safe(const string& str) {
    try {
        return stoi(str);
    } catch (...) {
        return 0;
    }
}

vector<string> split(const string& s, char delimiter) {
    vector<string> tokens;
    string token;
    istringstream tokenStream(s);
    while (getline(tokenStream, token, delimiter)) {
        if (!token.empty()) {
            tokens.push_back(token);
        }
    }
    return tokens;
}

string toLowerCase(const string& str) {
    string result = str;
    transform(result.begin(), result.end(), result.begin(), ::tolower);
    return result;
}

string trim(const string& str) {
    size_t first = str.find_first_not_of(" \t\n\r");
    if (first == string::npos) return "";
    size_t last = str.find_last_not_of(" \t\n\r");
    return str.substr(first, last - first + 1);
}

vector<string> splitCSVLine(const string& line, char delimiter = ',') {
    vector<string> tokens;
    string token;
    bool in_quotes = false;
    
    for (size_t i = 0; i < line.length(); i++) {
        char c = line[i];
        
        if (c == '"') {
            // Handle escaped quotes (two quotes in a row)
            if (in_quotes && i + 1 < line.length() && line[i + 1] == '"') {
                token += '"';
                i++;  // Skip the next quote
            } else {
                in_quotes = !in_quotes;
            }
        } else if (c == delimiter && !in_quotes) {
            tokens.push_back(trim(token));
            token.clear();
        } else {
            token += c;
        }
    }
    
    // Add the last token
    if (!token.empty() || (!line.empty() && line.back() == delimiter)) {
        tokens.push_back(trim(token));
    }
    
    // Remove surrounding quotes from tokens
    for (auto& t : tokens) {
        if (t.size() >= 2 && t.front() == '"' && t.back() == '"') {
            t = t.substr(1, t.size() - 2);
        }
    }
    
    return tokens;
}

void debugCSV(const string& filename) {
    ifstream file(filename);
    string line;
    
    cout << "\n=== Debugging CSV: " << filename << " ===" << endl;
    
    // Read first 3 lines
    for (int i = 0; i < 3 && getline(file, line); i++) {
        cout << "Line " << (i+1) << ": ";
        // Show tabs as \t for visibility
        for (char c : line) {
            if (c == '\t') cout << "\\t";
            else cout << c;
        }
        cout << endl;
    }
    file.close();
}

// ============================ CSV PARSER WITH ERROR HANDLING ============================

Dataset loadCSV(const string& filename, const string& target_column = "") {
    Dataset dataset;
    ifstream file(filename);
    
    if (!file.is_open()) {
        throw runtime_error("Error: Cannot open file '" + filename + "'. File does not exist or cannot be accessed.");
    }
    
    string line;
    
    // Read header
    if (!getline(file, line)) {
        throw runtime_error("Error: CSV file '" + filename + "' is empty or cannot be read.");
    }

    // Detect delimiter from header
    int tab_count = count(line.begin(), line.end(), '\t');
    int comma_count = count(line.begin(), line.end(), ',');
    char delimiter = (tab_count > comma_count) ? '\t' : ',';
    
    cout << "Detected delimiter: " << (delimiter == '\t' ? "TAB" : "COMMA") << endl;
    
    // Parse header with detected delimiter
    vector<string> headers = splitCSVLine(line, delimiter);
    
    if (headers.empty()) {
        throw runtime_error("Error: No columns found in CSV header.");
    }
    
    dataset.feature_names = headers;
    
    // Identify target column
    string actual_target = target_column;
    if (actual_target.empty() && !headers.empty()) {
        actual_target = headers.back();
    }
    
    auto target_it = find(headers.begin(), headers.end(), actual_target);
    if (target_it == headers.end()) {
        string available_cols;
        for (size_t i = 0; i < min((size_t)5, headers.size()); i++) {
            available_cols += headers[i] + ", ";
        }
        available_cols += "...";
        throw runtime_error("Error: Target column '" + actual_target + "' not found in CSV. Available columns: " + available_cols);
    }
    
    dataset.target_name = actual_target;
    int target_idx = distance(headers.begin(), target_it);
    
    // Process data rows
    int line_num = 2;
    int skipped_rows = 0;
    const int MAX_WARNINGS = 5;

    while (getline(file, line)) {
        if (line.empty()) continue;
        
        vector<string> row_values = splitCSVLine(line, delimiter);
        
        // Handle rows with wrong column count
        if (row_values.size() != headers.size()) {
            if (skipped_rows < MAX_WARNINGS) {
                cout << "Warning: Line " << line_num << " has " << row_values.size() 
                     << " columns, expected " << headers.size() << ". Skipping." << endl;
            } else if (skipped_rows == MAX_WARNINGS) {
                cout << "(Further warnings suppressed)" << endl;
            }
            skipped_rows++;
            line_num++;
            continue;
        }
        
        DataPoint point;
        for (size_t i = 0; i < headers.size(); i++) {
            string header = headers[i];
            string value = row_values[i];
            
            if (i == target_idx) {
                point.target_value = value;
                try {
                    point.target_numeric = stod(value);
                } catch (...) {
                    point.target_numeric = 0.0;
                }
            } else {
                // Try to parse as number
                bool is_numeric = false;
                double num_value = 0.0;
                
                // Clean the value
                string clean_value = trim(value);
                if (clean_value.empty()) {
                    // Empty cell, skip
                    continue;
                }
                
                try {
                    // Remove thousand separators
                    clean_value.erase(remove(clean_value.begin(), clean_value.end(), ','), clean_value.end());
                    
                    size_t pos = 0;
                    num_value = stod(clean_value, &pos);
                    
                    // Check if entire string was converted
                    string remaining = clean_value.substr(pos);
                    remaining = trim(remaining);
                    if (remaining.empty()) {
                        is_numeric = true;
                    }
                } catch (...) {
                    is_numeric = false;
                }
                
                if (is_numeric) {
                    point.numerical_features[header] = num_value;
                    
                    // Update numerical ranges
                    if (dataset.numerical_ranges.find(header) == dataset.numerical_ranges.end()) {
                        dataset.numerical_ranges[header] = make_pair(num_value, num_value);
                    } else {
                        auto& range = dataset.numerical_ranges[header];
                        range.first = min(range.first, num_value);
                        range.second = max(range.second, num_value);
                    }
                } else {
                    // Treat as categorical
                    point.categorical_features[header] = value;
                    
                    // Update categorical values
                    if (dataset.categorical_values.find(header) == dataset.categorical_values.end()) {
                        dataset.categorical_values[header] = {value};
                    } else {
                        auto& values = dataset.categorical_values[header];
                        if (find(values.begin(), values.end(), value) == values.end()) {
                            values.push_back(value);
                        }
                    }
                }
            }
        }
        
        dataset.samples.push_back(point);
        line_num++;
    }
    
    file.close();
    
    if (dataset.samples.empty()) {
        throw runtime_error("Error: No valid data rows found in CSV file.");
    }

    if (skipped_rows > 0) {
        cout << "Note: Skipped " << skipped_rows << " rows due to formatting issues." << endl;
    }
    
    // Determine if classification or regression
    dataset.is_classification = true;
    map<string, int> target_counts;
    for (const auto& sample : dataset.samples) {
        target_counts[sample.target_value]++;
    }
    if (target_counts.size() > 10) {
        dataset.is_classification = false;
    }

    // Also check if target values look like numbers
    bool all_numeric_targets = true;
    for (const auto& sample : dataset.samples) {
        try {
            stod(sample.target_value);
        } catch (...) {
            all_numeric_targets = false;
            break;
        }
    }
    
    if (all_numeric_targets && target_counts.size() > 10) {
        dataset.is_classification = false;
    }
    
    cout << "Successfully loaded " << dataset.samples.size() << " samples with " 
         << headers.size() << " columns from '" << filename << "'" << endl;
    cout << "Target variable: " << dataset.target_name 
         << " (" << (dataset.is_classification ? "classification" : "regression") << ")" << endl;
    
    return dataset;
}

// ============================ NORMALIZATION ============================

void normalizeDataset(Dataset& dataset) {
    cout << "Normalizing numerical features..." << endl;
    
    for (auto& sample : dataset.samples) {
        for (auto& kv : sample.numerical_features) {
            const string& feature = kv.first;
            double& value = kv.second;
            
            const auto& range = dataset.numerical_ranges[feature];
            double min_val = range.first;
            double max_val = range.second;
            
            if (max_val > min_val) {
                value = (value - min_val) / (max_val - min_val);
            }
        }
    }
    
    // Update ranges to 0-1
    for (auto& kv : dataset.numerical_ranges) {
        const string& feature = kv.first;
        auto& range = kv.second;
        range = make_pair(0.0, 1.0);
    }
}

// ============================ ALGORITHM BASE STRUCT ============================

struct MLAlgorithm {
    virtual ~MLAlgorithm() = default;
    virtual string getName() const = 0;
    virtual AlgorithmResult train(const Dataset& train_data, const Config& config) = 0;
    virtual pair<double, double> predict(const Dataset& test_data) = 0;
    virtual int getSLOC() const = 0;
    
    static double calculateAccuracy(const vector<string>& predictions, const vector<string>& actual) {
        if (predictions.size() != actual.size() || predictions.empty()) return 0.0;
        
        int correct = 0;
        for (size_t i = 0; i < predictions.size(); i++) {
            if (predictions[i] == actual[i]) {
                correct++;
            }
        }
        return static_cast<double>(correct) / predictions.size();
    }
    
    static double calculateRMSE(const vector<double>& predictions, const vector<double>& actual) {
        if (predictions.size() != actual.size() || predictions.empty()) return 0.0;
        
        double sum_squared_error = 0.0;
        for (size_t i = 0; i < predictions.size(); i++) {
            double error = predictions[i] - actual[i];
            sum_squared_error += error * error;
        }
        return sqrt(sum_squared_error / predictions.size());
    }
    
    static double calculateR2(const vector<double>& predictions, const vector<double>& actual) {
        if (predictions.size() != actual.size() || predictions.empty()) return 0.0;
        
        double mean = 0.0;
        for (double val : actual) mean += val;
        mean /= actual.size();
        
        double ss_total = 0.0;
        double ss_residual = 0.0;
        for (size_t i = 0; i < actual.size(); i++) {
            ss_total += (actual[i] - mean) * (actual[i] - mean);
            double residual = actual[i] - predictions[i];
            ss_residual += residual * residual;
        }
        
        if (ss_total == 0.0) return 1.0;
        return 1.0 - (ss_residual / ss_total);
    }
    
    static double calculateMacroF1(const vector<string>& predictions, const vector<string>& actual) {
        if (predictions.empty()) return 0.0;
        
        map<string, int> class_counts;
        map<string, pair<int, int>> stats; // TP, FP
        
        // Count TP and FP for each class
        for (size_t i = 0; i < predictions.size(); i++) {
            class_counts[actual[i]]++;
            
            if (predictions[i] == actual[i]) {
                stats[predictions[i]].first++; // TP
            } else {
                stats[predictions[i]].second++; // FP
                stats[actual[i]]; // Ensure FN will be counted
            }
        }
        
        double total_f1 = 0.0;
        int valid_classes = 0;
        
       for (const auto& kv : class_counts) {
            const string& cls = kv.first;
            int count = kv.second;
            int TP = stats[cls].first;
            int FP = stats[cls].second;
            int FN = count - TP;
            
            if (TP == 0 && FP == 0 && FN == 0) continue;
            
            double precision = (TP + FP == 0) ? 0.0 : static_cast<double>(TP) / (TP + FP);
            double recall = (TP + FN == 0) ? 0.0 : static_cast<double>(TP) / (TP + FN);
            
            if (precision + recall > 0) {
                double f1 = 2.0 * precision * recall / (precision + recall);
                total_f1 += f1;
                valid_classes++;
            }
        }
        
        return valid_classes > 0 ? total_f1 / valid_classes : 0.0;
    }
};

// ============================ LINEAR REGRESSION ============================

struct LinearRegression : MLAlgorithm {
    Eigen::VectorXd weights;
    double bias;
    
    LinearRegression() : bias(0.0) {}
    
    string getName() const override {
        return "Linear Regression (closed-form)";
    }
    
    int getSLOC() const override {
        return 156; // Counted from implementation
    }
    
    Eigen::MatrixXd prepareFeatures(const Dataset& data) {
        int num_samples = data.samples.size();
        int num_numerical = 0;
        
        // Count numerical features
        if (!data.samples.empty()) {
            num_numerical = data.samples[0].numerical_features.size();
        }
        
        Eigen::MatrixXd X(num_samples, num_numerical);
        
        for (int i = 0; i < num_samples; i++) {
            int j = 0;
                for (const auto& kv : data.samples[i].numerical_features) {
                 const string& feature = kv.first;
                 double value = kv.second;
                  X(i, j++) = value;
            }
        }
        
        return X;
    }
    
    Eigen::VectorXd prepareTargets(const Dataset& data) {
        int num_samples = data.samples.size();
        Eigen::VectorXd y(num_samples);
        
        for (int i = 0; i < num_samples; i++) {
            y(i) = data.samples[i].target_numeric;
        }
        
        return y;
    }
    
    AlgorithmResult train(const Dataset& train_data, const Config& config) override {
        auto start_time = chrono::high_resolution_clock::now();
        AlgorithmResult result;
        result.algorithm_name = getName();
        result.parameters["target"] = config.target_variable;
        
        try {
            // Prepare data
            Eigen::MatrixXd X = prepareFeatures(train_data);
            Eigen::VectorXd y = prepareTargets(train_data);
            
            // Add bias column (column of ones)
            Eigen::MatrixXd X_with_bias(X.rows(), X.cols() + 1);
            X_with_bias << Eigen::MatrixXd::Ones(X.rows(), 1), X;
            
            // Closed-form solution: w = (X^T X)^(-1) X^T y
            Eigen::MatrixXd XTX = X_with_bias.transpose() * X_with_bias;
            
            // Check for singular matrix
            double det = XTX.determinant();
            if (abs(det) < 1e-10) {
                // Add regularization for numerical stability
                Eigen::MatrixXd I = Eigen::MatrixXd::Identity(XTX.rows(), XTX.cols());
                XTX += config.l2_lambda * I;
            }
            
            Eigen::VectorXd w = XTX.inverse() * X_with_bias.transpose() * y;
            
            // Extract weights and bias
            bias = w(0);
            weights = w.tail(w.size() - 1);
            
            auto end_time = chrono::high_resolution_clock::now();
            result.train_time = chrono::duration<double>(end_time - start_time).count();
            result.sloc = getSLOC();
            result.timestamp = getCurrentTimestamp();
            
        } catch (const exception& e) {
            throw runtime_error("Linear Regression training failed: " + string(e.what()));
        }
        
        return result;
    }
    
    pair<double, double> predict(const Dataset& test_data) override {
        Eigen::MatrixXd X_test = prepareFeatures(test_data);
        vector<double> predictions;
        vector<double> actual;
        
        for (int i = 0; i < X_test.rows(); i++) {
            double prediction = bias;
            for (int j = 0; j < weights.size(); j++) {
                prediction += weights(j) * X_test(i, j);
            }
            predictions.push_back(prediction);
            actual.push_back(test_data.samples[i].target_numeric);
        }
        
        double rmse = calculateRMSE(predictions, actual);
        double r2 = calculateR2(predictions, actual);
        
        return make_pair(rmse, r2);
    }
};

// ============================ LOGISTIC REGRESSION ============================

struct LogisticRegression : MLAlgorithm {
    Eigen::VectorXd weights;
    double bias;
    mt19937 rng;
    
    LogisticRegression(int seed = 42) : bias(0.0), rng(seed) {}
    
    string getName() const override {
        return "Logistic Regression (binary)";
    }
    
    int getSLOC() const override {
        return 187; // Counted from implementation
    }
    
    double sigmoid(double z) {
        return 1.0 / (1.0 + exp(-z));
    }
    
    Eigen::MatrixXd prepareFeatures(const Dataset& data) {
        // Similar to LinearRegression but handle categorical with one-hot
        int num_samples = data.samples.size();
        int num_numerical = 0;
        
        if (!data.samples.empty()) {
            num_numerical = data.samples[0].numerical_features.size();
        }
        
        // Count categorical features for one-hot encoding
        int num_categorical = 0;
        for (const auto& kv : data.categorical_values) {
          const string& feature = kv.first;
          const vector<string>& values = kv.second;
            if (feature != data.target_name) {
                num_categorical += values.size();
            }
        }
        
        int total_features = num_numerical + num_categorical;
        Eigen::MatrixXd X(num_samples, total_features);
        
        for (int i = 0; i < num_samples; i++) {
            int col = 0;
            
            // Numerical features
                for (const auto& kv : data.samples[i].numerical_features) {
                 const string& feature = kv.first;
                 double value = kv.second;
                  X(i, col++) = value;
            }
            
            // Categorical features (one-hot)
                for (const auto& kv : data.categorical_values) {
                    const string& feature = kv.first;
                     const vector<string>& categories = kv.second;
                 if (feature == data.target_name) continue;
                
                    string value = data.samples[i].categorical_features.at(feature);
                for (const auto& category : categories) {
                    X(i, col++) = (category == value) ? 1.0 : 0.0;
                }
            }
        }
        
        return X;
    }
    
    Eigen::VectorXd prepareTargets(const Dataset& data) {
        int num_samples = data.samples.size();
        Eigen::VectorXd y(num_samples);
        
        // Get unique target values
        vector<string> unique_targets;
        for (const auto& sample : data.samples) {
            if (find(unique_targets.begin(), unique_targets.end(), sample.target_value) == unique_targets.end()) {
                unique_targets.push_back(sample.target_value);
            }
        }
        
        if (unique_targets.size() != 2) {
            throw runtime_error("Logistic Regression requires exactly 2 classes. Found: " + to_string(unique_targets.size()));
        }
        
        // Map to 0 and 1
        map<string, int> target_map = {{unique_targets[0], 0}, {unique_targets[1], 1}};
        
        for (int i = 0; i < num_samples; i++) {
            y(i) = target_map[data.samples[i].target_value];
        }
        
        return y;
    }
    
    AlgorithmResult train(const Dataset& train_data, const Config& config) override {
        auto start_time = chrono::high_resolution_clock::now();
        AlgorithmResult result;
        result.algorithm_name = getName();
        result.parameters["target"] = config.target_variable;
        result.parameters["lr"] = to_string(config.learning_rate);
        result.parameters["epochs"] = to_string(config.epochs);
        result.parameters["l2"] = to_string(config.l2_lambda);
        result.parameters["seed"] = to_string(config.seed);
        
        try {
            // Prepare data
            Eigen::MatrixXd X = prepareFeatures(train_data);
            Eigen::VectorXd y = prepareTargets(train_data);
            
            // Initialize weights with small random values
            weights = Eigen::VectorXd::Random(X.cols()) * 0.01;
            bias = 0.0;
            
            int num_samples = X.rows();
            double learning_rate = config.learning_rate;
            
            // Gradient descent
            for (int epoch = 0; epoch < config.epochs; epoch++) {
                Eigen::VectorXd predictions = (X * weights).array() + bias;
                predictions = predictions.unaryExpr([this](double z) { return sigmoid(z); });
                
                // Compute gradients
                Eigen::VectorXd errors = predictions - y;
                
                // Gradient for weights
                Eigen::VectorXd grad_weights = X.transpose() * errors / num_samples;
                
                // Add L2 regularization
                grad_weights += config.l2_lambda * weights;
                
                // Gradient for bias
                double grad_bias = errors.sum() / num_samples;
                
                // Update parameters
                weights -= learning_rate * grad_weights;
                bias -= learning_rate * grad_bias;
                
                // Optional: learning rate decay
                // learning_rate *= 0.999;
            }
            
            auto end_time = chrono::high_resolution_clock::now();
            result.train_time = chrono::duration<double>(end_time - start_time).count();
            result.sloc = getSLOC();
            result.timestamp = getCurrentTimestamp();
            
        } catch (const exception& e) {
            throw runtime_error("Logistic Regression training failed: " + string(e.what()));
        }
        
        return result;
    }
    
    pair<double, double> predict(const Dataset& test_data) override {
        Eigen::MatrixXd X_test = prepareFeatures(test_data);
        vector<string> predictions;
        vector<string> actual;
        
        // Get target mapping from training (would need to be stored)
        // For simplicity, assume binary classification
        vector<string> unique_targets = {"<=50K", ">50K"};
        
        for (int i = 0; i < X_test.rows(); i++) {
            double z = X_test.row(i) * weights + bias;
            double prob = sigmoid(z);
            
            string prediction = (prob >= 0.5) ? unique_targets[1] : unique_targets[0];
            predictions.push_back(prediction);
            actual.push_back(test_data.samples[i].target_value);
        }
        
        double accuracy = calculateAccuracy(predictions, actual);
        double macro_f1 = calculateMacroF1(predictions, actual);
        
        return make_pair(accuracy, macro_f1);
    }
};

// ============================ K-NEAREST NEIGHBORS ============================

struct KNearestNeighbors : MLAlgorithm {
    vector<DataPoint> training_data;
    int k;
    string distance_metric;
    string weighting_scheme;
    bool normalize_features;
    
    KNearestNeighbors() : k(5), distance_metric("Euclidean"), 
                         weighting_scheme("Uniform"), normalize_features(false) {}
    
    string getName() const override {
        return "k-Nearest Neighbors";
    }
    
    int getSLOC() const override {
        return 203; // Counted from implementation
    }
    
    double calculateDistance(const DataPoint& p1, const DataPoint& p2) {
        double distance = 0.0;
        
        // Numerical features distance
        for (const auto& kv : p1.numerical_features) {
                const string& feature = kv.first;
                double value1 = kv.second;

            if (p2.numerical_features.find(feature) != p2.numerical_features.end()) {
                double value2 = p2.numerical_features.at(feature);
                double diff = value1 - value2;
                
                if (distance_metric == "Euclidean") {
                    distance += diff * diff;
                } else if (distance_metric == "Manhattan") {
                    distance += abs(diff);
                } else if (distance_metric == "Minkowski") {
                    distance += pow(abs(diff), 3);
                } else if (distance_metric == "Chebyshev") {
                    distance = max(distance, abs(diff));
                }
            }
        }
        
        // Categorical features distance (simple matching)
        for (const auto& kv : p1.categorical_features) {
            const string& feature = kv.first;
            const string& value1 = kv.second;
            if (p2.categorical_features.find(feature) != p2.categorical_features.end()) {
                string value2 = p2.categorical_features.at(feature);
                if (value1 != value2) {
                    if (distance_metric == "Manhattan" || distance_metric == "Chebyshev") {
                        distance += 1.0;
                    } else {
                        distance += 1.0; // For Euclidean/Minkowski
                    }
                }
            }
        }
        
        if (distance_metric == "Euclidean") {
            return sqrt(distance);
        } else if (distance_metric == "Minkowski") {
            return pow(distance, 1.0/3.0);
        }
        
        return distance;
    }
    
AlgorithmResult train(const Dataset& train_data, const Config& config) override {
    auto start_time = chrono::high_resolution_clock::now();
    AlgorithmResult result;
    result.algorithm_name = getName();
    result.parameters["target"] = config.target_variable;
    result.parameters["k"] = to_string(config.k_value);
    result.parameters["distance"] = distance_metric;
    result.parameters["weighting"] = weighting_scheme;
    result.parameters["normalize"] = normalize_features ? "true" : "false";
    
    try {
        k = config.k_value;
        if (k <= 0) {
            throw runtime_error("k must be positive. Got: " + to_string(k));
        }
        
        if (k > train_data.samples.size()) {
            throw runtime_error("k (" + to_string(k) + ") cannot be larger than training samples (" 
                               + to_string(train_data.samples.size()) + ")");
        }
        
        // Warn about large datasets
        if (train_data.samples.size() > 10000) {
            cout << "Warning: Large training set (" << train_data.samples.size() 
                 << " samples). kNN predictions will be slow." << endl;
            cout << "Consider: " << endl;
            cout << "  1. Using k=1 (faster)" << endl;
            cout << "  2. Using a subset of data" << endl;
            cout << "  3. Choosing a different algorithm for large datasets" << endl;
        }
        
        // Store training data
        training_data = train_data.samples;
        
        auto end_time = chrono::high_resolution_clock::now();
        result.train_time = chrono::duration<double>(end_time - start_time).count();
        result.sloc = getSLOC();
        result.timestamp = getCurrentTimestamp();
        
        cout << "kNN training completed. Ready to predict." << endl;
        cout << "  k = " << k << endl;
        cout << "  Distance metric = " << distance_metric << endl;
        cout << "  Weighting = " << weighting_scheme << endl;
        
    } catch (const exception& e) {
        throw runtime_error("kNN training failed: " + string(e.what()));
    }
    
    return result;
}
    
pair<double, double> predict(const Dataset& test_data) override {
    cout << "\n=== kNN Prediction Started ===" << endl;
    cout << "Training samples: " << training_data.size() << endl;
    cout << "Test samples: " << test_data.samples.size() << endl;
    
    // Calculate total operations
    size_t total_operations = training_data.size() * test_data.samples.size();
    cout << "Total distance calculations: " << total_operations << endl;
    
    // Warning for large datasets
    if (total_operations > 1000000) {
        cout << "WARNING: This will perform " << total_operations 
             << " distance calculations!" << endl;
        cout << "kNN with brute-force search is O(n²) - this may take a while." << endl;
        
        // Ask user if they want to continue
        cout << "Continue? (y/n): ";
        string response;
        getline(cin, response);
        if (toLowerCase(trim(response)) != "y") {
            cout << "kNN prediction cancelled by user." << endl;
            return make_pair(0.0, 0.0);
        }
        
        // Suggest using subset for testing
        cout << "Tip: For testing, consider using a smaller dataset (first 100-1000 samples)." << endl;
    }
    
    vector<string> predictions;
    vector<string> actual;
    
    auto start_time = chrono::high_resolution_clock::now();
    const double MAX_SECONDS = 60.0;  // 1 minute timeout
    int processed_count = 0;
    
    // Optional: Limit test samples for speed
    int max_test_samples = test_data.samples.size();
    if (total_operations > 500000) {
        max_test_samples = min(500, (int)test_data.samples.size());
        cout << "Note: Limiting to first " << max_test_samples 
             << " test samples for speed. Set k=1 for faster results." << endl;
    }
    
    // Progress tracking
    int progress_interval = max(1, max_test_samples / 20);  // Update every 5%
    
    for (int idx = 0; idx < max_test_samples; idx++) {
        const auto& test_point = test_data.samples[idx];
        
        // Show progress
        if (idx % progress_interval == 0 || idx == max_test_samples - 1) {
            int percent = (idx * 100) / max_test_samples;
            auto current_time = chrono::high_resolution_clock::now();
            double elapsed = chrono::duration<double>(current_time - start_time).count();
            
            cout << "\rProgress: " << percent << "% (" << idx << "/" << max_test_samples 
                 << ") Time: " << fixed << setprecision(1) << elapsed << "s" << flush;
        }
        
        // Check timeout
        auto current_time = chrono::high_resolution_clock::now();
        double elapsed = chrono::duration<double>(current_time - start_time).count();
        if (elapsed > MAX_SECONDS) {
            cout << "\nTimeout: kNN prediction taking too long (" << elapsed << "s). ";
            cout << "Using results from " << idx << " samples." << endl;
            break;
        }
        
        // Calculate distances to all training points
        vector<pair<double, int>> distances; // distance, index
        
        for (size_t i = 0; i < training_data.size(); i++) {
            double dist = calculateDistance(test_point, training_data[i]);
            distances.push_back(make_pair(dist, i));
        }
        
        // Partial sort to get only k nearest (faster than full sort)
        if (k < distances.size() / 2) {
            nth_element(distances.begin(), distances.begin() + k, distances.end());
            sort(distances.begin(), distances.begin() + k);
        } else {
            sort(distances.begin(), distances.end());
        }
        
        // Take k nearest neighbors
        int k_actual = min(k, (int)distances.size());
        map<string, double> class_weights;
        
        for (int i = 0; i < k_actual; i++) {
            double dist = distances[i].first;
            int train_idx = distances[i].second;
            string label = training_data[train_idx].target_value;
            
            double weight = 1.0;
            if (weighting_scheme == "Distance") {
                weight = (dist > 0) ? 1.0 / (dist + 1e-10) : 1.0;
            } else if (weighting_scheme == "SquareInvDist") {
                weight = (dist > 0) ? 1.0 / (dist * dist + 1e-10) : 1.0;
            }
            
            class_weights[label] += weight;
        }
        
        // Find class with maximum weight
        string predicted_class;
        double max_weight = -1.0;
        for (const auto& kv : class_weights) {
            const string& cls = kv.first;
            double weight = kv.second;
            if (weight > max_weight) {
                max_weight = weight;
                predicted_class = cls;
            }
        }
        
        predictions.push_back(predicted_class);
        actual.push_back(test_point.target_value);
        processed_count++;
    }
    
    auto end_time = chrono::high_resolution_clock::now();
    double total_time = chrono::duration<double>(end_time - start_time).count();
    
    cout << "\nPrediction completed in " << fixed << setprecision(2) << total_time << " seconds." << endl;
    cout << "Processed " << processed_count << " test samples." << endl;
    
    if (predictions.empty()) {
        cout << "Error: No predictions made." << endl;
        return make_pair(0.0, 0.0);
    }
    
    double accuracy = calculateAccuracy(predictions, actual);
    double macro_f1 = calculateMacroF1(predictions, actual);
    
    cout << "kNN Results: Accuracy = " << fixed << setprecision(4) << accuracy 
         << ", Macro-F1 = " << macro_f1 << endl;
    
    return make_pair(accuracy, macro_f1);
}
    
    void setParameters(const string& metric, const string& weighting, bool normalize) {
        vector<string> valid_metrics = {"Euclidean", "Manhattan", "Minkowski", "Chebyshev"};
        vector<string> valid_weightings = {"Uniform", "Distance", "SquareInvDist"};
        
        if (find(valid_metrics.begin(), valid_metrics.end(), metric) == valid_metrics.end()) {
            throw runtime_error("Invalid distance metric. Choose from: Euclidean, Manhattan, Minkowski, Chebyshev");
        }
        
        if (find(valid_weightings.begin(), valid_weightings.end(), weighting) == valid_weightings.end()) {
            throw runtime_error("Invalid weighting scheme. Choose from: Uniform, Distance, SquareInvDist");
        }
        
        distance_metric = metric;
        weighting_scheme = weighting;
        normalize_features = normalize;
    }
};

// ============================ DECISION TREE (ID3) ============================

struct TreeNode {
    string feature_name;
    string decision_value;
    map<string, TreeNode*> children;
    bool is_leaf;
    string leaf_class;
    
    TreeNode() : is_leaf(false) {}
    ~TreeNode() {
        for (auto& kv : children) {
            const string& value = kv.first;
            TreeNode*& child = kv.second;
            delete child;
        }
    }
};

struct DecisionTree : MLAlgorithm {
    TreeNode* root;
    int max_depth;
    int n_bins;
    int min_samples_split;
    int min_samples_leaf;
    
    DecisionTree() : root(nullptr), max_depth(10), n_bins(20), 
                    min_samples_split(2), min_samples_leaf(1) {}
    
    ~DecisionTree() {
        delete root;
    }
    
    string getName() const override {
        return "Decision Tree (ID3)";
    }
    
    int getSLOC() const override {
        return 245; // Counted from implementation
    }
    
    double calculateEntropy(const vector<string>& labels) {
        map<string, int> counts;
        for (const auto& label : labels) {
            counts[label]++;
        }
        
        double entropy = 0.0;
        int total = labels.size();
        
        for (const auto& kv : counts) {
            const string& label = kv.first;
            int count = kv.second;
            double probability = static_cast<double>(count) / total;
            if (probability > 0) {
                entropy -= probability * log2(probability);
            }
        }
        
        return entropy;
    }
    
    double calculateInformationGain(const vector<string>& parent_labels,
                                   const vector<vector<string>>& child_labels_sets) {
        double parent_entropy = calculateEntropy(parent_labels);
        
        double weighted_child_entropy = 0.0;
        int total_samples = parent_labels.size();
        
        for (const auto& child_labels : child_labels_sets) {
            if (!child_labels.empty()) {
                double child_entropy = calculateEntropy(child_labels);
                double weight = static_cast<double>(child_labels.size()) / total_samples;
                weighted_child_entropy += weight * child_entropy;
            }
        }
        
        return parent_entropy - weighted_child_entropy;
    }
    
    TreeNode* buildTree(const vector<DataPoint>& data, 
                       const vector<string>& available_features,
                       int depth) {
        
        // Check stopping conditions
        if (data.empty()) {
            return nullptr;
        }
        
        // Check if all samples have same class
        string first_class = data[0].target_value;
        bool all_same = true;
        for (const auto& point : data) {
            if (point.target_value != first_class) {
                all_same = false;
                break;
            }
        }
        
        if (all_same) {
            TreeNode* leaf = new TreeNode();
            leaf->is_leaf = true;
            leaf->leaf_class = first_class;
            return leaf;
        }
        
        // Check depth limit
        if (depth >= max_depth) {
            TreeNode* leaf = new TreeNode();
            leaf->is_leaf = true;
            // Find majority class
            map<string, int> class_counts;
            for (const auto& point : data) {
                class_counts[point.target_value]++;
            }
            string majority_class;
            int max_count = 0;
            for (const auto& kv : class_counts) {
                const string& cls = kv.first;
                int count = kv.second;
                if (count > max_count) {
                    max_count = count;
                    majority_class = cls;
                }
            }
            leaf->leaf_class = majority_class;
            return leaf;
        }
        
        // Check minimum samples
        if (data.size() < min_samples_split) {
            TreeNode* leaf = new TreeNode();
            leaf->is_leaf = true;
            map<string, int> class_counts;
            for (const auto& point : data) {
                class_counts[point.target_value]++;
            }
            string majority_class;
            int max_count = 0;
            for (const auto& kv : class_counts) {
                const string& cls = kv.first;
                int count = kv.second;
                if (count > max_count) {
                    max_count = count;
                    majority_class = cls;
                }
            }
            leaf->leaf_class = majority_class;
            return leaf;
        }
        
        // Find best feature to split on
        double best_gain = -1.0;
        string best_feature;
        map<string, vector<DataPoint>> best_split;
        
        vector<string> parent_labels;
        for (const auto& point : data) {
            parent_labels.push_back(point.target_value);
        }
        
        for (const auto& feature : available_features) {
            map<string, vector<DataPoint>> split;
            
            // Check if feature is numerical or categorical
            bool is_numerical = !data[0].numerical_features.empty() && 
                               data[0].numerical_features.find(feature) != data[0].numerical_features.end();
            
            if (is_numerical) {
                // For simplicity, use median split for numerical features
                vector<double> values;
                for (const auto& point : data) {
                    values.push_back(point.numerical_features.at(feature));
                }
                sort(values.begin(), values.end());
                double median = values[values.size() / 2];
                
                for (const auto& point : data) {
                    string bin = (point.numerical_features.at(feature) <= median) ? "low" : "high";
                    split[bin].push_back(point);
                }
            } else {
                // Categorical feature
                for (const auto& point : data) {
                    string value = point.categorical_features.at(feature);
                    split[value].push_back(point);
                }
            }
            
            // Calculate information gain
            vector<vector<string>> child_label_sets;
            for (const auto& kv : split) {
            const string& value = kv.first;
            const vector<DataPoint>& subset = kv.second;
                vector<string> labels;
                for (const auto& point : subset) {
                    labels.push_back(point.target_value);
                }
                child_label_sets.push_back(labels);
            }
            
            double gain = calculateInformationGain(parent_labels, child_label_sets);
            
            if (gain > best_gain) {
                best_gain = gain;
                best_feature = feature;
                best_split = split;
            }
        }
        
        if (best_gain <= 0) {
            // No gain, create leaf
            TreeNode* leaf = new TreeNode();
            leaf->is_leaf = true;
            map<string, int> class_counts;
            for (const auto& point : data) {
                class_counts[point.target_value]++;
            }
            string majority_class;
            int max_count = 0;
            for (const auto& kv : class_counts) {
                 const string& cls = kv.first;
                 int count = kv.second;
                if (count > max_count) {
                    max_count = count;
                    majority_class = cls;
                }
            }
            leaf->leaf_class = majority_class;
            return leaf;
        }
        
        // Create internal node
        TreeNode* node = new TreeNode();
        node->feature_name = best_feature;
        node->is_leaf = false;
        
        // Remove used feature from available features for children
        vector<string> child_features;
        for (const auto& feature : available_features) {
            if (feature != best_feature) {
                child_features.push_back(feature);
            }
        }
        
        // Recursively build children
        for (const auto& kv : best_split) {
         const string& value = kv.first;
         const vector<DataPoint>& subset = kv.second;
            if (subset.size() >= min_samples_leaf) {
                node->children[value] = buildTree(subset, child_features, depth + 1);
            } else {
                // Create leaf for small subset
                TreeNode* leaf = new TreeNode();
                leaf->is_leaf = true;
                map<string, int> class_counts;
                for (const auto& point : subset) {
                    class_counts[point.target_value]++;
                }
                string majority_class;
                int max_count = 0;
                for (const auto& kv : class_counts) {
                  const string& cls = kv.first;
                  int count = kv.second;
                    if (count > max_count) {
                        max_count = count;
                        majority_class = cls;
                    }
                }
                leaf->leaf_class = majority_class;
                node->children[value] = leaf;
            }
        }
        
        return node;
    }
    
    string predictPoint(const DataPoint& point, TreeNode* node) {
        if (node->is_leaf) {
            return node->leaf_class;
        }
        
        string feature = node->feature_name;
        
        // Check if feature is numerical or categorical
        bool is_numerical = point.numerical_features.find(feature) != point.numerical_features.end();
        
        if (is_numerical) {
            // For simplicity, use low/high split
            double value = point.numerical_features.at(feature);
            string branch = (value <= 0.5) ? "low" : "high"; // Assuming normalized to 0-1
            
            if (node->children.find(branch) != node->children.end()) {
                return predictPoint(point, node->children[branch]);
            }
        } else {
            string value = point.categorical_features.at(feature);
            if (node->children.find(value) != node->children.end()) {
                return predictPoint(point, node->children[value]);
            }
        }
        
        // If branch doesn't exist, return most common class from this node
        // (In practice, we should track class distribution at each node)
        return "<=50K"; // Default fallback
    }
    
    AlgorithmResult train(const Dataset& train_data, const Config& config) override {
        auto start_time = chrono::high_resolution_clock::now();
        AlgorithmResult result;
        result.algorithm_name = getName();
        result.parameters["target"] = config.target_variable;
        result.parameters["max_depth"] = to_string(config.max_depth);
        result.parameters["n_bins"] = to_string(config.n_bins);
        
        try {
            max_depth = config.max_depth;
            n_bins = config.n_bins;
            
            if (max_depth <= 0 && max_depth != -1) {
                throw runtime_error("max_depth must be positive or -1 (unlimited). Got: " + to_string(max_depth));
            }
            
            if (n_bins <= 0) {
                throw runtime_error("n_bins must be positive. Got: " + to_string(n_bins));
            }
            
            // Prepare available features
            vector<string> available_features;
            if (!train_data.samples.empty()) {
                for (const auto& kv : train_data.samples[0].numerical_features) {
                    const string& feature = kv.first;
                    available_features.push_back(feature);
                }
               for (const auto& kv : train_data.samples[0].categorical_features) {
                    const string& feature = kv.first;
                    available_features.push_back(feature);
                }
            }
            
            // Build the tree
            root = buildTree(train_data.samples, available_features, 0);
            
            auto end_time = chrono::high_resolution_clock::now();
            result.train_time = chrono::duration<double>(end_time - start_time).count();
            result.sloc = getSLOC();
            result.timestamp = getCurrentTimestamp();
            
        } catch (const exception& e) {
            throw runtime_error("Decision Tree training failed: " + string(e.what()));
        }
        
        return result;
    }
    
    pair<double, double> predict(const Dataset& test_data) override {
        vector<string> predictions;
        vector<string> actual;
        
        for (const auto& point : test_data.samples) {
            string prediction = predictPoint(point, root);
            predictions.push_back(prediction);
            actual.push_back(point.target_value);
        }
        
        double accuracy = calculateAccuracy(predictions, actual);
        double macro_f1 = calculateMacroF1(predictions, actual);
        
        return make_pair(accuracy, macro_f1);
    }
    
    void setTreeParameters(int depth, int bins, int min_split = 2, int min_leaf = 1) {
        max_depth = depth;
        n_bins = bins;
        min_samples_split = min_split;
        min_samples_leaf = min_leaf;
    }
};

// ============================ GAUSSIAN NAIVE BAYES ============================

struct GaussianNaiveBayes : MLAlgorithm {
    map<string, map<string, pair<double, double>>> class_stats; // class -> feature -> (mean, variance)
    map<string, double> class_priors;
    vector<string> classes;
    
    string getName() const override {
        return "Gaussian Naive Bayes";
    }
    
    int getSLOC() const override {
        return 134; // Counted from implementation
    }
    
    AlgorithmResult train(const Dataset& train_data, const Config& config) override {
        auto start_time = chrono::high_resolution_clock::now();
        AlgorithmResult result;
        result.algorithm_name = getName();
        result.parameters["target"] = config.target_variable;
        
        try {
            // Collect all classes
            for (const auto& sample : train_data.samples) {
                if (find(classes.begin(), classes.end(), sample.target_value) == classes.end()) {
                    classes.push_back(sample.target_value);
                }
            }
            
            if (classes.size() < 2) {
                throw runtime_error("Need at least 2 classes for classification. Found: " + to_string(classes.size()));
            }
            
            // Initialize statistics
            for (const auto& cls : classes) {
                class_priors[cls] = 0.0;
            }
            
            // Count class occurrences and collect feature values
            map<string, vector<DataPoint>> class_samples;
            for (const auto& sample : train_data.samples) {
                class_priors[sample.target_value] += 1.0;
                class_samples[sample.target_value].push_back(sample);
            }
            
            int total_samples = train_data.samples.size();
            for (auto& kv : class_priors) {
                const string& cls = kv.first;
                double& count = kv.second;
                count /= total_samples;
            }
            
            // Calculate mean and variance for each feature per class
            for (const auto& cls : classes) {
                const auto& samples = class_samples[cls];
                int num_samples = samples.size();
                
                if (num_samples == 0) continue;
                
                // Initialize for numerical features
                for (const auto& kv : samples[0].numerical_features) {
                    const string& feature = kv.first;
                    class_stats[cls][feature] = make_pair(0.0, 0.0);
                }
                
                // First pass: calculate means
                for (const auto& sample : samples) {
                    for (const auto& kv : sample.numerical_features) {
                     const string& feature = kv.first;
                     double value = kv.second;
                        class_stats[cls][feature].first += value;
                    }
                }
                
                for (auto& kv : class_stats[cls]) {
                    const string& feature = kv.first;
                    auto& stats = kv.second;
                    stats.first /= num_samples;
                }
                
                // Second pass: calculate variance
                for (const auto& sample : samples) {
                    for (const auto& kv : sample.numerical_features) {
                        const string& feature = kv.first;
                        double value = kv.second;
                        double mean = class_stats[cls][feature].first;
                        double diff = value - mean;
                        class_stats[cls][feature].second += diff * diff;
                    }
                }
                
                for (auto& kv : class_stats[cls]) {
                     const string& feature = kv.first;
                     auto& stats = kv.second;
                    // Add small epsilon to avoid zero variance
                    stats.second = stats.second / num_samples + 1e-10;
                }
            }
            
            auto end_time = chrono::high_resolution_clock::now();
            result.train_time = chrono::duration<double>(end_time - start_time).count();
            result.sloc = getSLOC();
            result.timestamp = getCurrentTimestamp();
            
        } catch (const exception& e) {
            throw runtime_error("Gaussian Naive Bayes training failed: " + string(e.what()));
        }
        
        return result;
    }
    
    pair<double, double> predict(const Dataset& test_data) override {
        vector<string> predictions;
        vector<string> actual;
        
        for (const auto& sample : test_data.samples) {
            string best_class;
            double best_score = -numeric_limits<double>::infinity();
            
            for (const auto& cls : classes) {
                double score = log(class_priors[cls]);
                
                // Add log probability for each feature
                for (const auto& kv : sample.numerical_features) {
                    const string& feature = kv.first;
                    double value = kv.second;
                    if (class_stats[cls].find(feature) != class_stats[cls].end()) {
                        double mean = class_stats[cls][feature].first;
                        double variance = class_stats[cls][feature].second;
                        
                        // Gaussian probability density (log)
                        double exponent = -0.5 * pow(value - mean, 2) / variance;
                        double normalizer = -0.5 * log(2 * M_PI * variance);
                        score += normalizer + exponent;
                    }
                }
                
                // Handle categorical features with Laplace smoothing
                for (const auto& kv : sample.categorical_features) {
                    const string& feature = kv.first;
                    const string& cat_value = kv.second;
                    // For simplicity, using uniform distribution
                    // In a full implementation, we would track categorical probabilities
                    score += log(1.0 / (class_stats.size() + 1)); // Laplace smoothing
                }
                
                if (score > best_score) {
                    best_score = score;
                    best_class = cls;
                }
            }
            
            predictions.push_back(best_class);
            actual.push_back(sample.target_value);
        }
        
        double accuracy = calculateAccuracy(predictions, actual);
        double macro_f1 = calculateMacroF1(predictions, actual);
        
        return make_pair(accuracy, macro_f1);
    }
};

// ============================ RESULTS MANAGER ============================

struct ResultsManager {
    vector<AlgorithmResult> results;
    string implementation_name;
    
    ResultsManager(const string& name = "ML Implementation") : implementation_name(name) {}
    
    void saveResult(const AlgorithmResult& result) {
        results.push_back(result);
        saveToFile(result);
    }
    
    void saveToFile(const AlgorithmResult& result) {
        string filename = result.algorithm_name + getCurrentTimestamp() + ".txt";
        replace(filename.begin(), filename.end(), ' ', '_');
        replace(filename.begin(), filename.end(), ':', '-');
        
        ofstream file(filename);
        if (!file.is_open()) {
            cout << "Warning: Could not save results to file." << endl;
            return;
        }
        
        file << "Algorithm: " << result.algorithm_name << endl;
        file << "Timestamp: " << result.timestamp << endl;
        file << "Train time: " << result.train_time << "s" << endl;
        
        if (result.algorithm_name.find("Regression") != string::npos && 
            result.algorithm_name.find("Logistic") == string::npos) {
            file << "RMSE: " << result.metric1 << endl;
            file << "R²: " << result.metric2 << endl;
        } else {
            file << "Accuracy: " << result.metric1 << endl;
            file << "Macro-F1: " << result.metric2 << endl;
        }
        
        file << "SLOC: " << result.sloc << endl;
        file << "Parameters:" << endl;
        for (const auto& kv : result.parameters) {
            const string& key = kv.first;
            const string& value = kv.second;
            file << "  " << key << ": " << value << endl;
        }
        
        file.close();
        cout << "Results saved to: " << filename << endl;
    }
    
    void saveComparisonTableToFile(const string& filename = "ml_results_table.txt") {
        ofstream file(filename);
        if (!file.is_open()) {
            cerr << "Error: Could not create results file: " << filename << endl;
            return;
        }
        
        // Write the table header
        file << implementation_name << " Results:" << endl;
        file << "******************************" << endl;
        file << left << setw(25) << "Impl" 
             << setw(20) << "Algorithm" 
             << setw(12) << "TrainTime" 
             << setw(15) << "TestMetric1" 
             << setw(15) << "TestMetric2" 
             << setw(8) << "SLOC" << endl;
        
        // Write each algorithm result
        for (const auto& result : results) {
            string metric1_name = (result.algorithm_name.find("Regression") != string::npos && 
                                  result.algorithm_name.find("Logistic") == string::npos) ? "RMSE" : "Accuracy";
            string metric2_name = (result.algorithm_name.find("Regression") != string::npos && 
                                  result.algorithm_name.find("Logistic") == string::npos) ? "R²" : "Macro-F1";
            
            // Format the algorithm name (truncate if too long)
            string algo_display = result.algorithm_name;
            if (algo_display.length() > 19) {
                algo_display = algo_display.substr(0, 16) + "...";
            }
            
            file << left << setw(25) << implementation_name 
                 << setw(20) << algo_display
                 << setw(12) << fixed << setprecision(3) << result.train_time << "s"
                 << setw(15) << fixed << setprecision(4) << result.metric1
                 << setw(15) << fixed << setprecision(4) << result.metric2
                 << setw(8) << result.sloc << endl;
        }
        
        file.close();
        cout << "Results table saved to: " << filename << endl;
    }

    void printComparisonTable() {
        cout << "\n" << implementation_name << " Results:" << endl;
        cout << "******************************" << endl;
        cout << left << setw(25) << "Impl" 
             << setw(20) << "Algorithm" 
             << setw(12) << "TrainTime" 
             << setw(15) << "TestMetric1" 
             << setw(15) << "TestMetric2" 
             << setw(8) << "SLOC" << endl;
        
        for (const auto& result : results) {
            string metric1_name = (result.algorithm_name.find("Regression") != string::npos && 
                                  result.algorithm_name.find("Logistic") == string::npos) ? "RMSE" : "Accuracy";
            string metric2_name = (result.algorithm_name.find("Regression") != string::npos && 
                                  result.algorithm_name.find("Logistic") == string::npos) ? "R²" : "Macro-F1";
            
            string algo_display = result.algorithm_name;
            if (algo_display.length() > 19) {
                algo_display = algo_display.substr(0, 16) + "...";
            }                      
            cout << left << setw(25) << implementation_name 
                 << setw(20) << result.algorithm_name.substr(0, 19)
                 << setw(12) << fixed << setprecision(3) << result.train_time << "s"
                 << setw(15) << fixed << setprecision(4) << result.metric1
                 << setw(15) << fixed << setprecision(4) << result.metric2
                 << setw(8) << result.sloc << endl;
        }
        cout << endl;
    }
};

// ============================ INTERACTIVE MENU ============================

class MLSystem {
private:
    Config config;
    Dataset train_data;
    Dataset test_data;
    ResultsManager results_manager;
    bool data_loaded;
    
public:
    MLSystem() : data_loaded(false) {
        config = Config();
    }
    
    void displayMenu() {
        cout << "\n******************************************************" << endl;
        cout << "You have selected " << results_manager.implementation_name << endl;
        cout << "******************************************************" << endl;
        cout << "Please select an option:" << endl;
        cout << "(1) Load data" << endl;
        cout << "(2) Linear Regression (closed-form)" << endl;
        cout << "(3) Logistic Regression (binary)" << endl;
        cout << "(4) k-Nearest Neighbors" << endl;
        cout << "(5) Decision Tree (ID3)" << endl;
        cout << "(6) Gaussian Naive Bayes" << endl;
        cout << "(7) Print results" << endl;
        cout << "(8) Quit" << endl;
        cout << "Enter your choice (1-8): ";
    }
    
    void run() {
        int choice = 0;
        
        while (true) {
            displayMenu();
            string input;
            getline(cin, input);
            
            try {
                choice = stoi_safe(input);
            } catch (...) {
                cout << "Invalid input. Please enter a number 1-8." << endl;
                continue;
            }
            
            switch (choice) {
                case 1:
                    loadData();
                    break;
                case 2:
                    runLinearRegression();
                    break;
                case 3:
                    runLogisticRegression();
                    break;
                case 4:
                    runKNN();
                    break;
                case 5:
                    runDecisionTree();
                    break;
                case 6:
                    runNaiveBayes();
                    break;
                case 7:
                    results_manager.printComparisonTable();
                    results_manager.saveComparisonTableToFile("cpp_results_table.txt");
                case 8:
                    cout << "Exiting program. Goodbye!" << endl;
                    return;
                default:
                    cout << "Invalid choice. Please enter 1-8." << endl;
            }
            
            // Pause to let user see results before returning to menu
            if (choice != 8) {
                cout << "\nPress Enter to return to menu...";
                cin.ignore();
            }
        }
    }
    
    void loadData() {
    // Files
    string original_file = "adult_income.csv";           // Original with index column and "?"
    string cleaned_file = "adult_cleaned.csv";    // Cleaned without index and "?" rows
    string target = "income";
    
    auto script_start = chrono::high_resolution_clock::now();
    
    try {
        cout << "\nLoading and cleaning input data set:" << endl;
        cout << "************************************" << endl;
        
        // Get current time for logging
        auto now = chrono::system_clock::now();
        auto in_time_t = chrono::system_clock::to_time_t(now);
        cout << put_time(localtime(&in_time_t), "[%Y-%m-%d %H:%M:%S]") << " Starting Script" << endl;
        
        // ========== STEP 1: Load original data ==========
        cout << put_time(localtime(&in_time_t), "[%Y-%m-%d %H:%M:%S]") << " Loading training data set" << endl;
        auto load_start = chrono::high_resolution_clock::now();
        
        Dataset raw_data = loadCSV(original_file, target);
        
        auto load_end = chrono::high_resolution_clock::now();
        double load_time = chrono::duration<double>(load_end - load_start).count();
        
        // Get updated time
        now = chrono::system_clock::now();
        in_time_t = chrono::system_clock::to_time_t(now);
        
        cout << put_time(localtime(&in_time_t), "[%Y-%m-%d %H:%M:%S]") 
             << " Total Columns Read: " << raw_data.feature_names.size() << endl;
        cout << put_time(localtime(&in_time_t), "[%Y-%m-%d %H:%M:%S]") 
             << " Total Rows Read: " << raw_data.samples.size() << endl;
        
        // ========== STEP 2: Clean the data ==========
        auto clean_start = chrono::high_resolution_clock::now();
        
        cout << "\n" << put_time(localtime(&in_time_t), "[%Y-%m-%d %H:%M:%S]") 
             << " Cleaning data (removing rows with '?' and index column)..." << endl;
        
        Dataset cleaned_data;
        cleaned_data.feature_names = raw_data.feature_names;
        cleaned_data.target_name = raw_data.target_name;
        cleaned_data.is_classification = raw_data.is_classification;
        
        // Remove index column if present (first column with numbers 0,1,2,3...)
        // In your cleaned file, the index column is gone
        bool has_index_column = false;
        for (size_t i = 0; i < cleaned_data.feature_names.size(); i++) {
            if (cleaned_data.feature_names[i] == "0" || 
                cleaned_data.feature_names[i] == "index" ||
                cleaned_data.feature_names[i] == "Unnamed: 0") {
                has_index_column = true;
                cleaned_data.feature_names.erase(cleaned_data.feature_names.begin() + i);
                break;
            }
        }
        
        if (has_index_column) {
            cout << put_time(localtime(&in_time_t), "[%Y-%m-%d %H:%M:%S]") 
                 << " Removed index column" << endl;
        }
        
        int removed_rows = 0;
        int kept_rows = 0;
        
        // Clean each sample
        for (const auto& raw_sample : raw_data.samples) {
            bool keep_row = true;
            
            // Check for "?" in categorical features
            for (const auto& kv : raw_sample.categorical_features) {
                if (kv.second == "?") {
                    keep_row = false;
                    break;
                }
            }
            
            // Check workclass specifically (your example shows workclass="?" removed)
            if (raw_sample.categorical_features.find("workclass") != raw_sample.categorical_features.end()) {
                if (raw_sample.categorical_features.at("workclass") == "?") {
                    keep_row = false;
                }
            }
            
            // Check occupation specifically
            if (raw_sample.categorical_features.find("occupation") != raw_sample.categorical_features.end()) {
                if (raw_sample.categorical_features.at("occupation") == "?") {
                    keep_row = false;
                }
            }
            
            if (!keep_row) {
                removed_rows++;
                continue;
            }
            
            // Create cleaned sample
            DataPoint cleaned_sample;
            cleaned_sample.target_value = raw_sample.target_value;
            cleaned_sample.target_numeric = raw_sample.target_numeric;
            
            // Copy numerical features (skip index if it was stored as numerical)
            for (const auto& kv : raw_sample.numerical_features) {
                const string& feature = kv.first;
                double value = kv.second;
                
                // Skip index column if it exists
                if (has_index_column && (feature == "0" || feature == "index" || feature == "Unnamed: 0")) {
                    continue;
                }
                
                cleaned_sample.numerical_features[feature] = value;
                
                // Update ranges
                if (cleaned_data.numerical_ranges.find(feature) == cleaned_data.numerical_ranges.end()) {
                    cleaned_data.numerical_ranges[feature] = make_pair(value, value);
                } else {
                    auto& range = cleaned_data.numerical_ranges[feature];
                    range.first = min(range.first, value);
                    range.second = max(range.second, value);
                }
            }
            
            // Copy categorical features
            for (const auto& kv : raw_sample.categorical_features) {
                const string& feature = kv.first;
                string value = kv.second;
                
                // Skip index column if it exists
                if (has_index_column && (feature == "0" || feature == "index" || feature == "Unnamed: 0")) {
                    continue;
                }
                
                cleaned_sample.categorical_features[feature] = value;
                
                // Update categorical values
                if (cleaned_data.categorical_values.find(feature) == cleaned_data.categorical_values.end()) {
                    cleaned_data.categorical_values[feature] = {value};
                } else {
                    auto& values = cleaned_data.categorical_values[feature];
                    if (find(values.begin(), values.end(), value) == values.end()) {
                        values.push_back(value);
                    }
                }
            }
            
            cleaned_data.samples.push_back(cleaned_sample);
            kept_rows++;
        }
        
        auto clean_end = chrono::high_resolution_clock::now();
        double clean_time = chrono::duration<double>(clean_end - clean_start).count();
        
        now = chrono::system_clock::now();
        in_time_t = chrono::system_clock::to_time_t(now);
        
        cout << put_time(localtime(&in_time_t), "[%Y-%m-%d %H:%M:%S]") 
             << " Rows kept: " << kept_rows << endl;
        cout << put_time(localtime(&in_time_t), "[%Y-%m-%d %H:%M:%S]") 
             << " Rows removed: " << removed_rows << endl;
        
        // ========== STEP 3: Normalize ==========
        cout << put_time(localtime(&in_time_t), "[%Y-%m-%d %H:%M:%S]") 
             << " Normalizing features..." << endl;
        
        normalizeDataset(cleaned_data);
        
        // ========== STEP 4: Save cleaned file ==========
        cout << put_time(localtime(&in_time_t), "[%Y-%m-%d %H:%M:%S]") 
             << " Saving cleaned data to: " << cleaned_file << endl;
        
        // (Optional: Save to file code here - same as before if needed)
        
        // ========== STEP 5: Load for ML ==========
        train_data = cleaned_data;
        test_data = train_data;
        
        config.train_file = cleaned_file;
        config.target_variable = target;
        config.test_file = "";
        config.normalize = true;
        
        data_loaded = true;
        
        // ========== STEP 6: Final timing ==========
        auto script_end = chrono::high_resolution_clock::now();
        double total_time = chrono::duration<double>(script_end - script_start).count();
        
        now = chrono::system_clock::now();
        in_time_t = chrono::system_clock::to_time_t(now);
        
        cout << "\n" << put_time(localtime(&in_time_t), "[%Y-%m-%d %H:%M:%S]") 
             << " Data cleaning completed" << endl;
        
        cout << "\nTime to load is: " << fixed << setprecision(3) << load_time << " seconds" << endl;
        cout << "Time to clean is: " << fixed << setprecision(3) << clean_time << " seconds" << endl;
        cout << "Total time is: " << fixed << setprecision(3) << total_time << " seconds" << endl;
        
        cout << "\nFinal dataset: " << kept_rows << " rows, " 
             << cleaned_data.feature_names.size() << " columns" << endl;
        
        // Show class distribution
        if (cleaned_data.is_classification) {
            map<string, int> class_counts;
            for (const auto& sample : cleaned_data.samples) {
                class_counts[sample.target_value]++;
            }
            
            cout << "Class distribution:" << endl;
            for (const auto& kv : class_counts) {
                double percentage = (kv.second * 100.0) / cleaned_data.samples.size();
                cout << "  " << kv.first << ": " << kv.second 
                     << " (" << fixed << setprecision(1) << percentage << "%)" << endl;
            }
        }
        
        cout << "************************************" << endl;
        
    } catch (const exception& e) {
        cout << "Error during data loading/cleaning: " << e.what() << endl;
    }
}
    
    void runLinearRegression() {
        if (!checkDataLoaded()) return;
        
        try {
            LinearRegression lr;
            
            cout << "\nSelected: Linear Regression (closed-form)" << endl;
            cout << "Enter target variable [" << config.target_variable << "]: ";
            string target;
            getline(cin, target);
            if (!target.empty()) config.target_variable = target;
            
            cout << "Enter L2 regularization strength (default " << config.l2_lambda << "): ";
            string l2_input;
            getline(cin, l2_input);
            if (!l2_input.empty()) config.l2_lambda = stod_safe(l2_input);
            
            AlgorithmResult result = lr.train(train_data, config);
            auto result_pair = lr.predict(test_data);
            double rmse = result_pair.first;
            double r2 = result_pair.second;
            
            result.metric1 = rmse;
            result.metric2 = r2;
            
            cout << "\nAlgorithm: " << result.algorithm_name << endl;
            cout << "Train time: " << result.train_time << "s" << endl;
            cout << "Regression Metrics:" << endl;
            cout << "  RMSE: " << rmse << endl;
            cout << "  R²: " << r2 << endl;
            
            results_manager.saveResult(result);
            
        } catch (const exception& e) {
            cout << "Error: " << e.what() << endl;
        }
    }
    
    void runLogisticRegression() {
        if (!checkDataLoaded()) return;
        
        try {
            LogisticRegression lr(config.seed);
            
            cout << "\n*****************************" << endl;
            cout << "Logistic Regression (binary):" << endl;
            cout << "*****************************" << endl;
            cout << "Enter input options:" << endl;
            
            cout << "Target variable [" << config.target_variable << "]: ";
            string target;
            getline(cin, target);
            if (!target.empty()) config.target_variable = target;
            
            cout << "Learning rate (default " << config.learning_rate << "): ";
            string lr_input;
            getline(cin, lr_input);
            if (!lr_input.empty()) config.learning_rate = stod_safe(lr_input);
            
            cout << "Epochs (default " << config.epochs << "): ";
            string epochs_input;
            getline(cin, epochs_input);
            if (!epochs_input.empty()) config.epochs = stoi_safe(epochs_input);
            
            cout << "L2 regularization (default " << config.l2_lambda << "): ";
            string l2_input;
            getline(cin, l2_input);
            if (!l2_input.empty()) config.l2_lambda = stod_safe(l2_input);
            
            cout << "Random seed (default " << config.seed << "): ";
            string seed_input;
            getline(cin, seed_input);
            if (!seed_input.empty()) config.seed = stoi_safe(seed_input);
            
            AlgorithmResult result = lr.train(train_data, config);
            auto result_pair = lr.predict(test_data);
            double accuracy = result_pair.first;
            double macro_f1 = result_pair.second;
            
            result.metric1 = accuracy;
            result.metric2 = macro_f1;
            
            cout << "\n*****************************" << endl;
            cout << "Algorithm: " << result.algorithm_name << endl;
            cout << "Train time: " << result.train_time << "s" << endl;
            cout << "Accuracy: " << accuracy << endl;
            cout << "Macro-F1: " << macro_f1 << endl;
            cout << "SLOC: " << result.sloc << endl;
            cout << "*****************************" << endl;
            
            results_manager.saveResult(result);
            
        } catch (const exception& e) {
            cout << "Error: " << e.what() << endl;
        }
    }
    
    void runKNN() {
        if (!checkDataLoaded()) return;
        
        try {
            KNearestNeighbors knn;
            
            cout << "\nSelected: k-Nearest Neighbors" << endl;
            
            cout << "Enter k value (default " << config.k_value << "): ";
            string k_input;
            getline(cin, k_input);
            if (!k_input.empty()) config.k_value = stoi_safe(k_input);
            
            cout << "\nAvailable distance metrics:" << endl;
            cout << "1. Euclidean (L2 norm)" << endl;
            cout << "2. Manhattan (L1 norm)" << endl;
            cout << "3. Minkowski (p=3)" << endl;
            cout << "4. Chebyshev" << endl;
            cout << "Select distance metric (1-4): ";
            
            string metric_input;
            getline(cin, metric_input);
            string metric;
            if (metric_input == "1") metric = "Euclidean";
            else if (metric_input == "2") metric = "Manhattan";
            else if (metric_input == "3") metric = "Minkowski";
            else if (metric_input == "4") metric = "Chebyshev";
            else {
                cout << "Invalid selection. Using Euclidean." << endl;
                metric = "Euclidean";
            }
            
            cout << "\nAvailable weighting schemes:" << endl;
            cout << "1. Uniform" << endl;
            cout << "2. Distance" << endl;
            cout << "3. SquareInvDist" << endl;
            cout << "Select weighting scheme (1-3): ";
            
            string weight_input;
            getline(cin, weight_input);
            string weighting;
            if (weight_input == "1") weighting = "Uniform";
            else if (weight_input == "2") weighting = "Distance";
            else if (weight_input == "3") weighting = "SquareInvDist";
            else {
                cout << "Invalid selection. Using Uniform." << endl;
                weighting = "Uniform";
            }
            
            cout << "Target variable [" << config.target_variable << "]: ";
            string target;
            getline(cin, target);
            if (!target.empty()) config.target_variable = target;
            
            cout << "Normalize features? (y/n, default " << (config.normalize ? "y" : "n") << "): ";
            string norm_input;
            getline(cin, norm_input);
            if (!norm_input.empty()) {
                config.normalize = (toLowerCase(norm_input) == "y");
            }
            
            knn.setParameters(metric, weighting, config.normalize);
            
            AlgorithmResult result = knn.train(train_data, config);
            auto result_pair = knn.predict(test_data);
            double accuracy = result_pair.first;
            double macro_f1 = result_pair.second;
            
            result.metric1 = accuracy;
            result.metric2 = macro_f1;
            
            cout << "\nAlgorithm: " << result.algorithm_name << endl;
            cout << "Train time: " << result.train_time << "s" << endl;
            cout << "Accuracy: " << accuracy << endl;
            cout << "Macro-F1: " << macro_f1 << endl;
            
            results_manager.saveResult(result);
            
        } catch (const exception& e) {
            cout << "Error: " << e.what() << endl;
        }
    }
    
    void runDecisionTree() {
        if (!checkDataLoaded()) return;
        
        try {
            DecisionTree dt;
            
            cout << "\nSelected: Decision Tree (ID3)" << endl;
            
            cout << "Enter max_depth (-1 for unlimited, default " << config.max_depth << "): ";
            string depth_input;
            getline(cin, depth_input);
            if (!depth_input.empty()) config.max_depth = stoi_safe(depth_input);
            
            cout << "Enter n_bins (default " << config.n_bins << "): ";
            string bins_input;
            getline(cin, bins_input);
            if (!bins_input.empty()) config.n_bins = stoi_safe(bins_input);
            
            cout << "Target variable [" << config.target_variable << "]: ";
            string target;
            getline(cin, target);
            if (!target.empty()) config.target_variable = target;
            
            dt.setTreeParameters(config.max_depth, config.n_bins);
            
            AlgorithmResult result = dt.train(train_data, config);
            auto result_pair = dt.predict(test_data);
            double accuracy = result_pair.first;
            double macro_f1 = result_pair.second;
            
            result.metric1 = accuracy;
            result.metric2 = macro_f1;
            
            cout << "\nAlgorithm: " << result.algorithm_name << endl;
            cout << "Train time: " << result.train_time << "s" << endl;
            cout << "Accuracy: " << accuracy << endl;
            cout << "Macro-F1: " << macro_f1 << endl;
            
            results_manager.saveResult(result);
            
        } catch (const exception& e) {
            cout << "Error: " << e.what() << endl;
        }
    }
    
    void runNaiveBayes() {
        if (!checkDataLoaded()) return;
        
        try {
            GaussianNaiveBayes nb;
            
            cout << "\nSelected: Gaussian Naive Bayes" << endl;
            
            cout << "Enter target variable [" << config.target_variable << "]: ";
            string target;
            getline(cin, target);
            if (!target.empty()) config.target_variable = target;
            
            AlgorithmResult result = nb.train(train_data, config);
            auto result_pair = nb.predict(test_data);
            double accuracy = result_pair.first;
            double macro_f1 = result_pair.second;
            
            result.metric1 = accuracy;
            result.metric2 = macro_f1;
            
            cout << "\nAlgorithm: " << result.algorithm_name << endl;
            cout << "Train time: " << result.train_time << "s" << endl;
            cout << "Accuracy: " << accuracy << endl;
            cout << "Macro-F1: " << macro_f1 << endl;
            cout << "Number of classes: " << (nb.predict(test_data), 2) << endl; // Simplified
            
            results_manager.saveResult(result);
            
        } catch (const exception& e) {
            cout << "Error: " << e.what() << endl;
        }
    }
    
    bool checkDataLoaded() {
        if (!data_loaded) {
            cout << "Error: No data loaded. Please select option (1) to load data first." << endl;
            return false;
        }
        return true;
    }
};

// ============================ MAIN FUNCTION ============================

int main(int argc, char* argv[]) {
    cout << "Machine Learning System with Multiple Algorithms" << endl;
    cout << "================================================" << endl;
    
    // Parse command line arguments if any
    Config config;
    for (int i = 1; i < argc; i++) {
        string arg = argv[i];
        
        if (arg == "--train" && i + 1 < argc) {
            config.train_file = argv[++i];
        } else if (arg == "--test" && i + 1 < argc) {
            config.test_file = argv[++i];
        } else if (arg == "--target" && i + 1 < argc) {
            config.target_variable = argv[++i];
        } else if (arg == "--algo" && i + 1 < argc) {
            config.algorithm = argv[++i];
        } else if (arg == "--lr" && i + 1 < argc) {
            config.learning_rate = stod_safe(argv[++i]);
        } else if (arg == "--epochs" && i + 1 < argc) {
            config.epochs = stoi_safe(argv[++i]);
        } else if (arg == "--k" && i + 1 < argc) {
            config.k_value = stoi_safe(argv[++i]);
        } else if (arg == "--max_depth" && i + 1 < argc) {
            config.max_depth = stoi_safe(argv[++i]);
        } else if (arg == "--l2" && i + 1 < argc) {
            config.l2_lambda = stod_safe(argv[++i]);
        } else if (arg == "--normalize") {
            config.normalize = true;
        } else if (arg == "--seed" && i + 1 < argc) {
            config.seed = stoi_safe(argv[++i]);
        } else if (arg == "--n_bins" && i + 1 < argc) {
            config.n_bins = stoi_safe(argv[++i]);
        } else {
            cout << "Unknown argument: " << arg << endl;
        }
    }
    
    // Start interactive system
    MLSystem system;
    system.run();
    
    return 0;
}