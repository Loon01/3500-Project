################################################################################
# PROJECT: AI Library Design Comparison (Programming Languages)
# GROUP MEMBERS: Adrian R, Solomon A., Geneva R., Hermit S.
# ORGN: CSUB - CMPS 3500
# FILE: runner.py
# DATE: 12/05/2025
# COMPILE: python3 runner.py
# DESCRIPTION:
#   A python script that contains a menu that combines all files (C/C++, Java, 
#   and Lisp) that contain the algorithms required to be ran in there respective 
#   language
################################################################################
import os
import subprocess
import glob
import re
from datetime import datetime

def parse_results_table(filepath):
    """Parse the results table file into structured data"""
    results = []
    
    try:
        with open(filepath, 'r') as f:
            lines = f.readlines()
        
        # Skip header lines (first 3 lines)
        if len(lines) < 4:
            return results
            
        # Parse each result line
        for line in lines[3:]:  # Skip header rows
            line = line.strip()
            if not line:
                continue
                
            # Parse using the fixed column widths
            # Format: Impl (25) + Algorithm (20) + TrainTime (12) + TestMetric1 (15) + TestMetric2 (15) + SLOC (8)
            impl = line[:25].strip()
            algorithm = line[25:45].strip()
            train_time = line[45:57].strip().rstrip('s')  # Remove trailing 's'
            test_metric1 = line[57:72].strip()
            test_metric2 = line[72:87].strip()
            sloc = line[87:95].strip()
            
            if impl and algorithm:  # Valid row
                results.append({
                    'implementation': impl,
                    'algorithm': algorithm,
                    'train_time': float(train_time) if train_time.replace('.', '', 1).isdigit() else 0.0,
                    'test_metric1': float(test_metric1) if test_metric1.replace('.', '', 1).isdigit() else 0.0,
                    'test_metric2': float(test_metric2) if test_metric2.replace('.', '', 1).isdigit() else 0.0,
                    'sloc': int(sloc) if sloc.isdigit() else 0
                })
                
    except Exception as e:
        print(f"Error parsing {filepath}: {e}")
        
    return results

def compare_results(cpp_results, java_results, lisp_results):
    """Compare results from all languages"""
    print("\n" + "="*80)
    print("               LANGUAGE COMPARISON RESULTS")
    print("="*80)
    
    # Create a dictionary to organize by algorithm
    all_results = {}
    
    # Add C++ results
    if cpp_results:
        for result in cpp_results:
            algo = result['algorithm']
            if algo not in all_results:
                all_results[algo] = {'C++': None, 'Java': None, 'Lisp': None}
            all_results[algo]['C++'] = result
    
    # Add Java results
    if java_results:
        for result in java_results:
            algo = result['algorithm']
            if algo not in all_results:
                all_results[algo] = {'C++': None, 'Java': None, 'Lisp': None}
            all_results[algo]['Java'] = result
    
    # Add Lisp results
    if lisp_results:
        for result in lisp_results:
            algo = result['algorithm']
            if algo not in all_results:
                all_results[algo] = {'C++': None, 'Java': None, 'Lisp': None}
            all_results[algo]['Lisp'] = result
    
    # Print comparison table
    print("\n" + "-"*100)
    print(f"{'Algorithm':<25} {'Language':<10} {'TrainTime(s)':<12} {'Accuracy/RMSE':<15} {'F1/R²':<15} {'SLOC':<8}")
    print("-"*100)
    
    for algo, lang_results in all_results.items():
        print(f"{algo:<25}")
        for lang in ['C++', 'Java', 'Lisp']:
            if lang_results[lang]:
                result = lang_results[lang]
                # Determine metric names
                if 'Regression' in algo and 'Logistic' not in algo:
                    metric1_name = "RMSE"
                    metric2_name = "R²"
                else:
                    metric1_name = "Accuracy"
                    metric2_name = "Macro-F1"
                
                print(f"  {lang:<10} {result['train_time']:<12.3f} "
                      f"{result['test_metric1']:<15.4f} "
                      f"{result['test_metric2']:<15.4f} "
                      f"{result['sloc']:<8}")
            else:
                print(f"  {lang:<10} {'-':<12} {'-':<15} {'-':<15} {'-':<8}")
        print()
    
    # Calculate and display best performers
    print("\n" + "="*80)
    print("               PERFORMANCE SUMMARY")
    print("="*80)
    
    # Find best accuracy by language
    for lang_name, lang_results in [('C++', cpp_results), ('Java', java_results), ('Lisp', lisp_results)]:
        if lang_results:
            best_acc = max((r for r in lang_results if 'Accuracy' in r.get('algorithm', '')), 
                          key=lambda x: x['test_metric1'], default=None)
            if best_acc:
                print(f"{lang_name} Best Accuracy: {best_acc['test_metric1']:.4f} ({best_acc['algorithm']})")
    
    # Find fastest algorithm by language
    for lang_name, lang_results in [('C++', cpp_results), ('Java', java_results), ('Lisp', lisp_results)]:
        if lang_results:
            fastest = min(lang_results, key=lambda x: x['train_time'], default=None)
            if fastest:
                print(f"{lang_name} Fastest: {fastest['train_time']:.3f}s ({fastest['algorithm']})")
    
    # Save comparison to file
    save_comparison_csv(all_results)

def save_comparison_csv(all_results):
    """Save comparison results to CSV file"""
    import csv
    
    with open('language_comparison.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Algorithm', 'Language', 'TrainTime(s)', 'Accuracy/RMSE', 'F1/R²', 'SLOC'])
        
        for algo, lang_results in all_results.items():
            for lang in ['C++', 'Java', 'Lisp']:
                if lang_results[lang]:
                    result = lang_results[lang]
                    writer.writerow([
                        algo,
                        lang,
                        f"{result['train_time']:.3f}",
                        f"{result['test_metric1']:.4f}",
                        f"{result['test_metric2']:.4f}",
                        result['sloc']
                    ])
    
    print(f"\nComparison saved to: language_comparison.csv")

def print_header ():
    print("""
******************************************************
Welcome to the AI/ML Library Implementation Comparison
******************************************************
Please select an implementation to run:
(1) Procedural (C/C++)
(2) Object-Oriented (Java)
(3) Functional (Lisp)
(4) Print General Results
(5) Quit
""")

def main():
    while True:
        print_header()
        choice = input("Enter choice: ").strip()
        
        if choice == "1": 
            print("Compiling and running C++ implementation...")
    
            # Compile
            compile_result = subprocess.run(["g++", "-std=c++14", "alg2.cpp", "-o", "alg_executable"], capture_output=True, text=True)
    
            if compile_result.returncode == 0:
                # Run
                subprocess.run(["./alg_executable"])
            else:
                print("Compilation failed:")
                print(compile_result.stderr)
            
        elif choice == "2": 
            # Make sure build directory exists
            os.makedirs("src/bin", exist_ok=True)

            # Find all Java files
            java_files1 = glob.glob("src/**/*.java")
            #print("Java files found: ", java_files1)

            java_files2 = glob.glob("src/Main.java")
            #print("Java files found: ", java_files2)

            # Error handling
            if not java_files1:
                raise RuntimeError("No Java files1 found — check your path!")

            if not java_files2:
                raise RuntimeError("No Java files2 found — check your path!")

            subprocess.run(
                ["javac", "-encoding", "UTF-8", "-d", "src/bin"] + java_files1 + java_files2, 
                capture_output=True,
                text=True,
                )

            result = subprocess.run(
                ["java", "-cp", "src/bin", "Main"],
                stdin=None,     # allow keyboard input
                stdout=None,    # show Java output directly in terminal
                stderr=None 
                )

        elif choice == "3": 
            # Invoking the Lisp interpreter
            result = subprocess.run(
                ["sbcl", "--load", "functional.lisp"],
                stdin=None,     # allow keyboard input
                stdout=None,    # show Lisp output directly in terminal
                stderr=None
                )

        elif choice == "4": 
            #subprocess.run(
            #    ["cat", "cpp_results_table.txt"]
            #)
            print("No General results yet")
        
        elif choice == "5":
            print("Exiting.")
            break

        else:
            print("Invalid option.")
        
        #print("finished")
    
if __name__ == "__main__":
    main()
