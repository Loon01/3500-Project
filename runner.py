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
            compile_result = subprocess.run(["g++", "-std=c++11", "alg.cpp", "-o", "alg_executable"], capture_output=True, text=True)
    
            if compile_result.returncode == 0:
                # Run
                subprocess.run(["./alg_executable"])
           else:
                print("Compilation failed:")
                print(compile_result.stderr)
            
        if choice == "2": 
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
                ["javac", "-encoding", "UTF-8", "-d", "bin"] + java_files1, 
                capture_output=True,
                text=True,
                )

            subprocess.run(
                ["javac", "-encoding", "UTF-8", "-d", "bin"] + java_files2, 
                capture_output=True,
                text=True,
                )

            result = subprocess.run(
                ["java", "-cp", "bin", "Main"],
                stdin=None,     # allow keyboard input
                stdout=None,    # show Java output directly in terminal
                stderr=None 
                )

        if choice == "3": 
            # Invoking the Lisp interpreter
            result = subprocess.run(
                ["sbcl", "--load", "functional.lisp"],
                stdin=None,     # allow keyboard input
                stdout=None,    # show Lisp output directly in terminal
                stderr=None
                )

        if choice == "4": 
            print("No General results yet")
        
        elif choice == "5":
            print("Exiting.")
            break

        else:
            print("Invalid option.")
        
        #print("finished")
    
if __name__ == "__main__":
    main()
