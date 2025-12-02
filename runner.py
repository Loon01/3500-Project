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
            print("No C++ file yet")
            
        if choice == "2": 
            # Make sure build directory exists
            os.makedirs("build", exist_ok=True)

            # Find all Java files
            java_files = glob.glob("test/test/oop-java/ml/*.java")
            #print("Java files found: ", java_files)

            # Error handling
            if not java_files:
                raise RuntimeError("No Java files found — check your path!")

            subprocess.run(
                ["javac", "-encoding", "UTF-8", "-d", "build"] + java_files, 
                capture_output=True,
                text=True,
                )

            result = subprocess.run(
                ["java", "-cp", "build", "Main"],
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
        
        print("finished")
    
if __name__ == "__main__":
    main()
