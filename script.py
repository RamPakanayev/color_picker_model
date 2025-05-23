"""
Run test script that executes test_polygons.py on all JSON test files.
This script automatically runs the color prediction on all test files
and saves the output to the results directory.
"""

import subprocess
import sys
import os
import time
import glob

# Get the current directory
current_dir = os.path.dirname(os.path.abspath(__file__))

# Path to the shapes_jsons directory
shapes_dir = os.path.join(current_dir, "shapes_jsons")

# Find all JSON files in the directory
json_files = glob.glob(os.path.join(shapes_dir, "*.json"))

if not json_files:
    print("Error: No JSON files found in the shapes_jsons directory!")
    sys.exit(1)
else:
    print(f"Found {len(json_files)} JSON files to test.")

# Create results directory if it doesn't exist
results_dir = os.path.join(current_dir, "results")
os.makedirs(results_dir, exist_ok=True)

# Generate commands
commands = []
for json_file in json_files:
    base_name = os.path.basename(json_file).replace(".json", "")
    output_file = os.path.join(results_dir, f"output_{base_name}.png")
    commands.append([sys.executable, os.path.join(current_dir, "test_polygons.py"), json_file, output_file])

# Execute each command one after the other
print(f"Running tests on {len(json_files)} JSON files...")
start_time = time.time()

success_count = 0
failed_commands = []

for cmd in commands:
    print(f"\nRunning: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    # Check if the command executed successfully
    if result.returncode == 0:
        print(f"Command succeeded: {' '.join(cmd)}\n")
        success_count += 1
    else:
        print(f"Error running command: {' '.join(cmd)}")
        print("Stdout:", result.stdout)
        print("Stderr:", result.stderr)
        failed_commands.append(cmd)
        # Continue with other tests even if one fails

elapsed_time = time.time() - start_time
print(f"\nTest run completed in {elapsed_time:.2f} seconds")
print(f"Results: {success_count}/{len(commands)} tests passed")

if failed_commands:
    print("\nFailed commands:")
    for cmd in failed_commands:
        print(f"  - {' '.join(cmd)}")
else:
    print("\nAll commands executed successfully!")