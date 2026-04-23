#!/usr/bin/env python
import sys
import py_compile
import os

# Test 1: Import gyroid_utils
sys.path.insert(0, 'src')
try:
    import gyroid_utils
    print("✓ Test 1 PASS: gyroid_utils import successful")
except Exception as e:
    print(f"✗ Test 1 FAIL: {type(e).__name__}: {e}")
    sys.exit(1)

# Test 2: Compile all Python files in src
print("\nTest 2: Compiling all Python files in src/")
try:
    compile_errors = []
    for root, dirs, files in os.walk('src'):
        for file in files:
            if file.endswith('.py'):
                filepath = os.path.join(root, file)
                try:
                    py_compile.compile(filepath, doraise=True)
                except py_compile.PyCompileError as e:
                    compile_errors.append((filepath, str(e)))
    
    if compile_errors:
        print(f"✗ Test 2 FAIL: {len(compile_errors)} file(s) failed to compile:")
        for filepath, error in compile_errors:
            print(f"  {filepath}: {error}")
        sys.exit(1)
    else:
        print(f"✓ Test 2 PASS: All Python files compiled successfully")
except Exception as e:
    print(f"✗ Test 2 FAIL: {type(e).__name__}: {e}")
    sys.exit(1)

print("\n✓ All sanity checks passed")
