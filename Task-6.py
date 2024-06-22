#!/usr/bin/env python
# coding: utf-8

# In[1]:


import ast
import subprocess

def analyze_code_with_flake8(file_path):
    """Analyze code using flake8."""
    result = subprocess.run(['flake8', file_path], capture_output=True, text=True)
    return result.stdout

def analyze_code_with_ast(file_path):
    """Analyze code using the AST module."""
    with open(file_path, 'r') as file:
        tree = ast.parse(file.read(), filename=file_path)
    
    feedback = []

    # Check for function length
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            num_lines = len(node.body)
            if num_lines > 50:
                feedback.append(f"Function '{node.name}' is too long ({num_lines} lines). Consider refactoring.")

    # Check for variable names
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
            if len(node.id) == 1:
                feedback.append(f"Variable '{node.id}' has a too short name. Consider using more descriptive names.")
    
    return feedback

def review_code(file_path):
    """Review code for quality and best practices."""
    flake8_feedback = analyze_code_with_flake8(file_path)
    ast_feedback = analyze_code_with_ast(file_path)
    
    combined_feedback = {
        'flake8': flake8_feedback.split('\n'),
        'ast': ast_feedback
    }
    
    return combined_feedback

if __name__ == "__main__":
    file_path = "C:\Siddhant\CodeAlpha Python 6 Months\Task 1\Task-1.py"
    feedback = review_code(file_path)
    
    print("Flake8 Feedback:")
    for line in feedback['flake8']:
        if line:
            print(line)

    print("\nAST Feedback:")
    for line in feedback['ast']:
        print(line)


# In[ ]:




