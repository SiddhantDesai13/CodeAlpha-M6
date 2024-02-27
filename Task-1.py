#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Define a generator function to generate Fibonacci numbers
def fibonacci_generator():
    
    # Initialize variables a and b for Fibonacci sequence
    a, b = 0, 1
    
    # Infinite loop to yield Fibonacci numbers
    while True:
        yield a
        a, b = b, a + b

# Main function to interact with the user and generate Fibonacci numbers
def main():
    
    # Display a welcome message
    print("Welcome to the Fibonacci number generator!")
    
    try:
        # Ask the user for the number of Fibonacci numbers to generate
        number = int(input("How many Fibonacci numbers do you want to generate? "))
        
        # Check if the input is a positive integer
        if number <= 0:
            print("Please enter a positive integer:")
            return
    except ValueError:
        print("Invalid input. Please enter a valid integer:")
        return
    
    # Create a generator object
    generator = fibonacci_generator()
    
    # Generate Fibonacci numbers based on user input
    fibonacci_numbers = []
    for _ in range(number):
        fibonacci_numbers.append(next(generator))
    
    # Display the generated Fibonacci numbers
    print("Generated Fibonacci numbers:")
    print(fibonacci_numbers)

# Execute the main function if the script is run directly
if __name__ == "__main__":
    main()


# In[ ]:




