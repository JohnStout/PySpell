# Thanks bing copilot :)

from collections import Counter


def find_duplicate_characters(input_string):
    # Create a dictionary to store character frequencies
    char_count = {}
    
    # Iterate through the input string
    for char in input_string:
        # Increment the count for each character
        char_count[char] = char_count.get(char, 0) + 1
    
    # Initialize a list to store duplicate characters
    duplicates = []
    
    # Iterate through the dictionary
    for char, count in char_count.items():
        # If count is greater than 1, add the character to the duplicates list
        if count > 1:
            duplicates.append(char)
    
    return duplicates

def find_dup_char(input_string):
    char_counts = Counter(input_string)
    duplicates = [char for char, count in char_counts.items() if count > 1]
    return duplicates