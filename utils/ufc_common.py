"""
Common UFC utilities - EXACT SAME structure as tennis common.py
Contains helper functions used across UFC ELO system
"""

def mean(arr):
    """
    Calculate mean of array - EXACT SAME as tennis mean() function
    
    Args:
        arr: Array of numeric values
        
    Returns:
        float: Mean value, or 0.5 if array is empty
    """
    if len(arr) == 0:
        return 0.5
    else:
        total = 0
        for val in arr:
            total += val
        return total/(len(arr))


def getWinnerLoserIDS(f1_id, f2_id, result):
    """
    Get winner and loser IDs based on fight result - EXACT SAME as tennis getWinnerLoserIDS()
    
    Args:
        f1_id: Fighter 1 ID (Red fighter)
        f2_id: Fighter 2 ID (Blue fighter)  
        result: Fight result (1 if fighter 1 wins, 0 if fighter 2 wins)
        
    Returns:
        tuple: (winner_id, loser_id)
    """
    if result == 1 or result == "1":
        return f1_id, f2_id
    return f2_id, f1_id


if __name__ == '__main__':
    # Test the functions
    print("UFC Common Utils created successfully!")
    
    # Test mean function
    test_arr1 = [1, 2, 3, 4, 5]
    test_arr2 = []
    print(f"Mean of {test_arr1}: {mean(test_arr1)}")  # Should be 3.0
    print(f"Mean of empty array: {mean(test_arr2)}")  # Should be 0.5
    
    # Test getWinnerLoserIDS function
    fighter1_id = 1001
    fighter2_id = 1002
    
    result1 = 1  # Fighter 1 wins
    result2 = 0  # Fighter 2 wins
    
    winner1, loser1 = getWinnerLoserIDS(fighter1_id, fighter2_id, result1)
    winner2, loser2 = getWinnerLoserIDS(fighter1_id, fighter2_id, result2)
    
    print(f"Result 1 - Winner: {winner1}, Loser: {loser1}")  # Should be 1001, 1002
    print(f"Result 2 - Winner: {winner2}, Loser: {loser2}")  # Should be 1002, 1001
    
    print("✓ All functions working correctly!")
