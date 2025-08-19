def is_palindrome(number: int) -> bool:
    """
    Check if a given integer is a palindrome.

    Args:
        number (int): The integer to check.

    Returns:
        bool: True if number is palindrome, False otherwise.
    """
    num_str = str(number)
    return num_str == num_str[::-1]

if __name__ == "__main__":
    num = int(input("Enter an integer: "))
    if is_palindrome(num):
        print(f"{num} is a palindrome.")
    else:
        print(f"{num} is not a palindrome.")