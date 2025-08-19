# Flask Palindrome Checker App

## Overview
This is a simple Flask app to check if a given number is a valid palindrome via an API endpoint.

## Features
- API endpoint: `/isValidPalindrome?number=<number>`
- Returns `{ "message": "valid" }` if the number is a palindrome, otherwise `{ "message": "invalid" }`.
- Proper folder structure for scaling and maintainability.
- Includes requirements.txt for easy setup.

## Setup

1. **Clone the repo**  
   ```bash
   git clone https://github.com/hrudaym6-hue/copilot-tryout.git
   cd copilot-tryout/palindrome_app
   ```

2. **Install dependencies**  
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the app**  
   ```bash
   python app.py
   ```

## Usage

Send a GET request to:
```
http://localhost:5000/isValidPalindrome?number=121
```
Response:
```json
{ "message": "valid" }
```

## Folder Structure

```
palindrome_app/
├── __init__.py
├── app.py
requirements.txt
README.md
```

## License
MIT
