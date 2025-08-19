from flask import Flask, request, jsonify

app = Flask(__name__)

def is_palindrome(number: str) -> bool:
    return number == number[::-1]

@app.route('/isValidPalindrome', methods=['GET'])
def check_palindrome():
    number = request.args.get('number', '')
    if not number.isdigit():
        return jsonify({"message": "invalid"}), 400
    if is_palindrome(number):
        return jsonify({"message": "valid"})
    else:
        return jsonify({"message": "invalid"})

if __name__ == '__main__':
    app.run(debug=True)