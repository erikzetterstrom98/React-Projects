from flask import Flask
from flask_cors import CORS

app = Flask(__name__)

cors = CORS(
    app = app,
    origins = 'http://localhost:*'
)

@app.route('/')
def test_function():
    print('wooooo')
    return 'kurva'

def main():
    app.run(debug = True)

if __name__ == '__main__':
    main()