from flask import Flask
app = Flask(__name__) # create a Flask app

@app.route("/")
def hello():
    return "Hello World!"
    
if __name__=='__main__':
    app.run(port=3000, debug=True)