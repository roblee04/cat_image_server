from flask import Flask, render_template
import subprocess


# App Creation
app = Flask(__name__)

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/search/<query>', methods=['GET'])
def search(query: str):

    # QUERY
    # escape character for each search term
    
    # Popen() is async, run is synchronous
    rclip = subprocess.run(["rclip", "-fn", "-t", "5", query], cwd="static/img", encoding='utf-8', stdout=subprocess.PIPE)
    results = rclip.stdout.split('\n')

    for i in range(len(results)):
        results[i] = results[i][32:]
    
    return render_template('results.html', imgs=results)

# Start the Server

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8002, debug=True, use_reloader=False)