from flask import Flask, render_template, jsonify
from flask_cors import CORS, cross_origin
import subprocess

import model


# App Creation
app = Flask(__name__)
app.config['CORS_HEADERS'] = 'Content-Type'
cors = CORS(app, resources={r"/static/*": {"origins": "*", "allow_headers": "*", "expose_headers": "*"}})


@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/search/<query>', methods=['GET'])
@cross_origin(allow_headers=['Content-Type', 'ngrok-skip-browser-warning'])
def search(query: str):

    # QUERY
    # escape character for each search term
    # tilde is the escape character
    # parse search
    queries = []

    s = ""
    for i in range(len(query)):
        cur_char = query[i]

        if i == len(query) - 1:
            s += cur_char
            queries.append(s)
        
        if s != "" and (cur_char == "+" or cur_char == "-"):
            queries.append(s)
            s = ""
            if cur_char == "+":
                queries.append("+")
            else:
                queries.append("-")
            continue
        
        s += cur_char

    print(queries)

    img_model = model.Model()
    sim_res = img_model.query(queries[0])
    # have idx, convert to urls

    with open('image_urls.txt', 'r') as f:
        image_urls = [line.strip() for line in f.readlines()]
    
    results = []
    for sim, idx in sim_res:
        results.append((image_urls[idx], int(sim)))

    # rclip_proc = ["rclip", "-fn", "-t", "10"]
    # rclip_proc.extend(queries)
    
    # Popen() is async, run is synchronous
    # rclip = subprocess.run(rclip_proc, cwd="static/img", encoding='utf-8', stdout=subprocess.PIPE)
    # results = rclip.stdout.split('\n')

    # d_path = ""
    # for directory in results[0].split('/'):
    #     if directory == "static":
    #         break

    #     d_path += "/" + directory

    # for i in range(len(results)):
    #     results[i] = results[i][len(d_path):]

    response = jsonify({"query":query, "result":results})
    # response.headers.add('Access-Control-Allow-Origin', '*')
    

    # return response
    return render_template('results.html', imgs=results)

# Start the Server

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8002, debug=True, use_reloader=False)