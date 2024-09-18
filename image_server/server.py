from flask import Flask, render_template, jsonify, request, redirect
from flask_cors import CORS, cross_origin
import subprocess
import requests
from PIL import Image
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
    # INPUT: /search/potato+dog-red
    # OUTPUT: ['potato', '+', 'dog', '-', 'red']
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

    # initialize and search model
    img_model = model.Model()
    sim_res = img_model.query(queries)
    # have idx, convert to urls

    with open('image_urls.txt', 'r') as f:
        image_urls = [line.strip() for line in f.readlines()]
    
    results = []
    for sim, idx in sim_res:
        results.append((image_urls[idx], int(sim*1000)))

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
    
    return render_template('results.html', imgs=results)

@app.route('/upload/', methods=['GET', 'POST'])
def upload_files():
    if request.method == 'POST':
        urls = request.form['urls']
        urls_list = urls.splitlines()

        # Process urls
        image_database = []
        with open('image_urls.txt', 'a') as f:
            for url in urls_list:
                url = url.strip()
                # turn url into Pillow object
                image_database.append(Image.open(requests.get(url, stream=True).raw))
                # append to image_urls.txt
                f.write('\n' + url)
        
        print(image_database)
            
        img_model = model.Model()
        img_model.add_to_db(image_database)

        
        return redirect(request.url) 

    return render_template('upload.html')

# Start the Server

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8002, debug=True, use_reloader=False)