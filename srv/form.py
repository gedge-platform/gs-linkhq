from flask import Flask,render_template,request
import requests
import json
import os

SERVER_IP = os.environ.get('SERVER_IP')
SERVER_PORT = os.environ.get('SERVER_PORT')

AGENT_ADDRESS = os.environ.get('AGENT_ADDRESS')
AGENT_PORT = os.environ.get('AGENT_PORT')

AUTH_ID = os.environ.get('AUTH_ID')
AUTH_PASS = os.environ.get('AUTH_PASS')

app = Flask(__name__)

if '://' in AGENT_ADDRESS:
    agent_url = AGENT_ADDRESS
else:
    agent_url = 'http://' + AGENT_ADDRESS
if AGENT_PORT != '80':
    agent_url = agent_url + ':' + AGENT_PORT

def request_assigned_cluster(task_id):
    pass

def request_assign_task(task):
    endpoint = agent_url + '/task'
    data = {'task': {'req_edge': task.req_edge,
                     'resources': {'cpu': task.resources['cpu'],
                                   'memory': task.resources['memory'],
                                   'gpu': task.resources['gpu']
                                   },
                     'deadline': task.deadline}}
    data = json.dumps(data)

    try:
        res = requests.post(endpoint, data)
        if res.status_code == 201:
            data = res.json()
            task_id = data['task_id']
        
        elif res.status_code == 503:
            data = res.json()
    
    except requests.exceptions.RequestException:
        pass


@app.route('/')
def form():
    return render_template('form.html')
 
@app.route('/data/', methods = ['POST', 'GET'])
def data():
    if request.method == 'GET':
        return f"The URL /data is accessed directly. Try going to '/form' to submit form"
    if request.method == 'POST':
        form_data = request.form
        requests.post("http://agent/")
        return render_template('data.html', form_data = form_data)
 
 
app.run(host='0.0.0.0', port=5000, debug=True)