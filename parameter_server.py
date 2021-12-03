"""
Parameter server will function as the orchestrator as well. This way we can track time better. 
Order of operations is as follows:

1. Initialize model to random weights (or any kind of initialization like glorot). Set model update number to 0.
2. Write .pth to storage 
3. Wait for POST and GET requests

The parameter server contains 2 apis
GET: No payload, returns the current .pth URL and update number

POST: Payload contains parameters from a worker. 
- Needs to be aggregated until K batches. We can vary K, but for now it's something like 10-20. 

Parameter server will integrate with a google cloud store to write models. 

IMPORTANT: We only expect 1 running instance of this app. More instances will cause error behaviour. 
We are storing a global "state" in the context of this server. This will not work with multiple servers. 
"""

import json
import torch
import flask
from flask import request
import sys
import model.model
import threading

# use the ../model/model.py
def get_model():
    return model.model.CNN()

def aggregate_updates(model, updates):
    """
    Propagates gradient deltas in "updates" into the current model
    """
    with torch.no_grad():
        i = 0
        for p in model.parameters():
            new_v = p.grad + updates[i]
            p.copy(new_v)
            i += 1


def get_model_file_name(iteration):
    return f'model/model{iteration}.pth' #TODO update this to storage url

def parse_gradient_payload(contents):
    """
    expecting a json with
    {
    "parameters": []
    }
    """
    contents_json = json.loads(contents)
    return contents_json["parameters"]

"""
ParameterServer will handle gradient updates and sychronization, and batching.
"""
class ParameterServer():
    def __init__(self):
        self.model = get_model()
        self.update = 0
        self.k = 20 #update every 20 minibatches.
        
        # stores pending gradients count
        self.pending = 0

        # synchronization locks (mostly for self.model and pending batches)
        self.model_lock = threading.Lock()

    def add_gradient(self, gradients):
        """
        Adds the gradient to the pending list. 
        If pending list exceeds k, we aggregate and clear the list
        """

        # lock model 
        self.model_lock.acquire()

        # apply gradient
        aggregate_updates(self.model, gradients)

        # check 
        self.pending += 1

        if self.pending >= self.k:
            # write new model and update
            self.write_model()
            self.pending = 0 # reset

        #unlock model
        self.model_lock.release()
    
    def get_latest_model_payload(self):
        """
        Returns the latest model url
        {
            "update": 0,
            "model": "<url to pth file>"
        }
        """
        payload = {}
        payload["update"] = self.update
        payload["model"] = get_model_file_name(self.update)
        return json.dumps(payload)

    def write_model(self):
        # TODO - integrate with storage
        # currently, temporarily a local save:
        torch.save(self.model.state_dict(), get_model_file_name(self.update))
        self.update += 1

# Initialize p.s. holder. 
print("Starting parameter server and saving initial model.")    
ps = ParameterServer()

# If `entrypoint` is not defined in app.yaml, App Engine will look for an app
# called `app` in `main.py`.
app = flask.Flask(__name__)

@app.route("/", methods=["GET"])
def get_model():
    latest_model_payload = ps.get_latest_model_payload()
    return latest_model_payload

@app.route("/", methods=["POST"])
def update_model():
    # parse the payload
    gradients = parse_gradient_payload(request.get_json(force=True))

    # compute the update
    ps.add_gradient(gradients)


if __name__ == "__main__":
    # Used when running locally only. When deploying to Google App
    # Engine, a webserver process such as Gunicorn will serve the app. This
    # can be configured by adding an `entrypoint` to app.yaml.
    app.run(host="localhost", port=8080, debug=True)
    #app.run()
    

