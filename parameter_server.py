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

import ast
import json
import torch
import flask
from flask import request
import sys
import model.model
import numpy as np
import threading
from google.cloud import storage

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
            #print(f"p {p}")
            #print(f"updates[i]: {updates[i]}")
            new_v = np.array(p.tolist()) + np.array(updates[i])
            p.copy_(torch.from_numpy(new_v))
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
TODO: print a validation accuracy somewhere after x updates.
"""
class ParameterServer():
    def __init__(self):
        self.k = 20 #update every 20 minibatches.
        
        # stores pending gradients count
        self.pending = 0

        # synchronization locks (mostly for self.model and pending batches)
        self.model_lock = threading.Lock()

        # storage bucket
        self.storage_bucket = "project-distml-ayb2121-yz4053.appspot.com"
        self.storage_client = storage.Client.from_service_account_json('distributed/service-account.json')
        self.bucket = self.storage_client.get_bucket(self.storage_bucket)
        self.model_url = None

        # get and save version 0 of the model
        self.model = get_model()
        self.update = -1
        self.write_model()

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
        #payload["model"] = get_model_file_name(self.update)
        payload["model"] = self.model_url
        return json.dumps(payload)

    def write_model(self):
        # TODO - integrate with storage
        # currently, temporarily a local save:
        # IF deploying with app engine, this should technically be mounted.
        model_name = get_model_file_name(self.update)
        self.update += 1
        torch.save(self.model.state_dict(), model_name)

        # Now that we've saved this, upload to storage.
        new_blob = self.bucket.blob(model_name)
        new_blob.upload_from_filename(filename=model_name)
        new_blob.make_public() #Allow public access. This is not secure, but for ease of demonstration.
        self.model_url = new_blob.public_url

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

    # this is just to return a 200
    to_ret = {"status": True}
    return json.dumps(to_ret)


if __name__ == "__main__":
    # Used when running locally only. When deploying to Google App
    # Engine, a webserver process such as Gunicorn will serve the app. This
    # can be configured by adding an `entrypoint` to app.yaml.
    app.run(host="localhost", port=8080, debug=True)
    #app.run()
    

