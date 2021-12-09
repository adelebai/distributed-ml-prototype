"""
Learners are responsible for consuming minibatches, and computing gradients. 
Before learners take a minibatch, they must get a valid model from the P.S

P.S can return the following signals
- not started : i.e. wait and do not consume batches
- parameters returned: learner needs to use this set of parameters (which may be the same as current)

Possible payload received by P.S
{
    "update": 0,
    "model": "<url to pth file>"
}

Learners will download and load the model if the update number is different. 

Payload to send the P.S - just a list of model deltas
{
    "parameters": []
}
"""

from io import StringIO
import json
import numpy as np
import pandas as pd
import requests
from requests.models import Response
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
import sys
import time
from google.auth import jwt
from google.cloud import pubsub_v1
import model.model
import model.gen_dataset
import urllib.request

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

LEARNING_RATE = 0.001

class Learner():
    def __init__(self, ps_url):
        # The data we use in this experiment is small and can fit into memory.
        # In practice there's probably another live feed source.
        self.project_id = "project-distml-ayb2121-yz4053"
        self.topic_id = "data"
        self.sub_id = "data-sub" #subscriber name

        self.set_credentials()
        self.publisher = pubsub_v1.PublisherClient(credentials=self.credentials)
        self.topic_path = self.publisher.topic_path(self.project_id, self.topic_id)

        self.subscriber = pubsub_v1.SubscriberClient(credentials=self.credentials)
        self.sub_path = self.subscriber.subscription_path(self.project_id, self.sub_id)

        # param server url
        self.ps_url = ps_url

        # initial model stats
        self.model = model.model.CNN()
        self.model_url = None
        self.model_path = None
        self.model_update = -1

    # set up credentials
    def set_credentials(self):
        service_account_info = json.load(open("distributed/service-account.json"))
        audience = "https://pubsub.googleapis.com/google.pubsub.v1.Subscriber"
        self.credentials = jwt.Credentials.from_service_account_info(
            service_account_info, audience=audience
        )

    def read_model(self):
        """
        Update the current cached model with a more recent model
        TODO - if url, need to download the file first.
        """
        # download file first
        urllib.request.urlretrieve(self.model_url, self.model_path) # the urls should be sent publicly

        self.model.load_state_dict(torch.load(self.model_path))
        self.model.train()

    def update_model(self):
        """updates the model"""
        response = requests.get(self.ps_url)
        if response.status_code != 200:
            return False
        
        # we only update the model if the version has changed.
        response_json = response.json()
        update = int(response_json["update"])
        if self.model_update == -1 or self.model_update != update:
            self.model_update = update
            self.model_url = response_json["model"]
            self.model_path = f"learner_model{update}"
            self.read_model()

        return True

    def prep_data(self, data_str):
        num_rows = len([x for x in data_str.split('\n')])
        data = pd.DataFrame([x.split(',') for x in data_str.split('\n')])
        labels = data[data.columns[-1]].astype(int) # last column is label
        images = data.drop(columns=data.columns[-1])

        labels_tensor = torch.from_numpy(np.array(labels)).type(torch.LongTensor)
        train_data = model.gen_dataset.my_dataset(df=images, labels=labels_tensor, transform=transform)
        return train_data, num_rows


    def process_message(self, contents):
        """
        TODO - double check if this method is correct
        Effectively what we need to do here is read the minibatch, compute loss and gradients.
        Then finally take the gradient diffs and store in a list (return)
        """
        # convert bytestring to string (a csv minibatch)
        contents_str = contents.decode("utf-8")
        train_data, num_rows = self.prep_data(contents_str)
        
        prior_params = np.array([np.array(p.detach().numpy()) for p in self.model.parameters()]) # store prior params
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=LEARNING_RATE, betas=(0.9,0.999),eps=1e-8)

        # compute the step
        data_loader_train = DataLoader(train_data, batch_size=int(num_rows))
        for image, label in data_loader_train:
            image = Variable(image)
            label = Variable(label)
            
            output = self.model(image)
            loss = criterion(output, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # take updated parameters and subtract previous ones to get delta
        new_params = np.array([p.detach().numpy() for p in self.model.parameters()])
        #print(f"prior_new: {prior_params[1]}")
        #print(f"new: {new_params[1]}")
        deltas = new_params - prior_params
        #print(f"delta: {deltas[1]}")
        return deltas

    def post_parameters(self, parameters):
        # convert payload to 
        #print(f"payload index1: {parameters[1]}")
        payload = {"parameters": [p.tolist() for p in parameters]}
        #print(payload)
        response = requests.post(self.ps_url, json=json.dumps(payload))

        if response.status_code != 200:
            return False
        else:
            return True

    def next_batch(self):
        """
        Reads next queue item and
        Returns the minibatch list
        """

        response = self.subscriber.pull(
        request={
                "subscription": self.sub_path,
                "max_messages": 1, #read one minibatch at a time. We can actually vary this up a bit more if the calls are too expensive.
            }
        )

        if len(response.received_messages) == 0:
            print(f"No messages found in {self.sub_path}")
            return False # no messages

        msg = response.received_messages[0]
        parameters = self.process_message(msg.message.data)
        success = self.post_parameters(parameters)

        # ack the messages to mark them as read.
        if success:
            ack_ids = [msg.ack_id]
            self.subscriber.acknowledge(
                request={
                    "subscription": self.sub_path,
                    "ack_ids": ack_ids,
                }
            )

        return True

def run(ps_url):
    """
    1 - consume q
    2 - prepare data
    3 - run minibatch
    4 - post parameters
    """
    learner = Learner(ps_url)

    # First, ping p.s until we get a successful result.
    server_exists = False
    while not server_exists:
        try:
            learner.update_model()
            server_exists = True
        except requests.exceptions.ConnectionError:
            print("Parameter server connection not found...trying again in 2 seconds.")
            time.sleep(2) # wait 2 seconds before pinging again

    # then run next batch 
    result = True
    c = 0
    start = time.time()
    while result == True: # still minibatches to consume
        print(f"Reading next minibatch {c}")
        learner.update_model()
        result = learner.next_batch()
        if result:
            c += 1
    end = time.time()

    print(f"Processed {c} minibatches in total. Elapsed time (s) {end-start}")

if __name__ == "__main__":
    # We should replace with real URL if not localhost.
    # If running on localhost, you might need to change ports 
    # TODO - grab this value from args or config file.
    ps_url = "http://127.0.0.1:5000/" 
    run(ps_url)