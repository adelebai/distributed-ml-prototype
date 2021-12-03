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
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
import sys
import time
from google.auth import jwt
from google.cloud import pubsub_v1
import model.model
import model.gen_dataset

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
        self.model_update = -1

    # set up credentials
    def set_credentials(self):
        service_account_info = json.load(open("service-account.json"))
        audience = "https://pubsub.googleapis.com/google.pubsub.v1.Subscriber"
        self.credentials = jwt.Credentials.from_service_account_info(
            service_account_info, audience=audience
        )

    def read_model(self):
        """
        Update the current cached model with a more recent model
        TODO - if url, need to download the file first.
        """
        self.model.load_state_dict(torch.load(self.model_url))

    def update_model(self):
        """updates the model"""
        response = requests.get(self.ps_url)
        if response.status_code != 200:
            return False
        
        response_json = response.json()
        update = int(response_json["update"])
        if self.model_update == -1 or self.model_update != update:
            self.model_update = update
            self.model_url = response["model"]
        
        self.read_model()

    def prep_data(self, data_str):
        data = pd.DataFrame([x.split(',') for x in data_str.split('\n')])
        labels = data[data.columns[-1]] # last column is label
        images = data.drop(columns=data.columns[-1])

        train_data = model.gen_dataset.my_dataset(df=images, labels=labels, transform=transform)
        return train_data


    def process_message(self, contents):
        """
        TODO - double check if this method is correct
        Effectively what we need to do here is read the minibatch, compute loss and gradients.
        Then finally take the gradient diffs and store in a list (return)
        """
        # convert bytestring to string (a csv minibatch)
        contents_str = contents.decode("utf-8")
        train_data = self.prep_data(contents_str)
        
        prior_params = np.array([p for p in self.model.parameters()]) # store prior params
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=LEARNING_RATE, betas=(0.9,0.999),eps=1e-8)

        # compute the step
        output = self.model(train_data.df)
        loss = criterion(output, train_data.labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # take updated parameters and subtract previous ones to get delta
        new_params = np.array([p for p in self.model.parameters()])
        deltas = new_params - prior_params
        return deltas

    def post_parameters(self, parameters):
        payload = {"parameters": parameters}
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
            return False, None # no messages

        msg = response.received_messages[0]
        parameters = self.process_message(msg.message.data)
        success = self.post_parameters(self, parameters)

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
    while not learner.update_model():
        time.sleep(2) # wait 2 seconds before pinging again

    # then run next batch 
    result = True
    c = 0
    start = time.time()
    while result == True: # still minibatches to consume
        print(f"Reading next minibatch {c}")
        learner.update_model()
        result = learner.next_batch()
        c += 1
    end = time.time()

    print(f"Processed {c} minibatches in total. Elapsed time (s) {end-start}")

if __name__ == "__main__":
    # Initialize any frameworks
    # We should replace with real URL if not localhost.
    # If running on localhost, you might need to change ports 
    # TODO - grab this value from args
    ps_url = "http://localhost:8080/" 
    run(ps_url)