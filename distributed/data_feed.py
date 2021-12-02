"""
This was originally going to be coupled with the parameter server, but can probably be decoupled for easier runs. 
Data is written to google pub/sub
https://cloud.google.com/pubsub/lite/docs/quickstart#pubsublite-quickstart-publisher-python
"""
import csv
import json
from google.auth import jwt
from google.cloud import pubsub_v1

class DataFeed():
    def __init__(self, data_path, epochs):
        # The data we use in this experiment is small and can fit into memory.
        # In practice there's probably another source 
        self.project_id = "project-distml-ayb2121-yz4053"
        self.topic_id = "data"

        # set up credentials
        self.set_credentials()

        self.data_path = data_path
        self.epochs = epochs
        self.batch_size = 128 #TODO make configurable

    def set_credentials(self):
        service_account_info = json.load(open("service-account.json"))
        audience = "https://pubsub.googleapis.com/google.pubsub.v1.Publisher"
        self.credentials = jwt.Credentials.from_service_account_info(
            service_account_info, audience=audience
        )

    def prepare_data(self):
        batches = []
        # returns the minibatches of messages
        # reads csv
        with open(self.data_path) as csvfile:
            r = csv.reader(csvfile, delimiter=',')
            next(r) # skip header

            minibatch = []
            i = 0
            for row in r:
                image_row = ', '.join(row)
                minibatch.append(image_row)
                i += 1

                if i%self.batch_size == 0 and i != 0:
                    batches.append("\n".join(minibatch))
                    minibatch = []
            
            # We'll need to figure out if we want to append this final set.
            #if i%self.batch_size != 0:
                # append leftover minibatch - we probably need to make sure there's enough data here. 
            #    batches.append("\n".join(minibatch))
        return batches

    def run(self):
        # load csv file
        # chunk data into 128 row sized csv strings
        # TODO - built in shuffle so we can re-prepare every epoch such that each minibatch is different. 
        all_batches = self.prepare_data()

        # send to queue
        # TODO check if credentials needed.
        publisher = pubsub_v1.PublisherClient(credentials=self.credentials)
        topic_path = publisher.topic_path(self.project_id, self.topic_id)

        for i in range(self.epochs):
            for b in all_batches:
                future = publisher.publish(topic_path, bytes(b, 'utf-8')) # need to convert payload into bytes
                future.result()
            print(f"Completed sending for epoch {i}")

        print(f"Published messages to {topic_path}.")


import sys
import time

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("usage python data_feed.py <data file path> <epochs>")

    # prepare data
    data_path = sys.argv[1]
    epochs = int(sys.argv[2])

    # send
    start = time.time()
    print(f"Preparing data from {data_path} and sending for {epochs} epochs.")
    feed = DataFeed(data_path, epochs)
    print(f"Sending data to feed {feed.project_id} {feed.topic_id}")
    feed.run()
    end = time.time()

    elapsed_s = end-start
    print(f"Finished Sending. Time taken (s): {elapsed_s}")
