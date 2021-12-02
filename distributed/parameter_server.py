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
"""

def get_model():
    pass

class DataFeed():
    def __init__(self, data_path):
        # The data we use in this experiment is small and can fit into memory.
        # In practice there's probably another source 
        self.q_url = ""
        self.data_path = data_path

class ParameterServer():
    def __init__(self):
        self.model = get_model()
        self.update = 0

import sys

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("usage python parameter_server.py")

    # Initialize a server    