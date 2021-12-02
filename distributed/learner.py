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

Payload to send the P.S - will be fleshed out.
{
    "parameters": ??
}
"""

def prep_data():
    pass

def run(ps, q):
    """
    1 - consume q
    2 - prepare data
    3 - run minibatch
    4 - post parameters
    """

    pass

def run_minibatch(data):
    # returns a set of parameters
    pass

if __name__ == "__main__":
    # Initialize any frameworks
    ps_url = "" #TODO
    queue_url = ""

    # ping p.s until ready TODO
    run(ps_url, queue_url)