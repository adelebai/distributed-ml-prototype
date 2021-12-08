# distributed-ml-prototype
Distributed ML prototype architecture for COMES6998 project

# test run local serial version
```
cd model
python model.py
```

Make sure the csv file is in the right directory in /data


# How to run distributed version
Files are located in the /distributed folder. There are some library depencencies needed like. 

```
pip install google-cloud-pubsub
pip install google-cloud-storage
```

There are hardcoded instances of the pubsub, storage account and http server 

Data is written to [google pub/sub](https://cloud.google.com/pubsub/lite/docs/quickstart#pubsublite-quickstart-publisher-python).

You will need your own instance of pub/sub if trying to run this. 

## First, data needs to be sent to a queue  
Example usage  
```
cd distributed
python data_feed.py ../data/hmnist_28_28_RGB.csv <specify epochs>
```

## Then, run the parameter server which will host a local API.
Example usage
```
python parameter_server.py
```

If running from a GCP VM, this is the command:
```
~/.local/bin/gunicorn -b :8080 parameter_server:app
```

This will set up a server on localhost if running locally. 

## Finally, you need to create a separate instance (while parameter_server is running) and run learner.py  
You may need to edit the main function here and replace the ps_url with your own.   
Example usage
```
python learner.py
```