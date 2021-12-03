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
```

Data is written to [google pub/sub](https://cloud.google.com/pubsub/lite/docs/quickstart#pubsublite-quickstart-publisher-python).


## First, data needs to be sent to a queue  
Example usage  
```
cd distributed
python data_feed.py ../data/hmnist_28_28_RGB.csv <specify epochs>
```

## Then, run the parameter server which will host a local API.