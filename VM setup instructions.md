```
sudo apt-get update
sudo apt-get install unzip #to unzip files
sudo apt install python3-pip #install pip
pip3 install --upgrade pip #upgrade pip
```

Install the required python packages (you can find these in requirements.txt).  
Only the pytorch version is important, as I have not tested with another version. 
```
pip3 install torch==1.4.0 torchvision==0.5.0
pip3 install flask
pip3 install google-cloud-pubsub #if running distributed
pip3 install google-cloud-storage #if running distributed
pip3 install gunicorn #only if param server
pip3 install pandas
```
