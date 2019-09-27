# DockerExp
varying docker experiments

for Docker4Jupyter - follow below instruction to build the docker image and run it , expose port and access jupyter notebook from host machine 


1. sudo docker build -t zenodia/dl-notebook:v1 .
2. sudo docker run -it -v $(pwd):/root/share2host -p 8888:8888 -i zenodia/dl-notebook:v1 
3. source activate py36 # inside Data Science Virtual Machine on Azure 
4. firefox 
5 enter in the browser --> 0.0.0.0:8888 
6. run your notebook 
