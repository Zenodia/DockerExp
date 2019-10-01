# to install docker on ubuntu 

To install docker on Ubuntu 16.04, first add the GPG key for the official Docker repository to the system:

curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo apt-key add -

Add the Docker repository to APT sources:

sudo add-apt-repository "deb [arch=amd64] https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable"

Next, update the package database with the Docker packages from the newly added repo:

sudo apt-get update

Make sure you are about to install from the Docker repo instead of the default Ubuntu 16.04 repo:

apt-cache policy docker-ce

Finally, install Docker:
sudo apt-get install -y docker-ce

# DockerExp
varying docker experiments

for Docker4Jupyter - follow below instruction to build the docker image and run it , expose port and access jupyter notebook from host machine 


1. sudo docker build -t zenodia/dl-notebook:v1 .


2.(a) sudo docker run -it -v $(pwd):/root/share2host -p 8888:8888 -i zenodia/dl-notebook:v1 

note: one can run docker with one-liner ,
one needs to install docker-compose via following instructions on this url : https://docs.docker.com/compose/install/


2.(b) sudo docker-compose up 

3. source activate py36 # inside Data Science Virtual Machine on Azure 
4. firefox 
5 enter in the browser --> 0.0.0.0:8888 
6. run your notebook 
7. when you are down , go to the terminal where the jupyter notebook is running and enter Ctrl+C to shut down container image
sudo docker container stop ztest
sudo docker system prune --all
sudo docker volume prune 

or with one-liner 
sudo docker-compose down --volume 


for bash_with_docker , follow this blog post -


https://www.linkedin.com/pulse/use-bash-command-wrap-docker-build-run-stop-zenodia-charpy/

