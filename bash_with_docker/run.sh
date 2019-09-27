sudo docker container run -d -it --name $1 -p 8080:8080 -v $(pwd):/usr/share2host ztest:latest 
