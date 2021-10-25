# DINO
 
## Build docker image
`cd docker` 

`docker build -f Dockerfile -t dino .` 

## Run the image
`docker run -it --gpus all -v <PATH TO YOUR DINO FOLDER>:<PATH TO YOUR DINO FOLDER> dino`
