FROM nvcr.io/nvidia/tensorflow:19.10-py3
ARG 1_data 
ARG 2_load 
ARG 3_epochs 
ARG 4_batch_sz 
ARG 5_save_path
ARG 6_gpu_num

ENV data_path $1_data
ENV load_model $2_load
ENV epoch $3_epochs
ENV batch_size $4_batch_sz
ENV model_save $5_save_path
ENV gpu_device_num $6_gpu_num
RUN apt-get update
RUN apt-get install -y libsm6 libxext6 libxrender-dev

# Install any needed packages specified in requirements.txt
RUN pip install opencv-python opencv-contrib-python scikit-image scikit-learn pillow scipy scipy

# Make port 80 available to the world outside this container
EXPOSE 8888


# debug wrapper, followed by the model training script to execute
CMD ["python","unet_with_env_vars.py" ]