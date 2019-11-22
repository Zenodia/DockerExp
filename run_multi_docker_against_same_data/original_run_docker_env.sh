sudo docker run -e data_path=./data -e load_model=N -e epoch=2 -e batch_size=12 -e model_save=./checkpoint.h5 --runtime=nvidia -it -p 8888:8888 --rm -v $(pwd):/workspace zenodia/unet_with_vars:v0
