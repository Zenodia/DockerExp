sudo docker run -e data_path=./data -e load_model=N -e epoch=100 -e batch_size=24 -e model_save=./bz24_weight.h5 --runtime=nvidia -it -p 8181:8181 --rm -v $(pwd):/workspace zenodia/unet_with_vars:v0
