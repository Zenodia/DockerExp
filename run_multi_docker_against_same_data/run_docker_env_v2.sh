sudo docker run -e data_path=./data -e load_model=N -e epoch=100 -e batch_size=12 -e model_save=./bz12_model.h5 --runtime=nvidia -it -p 8080:8080 --rm -v $(pwd):/workspace zenodia/unet_with_vars:v0
