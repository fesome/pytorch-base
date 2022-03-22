

# pytorch基础镜像



## 镜像打包

```sh
$ sudo docker build . -f Dockerfile.cpu -t sensingx/pytorch:1.10.2-nocuda-ubuntu20.04

$ sudo docker build . -f Dockerfile.gpu -t sensingx/pytorch:1.10.2-cuda11.3-ubuntu20.04
```



## 运行

```sh
$ sudo docker run --rm -it --init \
  --gpus=all \
  --ipc=host \
  --user="$(id -u):$(id -g)" \
  --volume="$PWD:/app" \
  sensingx/pytorch:1.10.2-cuda11.3-ubuntu20.04 python3 main.py
```



