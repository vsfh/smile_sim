# triton_server_docker

1. Get the ChohoTech custom `backends.tar.gz` corresponding to version number from `share_file_server_addr/model/triton_customized_backends/`. Put the `tar.gz` file in this folder.

2. Use following command to build (you might need to use Proxy):

```bash
docker build --squash -t choho_triton_server:22.02.01 .
```

3. Use following command to run:

```bash
docker run --gpus 1 --rm -p8000:8000 -p8001:8001 -p8002:8002 -v <model_directories>:/models choho_triton_server:22.02.01 tritonserver --model-repository=/models --rate-limit execution_count --rate-limit-resource=R1:16:0
```

If you have many python models, you should add arguments like `--shm-size="1g"`

Above command suppose you have 16G GRAM on GPU 0 and you only have 1 GPU. In general, you need to specify GRAM for each GPU with one `--rate-limit-resource R1:<GRAM>:<device>` flag per GPU. For example, with GPU 0 GRAM 32G and GPU 1 GRAM 16G, you will need to use

```bash
--rate-limit-resource=R1:32:0 --rate-limit-resource=R1:16:1
```


# Build backends

The backens in the tar.gz are built from source code in Gitee.

Special notes: if you want to develop backends, please use AWS machines.
