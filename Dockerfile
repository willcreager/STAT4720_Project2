# Build:
#   docker build --platform linux/amd64 -t nngp-project .
#
# Run with the default (MNIST) command, mounting a folder so the PNG
# comes back out onto your machine:
#   mkdir -p output
#   docker run --platform linux/amd64 -v "$(pwd)/output":/nngp/output nngp-project
#
# Override the default to run CIFAR-10 instead (or change any flag):
#   docker run --platform linux/amd64 -v "$(pwd)/output":/nngp/output nngp-project \
#       --dataset=cifar10 --num_train=1000 --num_eval=1000 \
#       --hparams='depth=3,weight_var=2.0,bias_var=0.2' \
#       --nonlinearities='tanh,relu' \
#       --output_file=/nngp/output/uncertainty_fig3_cifar.png

FROM tensorflow/tensorflow:1.15.0-py3

WORKDIR /nngp

# Copy everything in the repo (original nngp source, grid_data/, and
# uncertainty_plot.py) into the image.
COPY . /nngp

# Two fixes needed for this old codebase to run on a current image, baked
# in at build time instead of by hand each container session:
#   1. Python 2 -> 3 (xrange doesn't exist in Python 3)
#   2. numpy now defaults to allow_pickle=False; the precomputed grid files
#      in grid_data/ were saved as pickled object arrays
RUN sed -i 's/\bxrange\b/range/g' *.py && \
    sed -i "s/np.load(f)/np.load(f, allow_pickle=True, encoding='latin1')/" \
        nngp.py

# matplotlib isn't in the base TensorFlow image, and is needed for the plot
RUN pip install --no-cache-dir matplotlib

# `docker run nngp-project` with no extra args reproduces the MNIST panel;
# any arguments passed to `docker run` after the image name replace the
# CMD list below and go straight to uncertainty_plot.py.
ENTRYPOINT ["python", "uncertainty_plot.py"]
CMD ["--dataset=mnist", "--num_train=1000", "--num_eval=1000", \
     "--hparams=depth=3,weight_var=2.0,bias_var=0.2", \
     "--nonlinearities=tanh,relu", \
     "--output_file=/nngp/output/uncertainty_fig3_mnist.png"]
