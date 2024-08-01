## Installation and Running

The repository contains implementation of both CNN and ViT for classifying tomographic reconstructions. The aim is a model
with the capacity to distinguish between what type of artifact is present in a reconstructed image. 

### Install

Install directory as a pip package

``` pip install -e . ```

### Running

The experiments directory provides sample training code. Examples of how to run one these

``` python experiments/train_cnn.py ```

### Generating reconstructions

Generating the reconstructions used for training the models. Run the script with the command.
With the command line args --directory the directory where the hdf5 or nxs data is and --data-type for type of data to generate.

``` python utils/data_rec.py --directory /user/tomo/data --data-type no-ring ```

### Inferencing with UI

Running inference on trained models run the command below or follow instructions in tomolint-app to run in apptainer.

``` python tomolint-app/app.py ```

