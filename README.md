Tomolint is a tool for detecting common tomography reconstruction artifacts.

# Introduction

New users of tomographic reconstruction often cannot recognize common
reconstruction artifacts which are the result of the tomographic reconstruction
and not real features of their object of interest. These artifacts are caused
by incorrect reconstruction parameters or experimental conditions which do not
match the computiational model or assumptions of the tomographic
reconstruction.

The aim of this tool is to help new users detect these artifacts and to provide
a short description of common approaches to mitigate the artfiact. In other
words, the tool is meant to accelerate user learning, but providing automated
feedback on their reconstructed images that is tailored to their use case.

# The App

The `tomolint-app` folder contains a web-based user interface built using the
`gradio` library. This app can be built and distributed in container format
using Apptainer. The container will start a web-server which runs on the local
machine and is accessible using a web browser.

# The library

The `tomolint` folder contains machine learning models and other infrastructure
used to detect the tomographic artifacts.
