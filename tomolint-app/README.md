## Installation and Running the Tomolint App

The documentation is tested to run on the APS linux based machines. Apptainer provides an alternative with the need of root privilegedes. This are the steps for installing and running the front-end application. 

#### Install Apptainer

1. Clone the git repository 

    ``` git clone https://github.com/tomopy/tomolint.git ```

2. Installing Apptainer
    a. Install using the easy install from the apptainer official documentation on [Linux](https://apptainer.org/docs/user/main/definition_files.html)

    ``` curl -s https://raw.githubusercontent.com/apptainer/apptainer/main/tools/install-unprivileged.sh | \ bash -s - install-dir ```

    b. The previous step creates a local directory with the Apptainer binary installed. It should present in the directory below.

    ``` install/bin/apptainer  ```

    c. Navigate to the app directory contains the definition file.

    ``` cd tomolint/tomolint-app ```

    d. Create and install an Apptainer conda virtual environment.

    ``` conda create -n apptainer apptainer --yes ```

    e. Build apptainer with the command below. Note the build-arg correspond to the working conda branch of this repository this overwrites the default main branch present in the definition file.
    
    ``` apptainer build --build-arg pkg_version=conda tomolint.sif tomolint.def ```

3. Run the app 
``` /install-dir/bin/apptainer run ./tomolint.sif ```