"""
    This is a simi-automated script to generate training data 
    it uses the tomopy-cli.


"""

import os
import subprocess
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging
import shutil
import dxchange
import tomopy
import numpy as np
import algotom.io.converter as conv
import algotom.io.loadersaver as losa
import yaml
from tqdm import tqdm
from tomopy_cli import find_center
from types import SimpleNamespace


import algotom.io.loadersaver as losa
import algotom.prep.correction as corr
import algotom.prep.calculation as calc
import algotom.rec.reconstruction as rec
import algotom.prep.removal as remo
import algotom.prep.filtering as filt
import algotom.util.utility as util

logging.basicConfig(level=logging.INFO)
import random

##TODO:
# 1. Fix when no value for cor, when yaml file is not present
# 2. reconstruct can continuning from where it stopped


def load_center_rotations(file_path):
    base = os.path.dirname(file_path)
    yml_file = os.path.join(base, "extra_params.yaml")
    with open(yml_file, "r") as file:
        data = yaml.safe_load(file)
    return data


def extract_tomo_id(file_path):
    base_name = os.path.basename(file_path)
    tomo_id = base_name.split("_")[1].split(".")[0]
    return tomo_id


def pixel_offset(rot):
    """
    Pixel offset to generate wrong center of rotation

    """
    if random.choice([True, False]):
        return rot + 100
    else:
        return rot - 100


def run_reconstruction_file(file_path, stripe_method):
    """
    Creates reconstructions
    """

    rec_dir = file_path.replace(".h5", "_rec")

    if os.path.exists(rec_dir):
        shutil.rmtree(rec_dir)
    os.makedirs(rec_dir)

    # find all centers of rotations in the dir
    # find_center_rot(file_path)
    center_rotations = load_center_rotations(file_path)
    tomo_id = extract_tomo_id(file_path)

    rot_center = center_rotations.get(os.path.basename(file_path), {}).get(
        "rotation-axis"
    )

    if stripe_method == "bad-center":
        # TODO: uncomment to generate to wrong center of rotation
        # rot_center = pixel_offset(rot_center)
        if rot_center is None:
            logging.info(
                f"Rotation center not found for {file_path}. Selectining a random center..."
            )
            rot_center = random.uniform(1.0, 1500.0)
            logging.info(f"Calculated rotation center: {rot_center}")
        else:
            rot_center = pixel_offset(rot_center)

    base_command = "tomopy recon --rotation-axis-auto manual --reconstruction-type full --nsino-per-chunk 1024 --fix-nan-and-inf True \
    --remove-stripe-method fw --rotation-axis <rotation_axis> --file-name <file_path>"

    command = base_command.replace("<file_path>", file_path).replace(
        "<rotation_axis>", str(rot_center)
    )

    command = command.strip().split()

    if stripe_method == "with-ring":
        command = [arg.replace("fw", "none") for arg in command]
        print("command when false ", command)

    logging.info(
        f"Running reconstruction on {file_path} with command: {' '.join(command)}"
    )
    print(f"Running reconstruction on {file_path}")
    subprocess.run(command)


# def run_reconstruction(directory, stripe_method):
#     files = [f for f in os.listdir(directory) if f.endswith(".h5")]
#     total_files = len(files)
#     print(f"Total number of .h5 files: {total_files}")

#     with tqdm(total=total_files) as pbar:
#         files_to_reconstruct = [os.path.join(directory, f) for f in files]
#         print(files_to_reconstruct)
#         with ThreadPoolExecutor(max_workers=2) as executor:
#             futures = [
#                 executor.submit(run_reconstruction_file, file_path, stripe_method)
#                 for file_path in files_to_reconstruct
#             ]
#             for future in as_completed(futures):
#                 future.result()  # To raise exceptions if any
#                 pbar.update(1)


def run_algotom(file_path, stripe_method):
    """
    Run reconstruction using algotom library

    """

    output_base = file_path.replace(".hdfs", "_rec").replace(".nxs", "_rec")
    rec_dir = file_path.replace(".hdfs", "_rec").replace(".nxs", "_rec")

    ## TODO: clean up key search

    # Provide path to datasets in the nxs file.
    data_key = "/entry1/tomo_entry/data/data"
    image_key = "/entry1/tomo_entry/instrument/detector/image_key"
    angle_key = "/entry1/tomo_entry/data/rotation_angle"

    ikey = np.squeeze(np.asarray(losa.load_hdf(file_path, image_key)))
    angles = np.squeeze(np.asarray(losa.load_hdf(file_path, angle_key)))
    data = losa.load_hdf(file_path, data_key)  # This is an object not ndarray.
    print("shsnspe ", data.shape)
    (depth, height, width) = data.shape

    # Load dark-field images and flat-field images, averaging each result.
    print("1 -> Load dark-field and flat-field images, average each result")
    dark_field = np.mean(
        np.asarray(data[np.squeeze(np.where(ikey == 2.0)), :, :]), axis=0
    )
    flat_field = np.mean(
        np.asarray(data[np.squeeze(np.where(ikey == 1.0)), :, :]), axis=0
    )
    print("2 -> Save few projection images as tifs")
    proj_idx = np.squeeze(np.where(ikey == 0))
    proj_corr = corr.flat_field_correction(
        data[proj_idx[0], 10:, :], flat_field[10:], dark_field[10:]
    )
    # losa.save_image(output_base + "/proj_corr/ff_corr_00000.tif", proj_corr)

    # Perform flat-field correction in the sinogram space and save the result.
    print("3 -> Generate a sinogram with flat-field correction and save the result")
    index = height // 2  # Index of a sinogram.
    sinogram = corr.flat_field_correction(
        data[proj_idx[0] : proj_idx[-1], index, :],
        flat_field[index, :],
        dark_field[index, :],
    )
    print("sinogram shape ", sinogram.shape)
    # losa.save_image(output_base + "/sinogram/sinogram_mid.tif", sinogram)
    print("4 -> Calculate the center-of-rotation")
    # center = calc.find_center_vo(sinogram, width//2-50, width//2+50)
    center = calc.find_center_vo(sinogram)
    print("Center-of-rotation is {}".format(center))
    # Perform reconstruction and save the result.
    # Users can choose CPU-based methods as follows
    thetas = angles[proj_idx[0] : proj_idx[-1]] * np.pi / 180
    # # DFI method, a built-in function:
    start_slice = 500
    stop_slice = 750

    print("5 -> Perform reconstruction without artifact removal methods")
    if stripe_method == "no-ring":
        # Options to include removal methods in the flat-field correction step.
        opt1 = {"method": "remove_zinger", "para1": 0.08, "para2": 1}
        opt2 = {"method": "remove_all_stripe", "para1": 3.0, "para2": 51, "para3": 17}
        # Load sinograms, and perform pre-processing.
        sinograms = corr.flat_field_correction(
            data[proj_idx[0] : proj_idx[-1], start_slice:stop_slice, :],
            flat_field[start_slice:stop_slice, :],
            dark_field[start_slice:stop_slice, :],
            option1=opt1,
            option2=opt2,
        )
        # Perform reconstruction
        print("9 -> Perform reconstruction on this chunk in parallel...")
        recon_img = util.parallel_process_slices(
            sinograms, rec.dfi_reconstruction, [center]
        )
        for i in range(start_slice, stop_slice):
            name = "0000" + str(i)
            losa.save_image(
                rec_dir + "/rec_" + name[-5:] + ".tiff",
                recon_img[:, i - start_slice, :],
            )
        print("!!! Done !!!")

    elif stripe_method == "with-ring":
        opt1 = None
        opt2 = None
        sinograms = corr.flat_field_correction(
            data[proj_idx[0] : proj_idx[-1], start_slice:stop_slice, :],
            flat_field[start_slice:stop_slice, :],
            dark_field[start_slice:stop_slice, :],
            option1=opt1,
            option2=opt2,
        )

        recon_img = util.parallel_process_slices(
            sinograms, rec.dfi_reconstruction, [center]
        )
        for i in range(start_slice, stop_slice):
            name = "0000" + str(i)
            losa.save_image(
                rec_dir + "/rec_" + name[-5:] + ".tiff",
                recon_img[:, i - start_slice, :],
            )
        print("!!! Done !!!")

    elif stripe_method == "bad-center":
        opt1 = None
        opt2 = None

        center = pixel_offset(center)

        sinograms = corr.flat_field_correction(
            data[proj_idx[0] : proj_idx[-1], start_slice:stop_slice, :],
            flat_field[start_slice:stop_slice, :],
            dark_field[start_slice:stop_slice, :],
            option1=opt1,
            option2=opt2,
        )

        recon_img = util.parallel_process_slices(
            sinograms, rec.dfi_reconstruction, [center]
        )
        for i in range(start_slice, stop_slice):
            name = "0000" + str(i)
            losa.save_image(
                rec_dir + "/rec_" + name[-5:] + ".tiff",
                recon_img[:, i - start_slice, :],
            )
        print("!!! Done !!!")

    else:
        print("Invalid stripe method")


def run_reconstruction(directory, stripe_method):
    # List all files in the directory
    all_files = os.listdir(directory)

    # Separate .h5 and (hdfs, nxs) files
    h5_files = [f for f in all_files if f.endswith(".h5")]
    other_files = [f for f in all_files if f.endswith((".nxs"))]

    total_files = len(h5_files) + len(other_files)
    print(f"Total number of files: {total_files}")
    print(f"Number of .h5 files: {len(h5_files)}")
    print(f"Number of (hdfs, nxs) files: {len(other_files)}")

    with tqdm(total=total_files) as pbar:
        # Prepare file paths for .h5 files
        h5_files_to_reconstruct = [os.path.join(directory, f) for f in h5_files]
        other_files_to_process = [os.path.join(directory, f) for f in other_files]

        print(h5_files_to_reconstruct)
        print(other_files_to_process)

        with ThreadPoolExecutor(max_workers=2) as executor:
            # Submit reconstruction tasks for .h5 files
            h5_futures = [
                executor.submit(run_reconstruction_file, file_path, stripe_method)
                for file_path in h5_files_to_reconstruct
            ]

            # Submit algorithm tasks for (hdfs, nxs) files
            other_futures = [
                executor.submit(run_algotom, file_path, stripe_method)
                for file_path in other_files_to_process
            ]

            # Combine futures and update progress bar as they complete
            all_futures = h5_futures + other_futures
            for future in as_completed(all_futures):
                future.result()  # To raise exceptions if any
                pbar.update(1)


def copy_tiff_file(src_path, dest_path, file, dest_directory):
    shutil.copy(src_path, dest_path)
    print(f"Copied {file} to {dest_directory} as {os.path.basename(dest_path)}")


def copy_tiff_files(src_directory, dest_directory):
    print("destination ", dest_directory)
    os.makedirs(dest_directory, exist_ok=True)
    # src_directory = src_directory + "_rec"
    files_to_copy = []
    for root, _, files in os.walk(src_directory):
        for file in files:
            if file.endswith(".tiff"):
                src_path = os.path.join(root, file)
                parent_dir_name = os.path.basename(root)
                dest_dir_name = os.path.basename(dest_directory)
                new_file_name = f"{dest_dir_name}_{parent_dir_name}_{file}"
                dest_path = os.path.join(dest_directory, new_file_name)
                files_to_copy.append((src_path, dest_path, file))

    with ThreadPoolExecutor() as executor:
        futures = [
            executor.submit(copy_tiff_file, src_path, dest_path, file, dest_directory)
            for src_path, dest_path, file in files_to_copy
        ]
        for future in as_completed(futures):
            future.result()  # To raise exceptions if any


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run reconstruction on all .h5 files in a directory."
    )
    parser.add_argument(
        "--directory", type=str, help="The directory containing the .h5 files."
    )

    # Add the arguments
    parser.add_argument(
        "--data-type",
        choices=["with-ring", "no-ring", "bad-center"],
        nargs="?",
        default="with-ring",
        help="Choose one of the available options to generate data : with-ring( default ), no-ring, or bad-center.",
    )

    args = parser.parse_args()

    # Handle the options
    if args.data_type == "with-ring":
        file_name = "datasets-with-ring"
        print(file_name)
    elif args.data_type == "no-ring":
        file_name = "datasets-no-ring"
        print(file_name)
    elif args.data_type == "bad-center":
        file_name = "bad-center"
        print(file_name)
    else:
        print("Invalid option")

    run_reconstruction(args.directory, args.data_type)

    # Define the destination directory
    datasets_directory = os.path.join(args.directory + "_rec", file_name)
    # Copy the .tiff files to the datasets directory
    copy_tiff_files(args.directory, datasets_directory)
