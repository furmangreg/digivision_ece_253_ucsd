DigiVision - UCSD ECE 253 Final Project README

This repo contains relevant files used for the ECE 253 final project. All versions of the dataset are contained in the "final_pipeline_data" folder. Folders named "collected_xxxx" are the collected low-light images, compressed using different methods. Folders named "miotcd_xxxx" are the MIOTCD baseline image set, with separate subdirectories for cars/background classes. We saved off the following algorithm combinations in the "final_pipeline_data" folder:

"diffuse" - anisotropic diffusion
"gauss" - Gaussian
"gauss_diffuse" - Gaussian + anisotropic diffusion
"gauss_med" - Gaussian + median
"med" - median
"med_gauss" - median + Gaussian

First, make sure you have Python 3.6 or newer installed, as well as torchvision, tqdm and any other needed dependencies to run the script. To run evaluation on all directories mentioned above, there is a Bash script that runs model evaluation for each subdirectory. To run this Bash script, run "bash ./final_script.sh". The alternative (if you aren't on a Unix-based OS or only want to run a specific case) is to run the Python script directory. The command to do this is:
"python evaluate_binary_classifier.py --model_path model/resnet18_binary_best.pth --data-dir "$test_folder" > "${test_folder}_report.txt"

The above command will run evaluation on the specified directory, and then save off the results to a report text file.


It should not be required to run this, but there is a Matlab script "imgaussfilt_script.m", which was used to generate the filtered images mentioned above. It has been added to the repo for reference. compression_script.sh, which was used to run libjpeg to compress each fileset at the different compression levels, is also included for reference.

Low-light instructions:

Create directory collected_images.
Inside collected_images, create the following directories.
collected_images
collected_images_base
collected_images_base_gray
collected_images_dark
collected_images_reti
collected_images_ahe
collected_images_clahe

In each of the above directories, create background/ and car/ directories.
These will store the actual images.
Unzip or copy image files into collected_images/collected_images.

Make directory model/.
In model/, copy resnet18_binary_best.pth if it is not already there.

Run each cell of collected_image_AHE.ipynb.
Then, run python evaluate_binary_classifier.py --model-path model/resnet18_binary_best.pth --data-dir collected_images/collected_images_[type].
[type] should be one of the options from above (RETI, AHE, CLAHE, Dark).
