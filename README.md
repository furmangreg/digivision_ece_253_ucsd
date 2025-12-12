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

---

Color banding instructions:

Prerequisites:
Install dependencies: torch, torchvision, pillow, pandas, tqdm, numpy

Dataset:
Sample data is provided in final_pipeline_data/miotcd_preprocessing/ with structure:
  compressed_quality_0/
    car/
    background/
  compressed_quality_1/
    car/
    background/
  compressed_quality_2/
    car/
    background/
  compressed_quality_3/
    car/
    background/

The scripts are pre-configured to use this sample data.

Run dithering experiment:
cd color_banding/dithering
python run_dithering_experiment.py

Run gaussian blur experiment:
cd color_banding/gaussian_blur
python run_gaussian_experiment.py

Output:
Results saved to CSV files in the experiment directory.
Processed images saved to dithered_output/ or blurred_output/ in the experiment directory.

Using your own dataset:
Edit configuration section in the scripts:
  - BASE_DIR: Path to your dataset directory
  - COMPRESSION_LEVELS: List of subdirectory names to test
  - OUTPUT_DIR: Where to save processed images
  - BLUR_RADII (gaussian only): Blur strengths to test
