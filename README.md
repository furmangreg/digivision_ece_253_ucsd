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
