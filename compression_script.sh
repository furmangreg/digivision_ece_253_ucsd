#! /bin/bash
#jpg_folder="original/uncompressed_jpg"
#ppm_folder="original/uncompressed_ppm"
#output_folder="original/compressed_quality_"

# code to convert images from jpg to ppm (required format for libjpeg)
#for file in $jpg_folder/*/*.jpg; do
#	IFS='/' read -a filepathsplit <<< "$file"
#	IFS='.' read -a filename <<< "${filepathsplit[3]}"
#	newfilepath=$ppm_folder/${filepathsplit[2]}/$filename.ppm
#	convert -resize 224x224\! $file $newfilepath 
#done

# code to compress images
#for i in {1..10}; do
	#mkdir "$output_folder$i"
	#mkdir "$output_folder$i/background"
	#mkdir "$output_folder$i/car"
#	for file in $ppm_folder/*/*.ppm; do
#		IFS='/' read -a filepathsplit <<< "$file"
#		IFS='.' read -a filename <<< "${filepathsplit[3]}"
#		newfilepath=$output_folder$i/${filepathsplit[2]}/$filename.jpg
#		cjpeg -quality $i $file > $newfilepath
#	done
#done
#
#mkdir "miotcd/ultra_compressed"
#mkdir "miotcd/ultra_compressed/background"
#mkdir "miotcd/ultra_compressed/car"
#mkdir "final_pipeline_data/collected_images_dark/collected_images_dark_quality_0"
#mkdir "final_pipeline_data/collected_images_dark/collected_images_dark_quality_0/background"
#mkdir "final_pipeline_data/collected_images_dark/collected_images_dark_quality_0/car"
for file in final_pipeline_data/collected_images_dark/collected_images_dark_quality_1/*/*.jpg; do
	IFS='/' read -a filepathsplit <<< "$file"
	IFS='.' read -a filename <<< "${filepathsplit[4]}"
	newppmfilepath="final_pipeline_data/collected_images_dark/collected_images_dark_quality_0/${filepathsplit[3]}/$filename.ppm"
	newjpgfilepath="final_pipeline_data/collected_images_dark/collected_images_dark_quality_0/${filepathsplit[3]}/$filename.jpg"
	convert -resize 32x32\! $file $newppmfilepath
	echo "$file\n"
	cjpeg -quality 1 $newppmfilepath > $newjpgfilepath
	rm $newppmfilepath
done
