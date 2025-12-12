% This script runs through each image in a specified directory, iterates through each quality level,
% applies median and Gaussian filtering, and then writes the output files to a new "filtered" directory

sigma = 1;

% process collected images
for quality=0:3
    if quality == 0
        imsize = 64;
    else
        imsize = 224;
    end
    files = dir(['final_pipeline_data/collected_preprocessing/collected_images_reti_quality_',num2str(quality),'/car/*.jpg']);
    gt_1 = zeros(1,2);
    gt_2 = zeros(1,2);
    gt_3 = zeros(1,2);
    num_iters_1 = 0;
    num_iters_2 = 0;
    num_iters_3 = 0;
    for file = files'
        file
        I = im2double(imread(['final_pipeline_data/collected_preprocessing/collected_images_reti_quality_',num2str(quality),'/car/',file.name]));
        I = imresize3(I,[imsize imsize 3]);
        I2 = medfilt3(I,[3 3 3]);
        I3 = imgaussfilt(I2,sigma);
        I3 = zeros(imsize, imsize, 3);
        if (num_iters_1 == 0)
            [gt_1,num_iters_1] = imdiffuseest(I2(:,:,1))
            [gt_2,num_iters_2] = imdiffuseest(I2(:,:,2))
            [gt_3,num_iters_3] = imdiffuseest(I2(:,:,3))
        end
        I3(:,:,1) = imdiffusefilt(I2(:,:,1),GradientThreshold=gt_1,NumberOfIterations=num_iters_1);
        I3(:,:,2) = imdiffusefilt(I2(:,:,2),GradientThreshold=gt_2,NumberOfIterations=num_iters_2);
        I3(:,:,3) = imdiffusefilt(I2(:,:,3),GradientThreshold=gt_3,NumberOfIterations=num_iters_3);
        dir_name = ['final_pipeline_data/collected_diffuse/filtered_images_quality_',num2str(quality)];
        mkdir(dir_name);
        mkdir([dir_name,'/car']);
        imwrite(I3,[dir_name,'/car/',file.name]);
    end
    files = dir(['final_pipeline_data/collected_preprocessing/collected_images_reti_quality_',num2str(quality),'/background/*.jpg']);
    gt_1 = zeros(1,2);
    gt_2 = zeros(1,2);
    gt_3 = zeros(1,2);
    num_iters_1 = 0;
    num_iters_2 = 0;
    num_iters_3 = 0;
    for file = files'
        file
        I = im2double(imread(['final_pipeline_data/collected_preprocessing/collected_images_reti_quality_',num2str(quality),'/background/',file.name]));
        I = imresize3(I,[imsize imsize 3]);
        I2 = medfilt3(I,[3 3 3]);
        I3 = imgaussfilt(I2,sigma);
        I3 = zeros(imsize, imsize, 3);
        if (num_iters_1 == 0)
            [gt_1,num_iters_1] = imdiffuseest(I2(:,:,1))
            [gt_2,num_iters_2] = imdiffuseest(I2(:,:,2))
            [gt_3,num_iters_3] = imdiffuseest(I2(:,:,3))
        end
        I3(:,:,1) = imdiffusefilt(I2(:,:,1),GradientThreshold=gt_1,NumberOfIterations=num_iters_1);
        I3(:,:,2) = imdiffusefilt(I2(:,:,2),GradientThreshold=gt_2,NumberOfIterations=num_iters_2);
        I3(:,:,3) = imdiffusefilt(I2(:,:,3),GradientThreshold=gt_3,NumberOfIterations=num_iters_3);
        dir_name = ['final_pipeline_data/collected_diffuse/filtered_images_quality_',num2str(quality)];
        mkdir(dir_name);
        mkdir([dir_name,'/background']);
        imwrite(I3,[dir_name,'/background/',file.name]);
    end
    files = dir(['final_pipeline_data/miotcd_preprocessing/compressed_quality_',num2str(quality),'/car/*.jpg']);
    gt_1 = zeros(1,2);
    gt_2 = zeros(1,2);
    gt_3 = zeros(1,2);
    num_iters_1 = 0;
    num_iters_2 = 0;
    num_iters_3 = 0;
    for file = files'
        file
        I = im2double(imread(['final_pipeline_data/miotcd_preprocessing/compressed_quality_',num2str(quality),'/car/',file.name]));
        I = imresize3(I,[imsize imsize 3]);
        I2 = medfilt3(I,[3 3 3]);
        I3 = imgaussfilt(I2,sigma);
        I3 = zeros(imsize, imsize, 3);
        if (num_iters_1 == 0)
            [gt_1,num_iters_1] = imdiffuseest(I2(:,:,1))
            [gt_2,num_iters_2] = imdiffuseest(I2(:,:,2))
            [gt_3,num_iters_3] = imdiffuseest(I2(:,:,3))
        end
        I3(:,:,1) = imdiffusefilt(I2(:,:,1),GradientThreshold=gt_1,NumberOfIterations=num_iters_1);
        I3(:,:,2) = imdiffusefilt(I2(:,:,2),GradientThreshold=gt_2,NumberOfIterations=num_iters_2);
        I3(:,:,3) = imdiffusefilt(I2(:,:,3),GradientThreshold=gt_3,NumberOfIterations=num_iters_3);
        dir_name = ['final_pipeline_data/miotcd_diffuse/filtered_images_quality_',num2str(quality)];
        mkdir(dir_name);
        mkdir([dir_name,'/car']);
        imwrite(I3,[dir_name,'/car/',file.name]);
    end
    files = dir(['final_pipeline_data/miotcd_preprocessing/compressed_quality_',num2str(quality),'/background/*.jpg']);
    gt_1 = zeros(1,2);
    gt_2 = zeros(1,2);
    gt_3 = zeros(1,2);
    num_iters_1 = 0;
    num_iters_2 = 0;
    num_iters_3 = 0;
    for file = files'
        file
        I = im2double(imread(['final_pipeline_data/miotcd_preprocessing/compressed_quality_',num2str(quality),'/background/',file.name]));
        I = imresize3(I,[imsize imsize 3]);
        I2 = medfilt3(I,[3 3 3]);
        I3 = imgaussfilt(I2,sigma);
        I3 = zeros(imsize, imsize, 3);
        if (num_iters_1 == 0)
            [gt_1,num_iters_1] = imdiffuseest(I2(:,:,1))
            [gt_2,num_iters_2] = imdiffuseest(I2(:,:,2))
            [gt_3,num_iters_3] = imdiffuseest(I2(:,:,3))
        end
        I3(:,:,1) = imdiffusefilt(I2(:,:,1),GradientThreshold=gt_1,NumberOfIterations=num_iters_1);
        I3(:,:,2) = imdiffusefilt(I2(:,:,2),GradientThreshold=gt_2,NumberOfIterations=num_iters_2);
        I3(:,:,3) = imdiffusefilt(I2(:,:,3),GradientThreshold=gt_3,NumberOfIterations=num_iters_3);
        dir_name = ['final_pipeline_data/miotcd_diffuse/filtered_images_quality_',num2str(quality)];
        mkdir(dir_name);
        mkdir([dir_name,'/background']);
        imwrite(I3,[dir_name,'/background/',file.name]);
    end
end
