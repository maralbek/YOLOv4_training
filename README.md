# Training customized YOLOv4 model

## Setting up GPU Drivers and Environment

1. **Install GPU Drivers:** 
    - Set up GPU drivers using the instructions [here](https://qiita.com/sebastianrettig/items/33ead90d3bde4cc9b6b0) for a clean installation in a separate CONDA environment.
    - Create a CONDA environment:
        ```bash
        conda create -n YOLOv4_conda python=3.7 opencv=3
        ```
    - Install CUDA-Toolkit and cuDNN library:
        ```bash
        conda install cuda cudnn -c nvidia -n YOLOv4_conda
        ```
    - Activate the environment:
        ```bash
        conda activate YOLOv4_conda
        ```

## Cloning and Configuring Darknet Framework

2. **Clone and Configure Darknet Framework:**
    - Clone the repository and navigate to it in the terminal:
        ```bash
        git clone https://github.com/AlexeyAB/darknet.git
        cd darknet
        ```
    - Open and edit the Makefile, setting GPU, CUDNN, CUDNN_HALF, and OPENCV to 1.
      
    - Uncomment the compute capability of your GPU in the Makefile.
    - Build Darknet:
        ```bash
        sudo make
        ```

## Dataset Preparation

3. **Prepare Dataset for Training:**
    - Create the dataset according to [Darknet documentation](https://github.com/AlexeyAB/darknet#how-to-train-to-detect-your-custom-objects) in the `/darknet/data` folder or any location within the `darknet` repository folder.
    - Necessary files:
        - `obj.names`: Contains class names.
        - `obj.data`: Information with paths to required files.
        - `train.txt` and `test.txt`: Lists of images for training and testing.
        - `obj_train_data`: Folder containing images and corresponding annotations.
    - Annotation format:
        ```
        <object-class> <x_center> <y_center> <width> <height>
        ```
    ![Dataset Structure](https://prod-files-secure.s3.us-west-2.amazonaws.com/96f8a452-7206-4cb0-94b8-7fb06cb53922/a32f5f51-f712-469d-a6c5-85abbff43df5/Untitled.png)

## Download Pretrained Model and Configuration

4. **Download Pretrained Model and Configuration:**
    - Download the model:
        ```bash
        wget https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v3_optimal/yolov4.conv.137
        ```
    - Copy `yolov4-custom.cfg` into the dataset folder:
        ```bash
        cp cfg/yolov4-custom.cfg data/
        ```
    - Customize the configuration as per the [training manual](https://github.com/AlexeyAB/darknet#how-to-train-to-detect-your-custom-objects).
    
## Training the Model

5. **Train the Model:**
    - Start training:
        ```bash
        sudo ./darknet detector train /data/obj.data /data/yolov4-custom.cfg /data/yolov4.conv.137 -map
        ```
    - Monitor the training process.

    ![Training Process](https://prod-files-secure.s3.us-west-2.amazonaws.com/96f8a452-7206-4cb0-94b8-7fb06cb53922/806346c9-2d8c-463e-82fa-6e5895394464/Screenshot_2024-03-12_at_12.39.05_PM.png)

    - Refer to [tips](https://haobin-tan.netlify.app/ai/computer-vision/object-detection/yolov4-training-tips/) for helpful insights during training.

6. **Testing the Model**
7. **Convert to TF2 Model**
