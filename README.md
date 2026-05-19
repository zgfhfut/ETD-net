# Learning to Ignore: Conservative Frequency Suppression Enables Robust Audio Tampering Detection under Physical Recapture

This repository contains the test code and pre-trained weights for our manuscript on robust audio tampering detection under physical recapture.

## Hardware Configuration

The following hardware configuration was used in our experiments:

- CPU: AMD Ryzen 7 3700X 8-Core Processor
- GPU: NVIDIA GeForce RTX 2070 SUPER (8 GB)
- RAM: 16 GB
- OS: Windows 10

## Environment Requirements

To ensure the detection results are consistent with the paper, please strictly configure the environment as follows:

- Python 3.6
- Keras==2.3.1
- tensorflow-gpu==1.14.0
- numpy==1.18.5
- opencv-python==3.4.2.16
- matplotlib==3.3.4
- tqdm==4.64.1
- h5py==2.10.0
- Pillow==8.4.0
- scipy==1.4.1

## File Description

| File | Description |
|------|-------------|
| test.py | Main test script. Loads the pre-trained model and performs tampering detection on mel-spectrogram images. |
| ETD_net.py | Model architecture definition (ETD-Net). |
| load.py | Custom layers (PAM, CAM, VGG block, etc.) and data loading utilities. |
| densenet.py | DenseNet block modules. |
| cbam.py | CBAM attention mechanism modules. |
| Optimize.py | Post-processing functions (threshold filtering and continuous column removal). |
| epoch-27.h5 | Pre-trained model weights (27th epoch). |

## Dataset

The test dataset is available at: https://www.scidb.cn/s/bmu2ie

Please download the test set from the above link for evaluation.

## How to Run the Test

1. Prepare Input Data

   Place your mel-spectrogram images (PNG format) in a directory. The model expects input size of 640 x 480.

2. Modify Paths in test.py

   Open test.py and modify the following two paths:
   - Input path: replace r'your mel path' with your actual mel-spectrogram directory.
   - Output path: replace r'your result path' with your desired result save directory.

3. Adjust Sample Range (Optional)

   The variable k in the loop controls which samples to detect:
   for k in tqdm(range(1, 434), position=0):
   You can modify the range according to your actual number of samples.

4. Run

   python test.py

## Output

For each input mel-spectrogram, the program will generate a corresponding result image in the specified output directory, with _res.png appended to the original filename (e.g., 1_mel.png -&gt; 1_res.png).

The output is a binary mask indicating the detected tampered regions.

## Notes

- The model weights file epoch-27.h5 must be placed in the same directory as test.py, or modify the load_weights() path accordingly.
- The input must be mel-spectrogram images. Raw audio files need to be converted to mel-spectrograms first before testing.
- GPU is recommended for inference. If running on CPU, modify the CUDA settings in test.py or remove the GPU configuration lines.
- Please strictly follow the experimental environment provided above. Otherwise, the detection performance cannot be guaranteed.
