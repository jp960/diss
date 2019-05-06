###################################################################################################

CM30082 - Individual Project
Generating 3D data from 2D sketches using Neural Networks
Janhavi Pal
Bachelor of Science in Computer Science with Honours
The University of Bath 
2019

The code base for my individual project is available at: https://github.com/jp960/diss/

Prerequisites:
- Linux
- Python 3 
- NVIDIA GPU + CUDA 9.0 + CuDNNv7.0.5
- TensorFlow 1.5.0

Running the code:
	- Train the model: 	python3 main.py --phase train
	- Test the model: 	python3 main.py --phase test
Optional Parameters:
--dataset_name   - name of the dataset
--train_size     - number of images used to train
--batch_size     - number of images in batch
--epoch          - number of of epoch
--sample_size    - number of images when sampling
--lr             - learning rate
--phase          - train or test
--checkpoint_dir - models are saved here
--sample_dir     - sample images are saved here
--test_dir       - testing sample images are saved here
--destdr         - output folder


Repository Structure:
______________________________________________________________________________________________
_______File/Directory_Name______|_________________________Description_________________________
NYU                             | NYU Dataset of preprocessed and depth data
SUNRGBD                         | SUNRGBD Dataset of preprocessed and depth data
TestingData                     | Testing Dataset of preprocessed and depth data
practise                        | Dataset of preprocessed and depth data with 1 sample
practise10                      | Dataset of preprocessed and depth data with 10 samples
.gitignore                      | 
imageProcess.py                 | Script to experiment with preprocessing filters
loss_check_script.py            | Script to test loss output
main.py                         | Main file that runs the network
model.py                        | Class for model and all model methods
model.yml                       | Config yaml for structured forests filter
ops.py                          | All functions used in the generator
plot_loss_graph.py              | Script to plot results graphs
runPreProcessingForAllImages.py | Script to run preprocessing for all samplesin a directory
utils.py                        | All utility functions used by the model
________________________________|_____________________________________________________________

###################################################################################################