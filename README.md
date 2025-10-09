# Swelling-Polymer-Project
Studying the swelling dynamics of synthetic and biopolymer by integrating in-situ imaging and advanced computational modelling. Captured image sequences of swelling polymer are then analysed using a 3D-ConvLSTM deep learning model, which is trained to quantify the rate and extent of swelling and to classify/track dynamic changes.

I’ll capture images at set intervals (not continuous video) to monitor swelling. The most common tool is Python with OpenCV (it works with most USB cameras). After installing OpenCV and connecting the camera, I’ll use the OpenCV to capture frames at desired intervals. 

A 3D-ConvLSTM model will be used for learning patterns in video or image sequences (like swelling over time). We could use TensorFlow/Keras or PyTorch. I am inclined to  Keras because it has good documentation and examples for ConvLSTM layers. For 3D-ConvLSTM, the principle is similar but with 3D convolutions. For video-captioning, we can add a language model like LSTM on top of the ConvLSTM output. 

Tools that I will use 

* Coding: Python (with Jupyter notebooks for prototyping)
* Deep Learning: TensorFlow/Keras 
* Image Processing: OpenCV 
* Data Handling: NumPy, Pandas 
* Visualization: Matplotlib, Seaborn 
* Version Control: Git 

It is decided to set up a Region of Interest (ROI) in the field of view of the camera.  As I will be placing the vials in the same spot throughout the experiment, I will hardcode the ROI in the camera capture code.  

# To Be Noted

Please run the code in Python 3.13.5 (base) or above.
