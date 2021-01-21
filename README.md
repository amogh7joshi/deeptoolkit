# DeepToolKit

DeepToolKit provides implementations of popular machine learning algorithms, extensions to existing
deep learning pipelines using TensorFlow and Keras, and convenience utilities to speed up the process
of implementing, training, and testing deep learning models. In addition, DeepToolKit includes an inbuilt 
computer vision module containing implementations of facial detection and image processing algorithms. 

## Installation

### Python Package

DeepToolKit can be installed directly from the command line:

```shell script
pip install deeptoolkit
```

You can then work with it either by importing the library as a whole, or by importing 
the functionality you need from the relevant submodules.

```python
# Complete library import.
import deeptoolkit as dtk

# Module and function imports.
from deeptoolkit.data import plot_data_cluster
from deeptoolkit.blocks import SeparableConvolutionBlock
from deeptoolkit.losses import CategoricalFocalLoss
```

### From Source

If you want to install DeepToolKit directly from source, (i.e. for local development), then first
install the git source:

```shell script
git clone https://github.com/amogh7joshi/deeptoolkit.git
```

Then install system requirements and activate the virtual environment. A Makefile is included for installation:

```shell script
make install
```


## Features

DeepToolKit provides a number of features to either use standalone or integrated in a deep learning model 
construction pipeline. Below is a high-level list of features in the module. Proper documentation is under construction.

### Model Architecture Blocks: `deeptoolkit.blocks`

- Generic model architecture blocks, including convolution and depthwise separable convolution blocks, implemented as 
`tf.keras.layers.Layer` objects so you can directly use them in a Keras model.
- Applied model architecture blocks, including squeeze and excitation blocks and ResNet identity blocks.

**For Example**:

```python
from tensorflow.keras.models import Model
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Input, Dense, Flatten
from deeptoolkit.blocks import ConvolutionBlock

# Construct a Keras Functional model like normal.
inp = Input((256, 256, 3))
x = ConvolutionBlock(32, kernel_size = (3, 3), activation = 'relu')(inp)
x = MaxPooling2D(pool_size = (2, 2))(x)
x = ConvolutionBlock(16, kernel_size = (3, 3), activation = 'relu')(x)
x = MaxPooling2D(pool_size = (2, 2))(x)
x = Flatten()(x)
x = Dense(1024, activation = 'relu')(x)
x = Dense(10, activation = 'relu')(x)
model = Model(inp, x)
```

### Loss Functions: `deeptoolkit.losses`

- Custom loss functions including binary and categorical focal loss, built as `tf.keras.losses.Loss` objects
so you can use them in a Keras model training pipeline as well.

**For Example**:

```python
from tensorflow.keras.optimizers import Adam
from deeptoolkit.losses import BinaryFocalLoss

# Using the model from the above example.
model.compile(
   optimizer = Adam(),
   loss = BinaryFocalLoss(),
   metrics = ['accuracy']
)
```

### Data Processing and Visualization: `deeptoolkit.data`

- Data preprocessing, including splitting data into train, validation, and test sets, and 
shuffling datasets while keeping data-label mappings intact.
- Data visualization, including cluster visualizations. 

**For Example:**

```python
import numpy as np
from deeptoolkit.data import train_val_test_split

X = np.random.random(100)
y = np.random.random(100)
X_train, X_val, X_test, y_train, y_val, y_test = train_val_test_split(X, y, split = [0.6, 0.2, 0.2])
```

### Model Evaluation: `deeptoolkit.evaluation`

- Model evaluation resources, including visualization of model training metrics over time.

### Computer Vision: `deeptoolkit.vision`

 - A pre-built facial detection model: `deeptoolkit.vision.FacialDetector`. A large number of modern 
 computer vision algorithms include a facial detection component, and DeepToolKit's facial detection module
 provides fast and accurate face detection using OpenCV's DNN implementation. To use it, simply execute the 
 following: 
 
 ```python
import cv2
from deeptoolkit.vision import FacialDetector

# Initialize detector.
detector = FacialDetector()

# Detect face from image path and save image to path.
detector.detect_face('image/path', save = 'image/save/path')

# Detect face from existing image and continue to use it.
image = cv2.imread('image/path')
annotated_image = detector.detect_face(image)
```

![Facial Detection Cartoon](examples/vision-example-image.png)

## License

All code in this repository is licensed under the [MIT License](https://github.com/amogh7joshi/deeptoolkit/blob/master/LICENSE).

## Issue Reporting 

If you notice any issues or bugs in the library, please create an issue under the issues tab. To get started 
and for more information, see the [issue templates](https://github.com/amogh7joshi/deeptoolkit/tree/master/.github/ISSUE_TEMPLATE).



