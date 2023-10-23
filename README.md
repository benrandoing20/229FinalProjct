# 229FinalProject
An investigation of ML and Generative AI techniques to diagnose AD severity

## Project Introduction 
Alzheimer's Disease (AD) is a neurodegenerative disease that may be 
categorized into 4 severity classes. 


## Datasets
The image [dataset](https://www.kaggle.com/datasets/uraninjo/augmented-alzheimer-mri-dataset/data)
consists of MRI images of brains clinically diagnosed into one of four
categories of AD severity. The data has four classes:

1. Mild Demented
2. Moderate Demented
3. Non Demented
4. Very Mild Demented 


## Project Setup

1. Download the Dataset above and unzip the directory.

2. Install [Miniconda](https://docs.conda.io/en/latest/miniconda.html)
  - Conda is a package manager that sandboxes your project’s dependencies in a virtual environment
  - Miniconda contains Conda and its dependencies with no extra packages by default (as opposed to Anaconda, which installs some extra packages)
2. Extract the zip file and run `conda env create -f environment.yml` from inside the extracted directory.
  - This creates a Conda environment called `229Final`
3. Run `source activate 229Final`
  - This activates the `229Final` environment
  - Do this each time you want to write/test your code
4. (Optional) If you use PyCharm:
  - Open the `src` directory in PyCharm
  - Go to `PyCharm` > `Preferences` > `Project` > `Project interpreter`
  - Click the gear in the top-right corner, then `Add`
  - Select `Conda environment` > `Existing environment` > Button on the right with `…`
  - Select `/Users/YOUR_USERNAME/miniconda3/envs/Final229/bin/python`
  - Select `OK` then `Apply`
5. If modifications are made to the `environment.yml` file, run `conda env update --file environment.yml`
to update the conda environment.


## MIT License
Copyright (c) [2023] [Benjamin Alexander Randoing & Elsa Bismuth & 
Prathibha Alam]

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.


