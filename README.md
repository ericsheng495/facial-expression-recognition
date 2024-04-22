# Facial Emotion Recognition for Smart Homes Appliances

> **CS 4641 Group 16:** Shize Sheng, Yuanhong Zhou, Chunzhen Hu, Jiasheng Cao, Xingyu Hu

<!--
[![NPM Version][npm-image]][npm-url]
[![Build Status][travis-image]][travis-url]
[![Downloads Stats][npm-downloads]][npm-url]
 -->
 

![](header.png)
<!--

## Installation

OS X & Linux:

```sh
npm install my-crazy-module --save
```

Windows:

```sh
edit autoexec.bat
```

## Usage example

A few motivating and useful examples of how your product can be used. Spice this up with code blocks and potentially more screenshots.

_For more examples and usage, please refer to the [Wiki][wiki]._
 -->
## Repo Structure

```
.
├── CNN                          # CNN related files.
│   ├── cnn_train.ipynb          # Notebook for preprocessing, training, and evaluating the CNN model.
│   └── model       
│       └── trained_cnn_model.h5
│ 
├── Random Forests               # Random Forest related files.
│   ├── model
│   │   └── trained_cnn_model.h5
│   └── rf_train.ipynb
│ 
├── Naive_Bayes                 # Naive Bayes related files.
│   └── naive_bayes_train.ipynb
│ 
├── README.md                   
├── requirements.txt             # List of python packages required for the project.
├── docs                         # GitHub Pages.
└── setData                      # Dataset directory.
    ├── original_data_set       
    │   ├── test                 # Testing set
    │   │   ├── angry
    │   │   ├── disgust
    │   │   ├── fear
    │   │   ├── happy
    │   │   ├── neutral
    │   │   ├── sad
    │   │   └── surprise
    │   └── train                # Training set
    │       ├── angry
    │       ├── disgust
    │       ├── fear
    │       ├── happy
    │       ├── neutral
    │       ├── sad
    │       └── surprise
    └── processed_data_set       # Processed dataset, simplified into numpy arrays for direct use in models.
        ├── test                
        │   └── data.npy
        └── train                # Processed training data in a numpy array.
            └── data.npy


```
        
### Directories
- `/CNN/`: Contains files related to Convolutional Neural Networks model training
  - `cnn_train.ipynb`: Jupyter notebook for the CNN that performs data preprocessing, visualization, CNN training, model evaulation
  - `model/trained_cnn_model.h5`: A saved model file that contains the weights and architecture of the trained convolutional neural network, no retraining needed for new datasets
- `/Random Forests/`: Contains files related to Random Forest model training
- `/Naive Bayes/`: Contains files related to Naive Bayes model training
- `/setData/`: Contains datasets for the project. It includes:
  - **Original dataset (`/setData/original_data_set/`)**: Prepared for initial data analysis. It is organized into `test` and `train` folders, each containing subfolders for different emotions such as `angry`, `disgust`, `fear`, `happy`, `neutral`, `sad`, and `surprise`.
    - `/test/`: Contains images for testing the model, divided into categories based on emotions
    - `/train/`: Contains images for training the model, similarly divided into emotional categories.
  - **Preprocessed dataset (`/setData/processed_data_set/`)**: Contains pre-processed data
    - `/test/`: Contains a `data.npy` file with preprocessed testing data.
    - `/train/`: Contains a `data.npy` file with preprocessed training data.
- `/docs/`: Used for GitHub Pages hosting; includes markdown, HTML, and other resources needed to build the project's webpage.


### Files
- `/requirements.txt`: Specifies all Python dependencies required by the project, which can be installed via pip.
- `/.gitignore`: Lists all files and directories that git should ignore, preventing them from being tracked or added to the repository.
- `/README.md`: Provides an overview of the project, setup instructions, and essential information for users and contributors.

## Development

### Setup  

1. Clone this project to your computer

```sh
git clone https://github.com/ericsheng495/facial-expression-recognition.git
```

2. Navigate to this project in your terminal
```sh
cd facial-expression-recognition
```

3. Pulling the latest update from main
```sh
git pull 
```

4. Creating your branch
```sh
git checkout -b your-branch-name
```

5. Running the model
```sh

```

### Commit & Push 

6. See which files you have modified
```sh
git status
```

7. Add your files to "staging"
```sh
git add .
```

8. Commit with message
```sh
git commit -m "your commit message"
```

9. Push your branch to main (remote)
```sh
git push
```






## Jekyll Setup (github pages)

1. Navigate to docs directory (this is where we keep github pages themes)
```sh
cd docs
```
2. Run Locally (on localhost:4000)
```sh
bundle exec jekyll serve  
```

---
## Additional Notes:
- Git Branching
- Git Commit & Push



<!--
## Release History

- 0.2.1
  - CHANGE: Update docs (module code remains unchanged)
- 0.2.0
  - CHANGE: Remove `setDefaultXYZ()`
  - ADD: Add `init()`
- 0.1.1
  - FIX: Crash when calling `baz()` (Thanks @GenerousContributorName!)
- 0.1.0
  - The first proper release
  - CHANGE: Rename `foo()` to `bar()`
- 0.0.1
  - Work in progress
 -->
<!--
## Meta

Your Name – [@YourTwitter](https://twitter.com/dbader_org) – YourEmail@example.com

Distributed under the XYZ license. See `LICENSE` for more information.

[https://github.com/yourname/github-link](https://github.com/dbader/)
 -->
 <!--
## Contributing

1. Fork it (<https://github.com/yourname/yourproject/fork>)
2. Create your feature branch (`git checkout -b feature/fooBar`)
3. Commit your changes (`git commit -am 'Add some fooBar'`)
4. Push to the branch (`git push origin feature/fooBar`)
5. Create a new Pull Request
 -->

<!-- Markdown link & img dfn's -->

[npm-image]: https://img.shields.io/npm/v/datadog-metrics.svg?style=flat-square
[npm-url]: https://npmjs.org/package/datadog-metrics
[npm-downloads]: https://img.shields.io/npm/dm/datadog-metrics.svg?style=flat-square
[travis-image]: https://img.shields.io/travis/dbader/node-datadog-metrics/master.svg?style=flat-square
[travis-url]: https://travis-ci.org/dbader/node-datadog-metrics
[wiki]: https://github.com/yourname/yourproject/wiki
