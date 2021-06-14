Title: DVC - Data Versioning
Date: 2021-06-14 15:18
Category: Tech
Tags: Data, Version Control, DVC
Slug: dvc-data-versioning
Authors: Lee-W
Series: Data Version Control Tutorial

<!--more-->

[TOC]

## About DVC (Data Version Control)
* What's DVC?
    * version control system for data science and machine learning
    * compatible with git (it's based on git)
* What can DVC do?
    * track
        * data
        * model
        * pipeline
        * metrics
    * use storage directly
    * no external services needed
* Who are the targeted users of DVC?
    * ML research / engineer
    * DevOps & Engineers
* Why DVC?
    * It links your data, model, and pipelines with your metrics.
        * reproducibility
        * trackable

Read [DVC - Versioning Data and Models](https://dvc.org/doc/use-cases/versioning-data-and-model-files) for more use cases

## How to use DVC?

### Install DVC globally
I suggest using [pipx](https://pypa.github.io/pipx/) if you're to install DVC globally. However, an even better way is to install it inside the virtual environment within your project.

```sh
$ pip install pipx
$ pipx install dvc
$ dvc --version

2.3.0
```

DVC also provides [Shell Completion](https://dvc.org/doc/install/completion) and [Syntax Highlighting Plugins](https://dvc.org/doc/install/plugins) for popular editors.

### Take a look at the example project

I'll use [dvc_example](https://github.com/Lee-W/dvc_example/) to demonstrate how I applied DVC to an existing machine learning project. The example is based on [Recognizing hand-written digits](https://scikit-learn.org/0.24/auto_examples/classification/plot_digits_classification.html) from scikit-learn documentation. All the DVC parts start from [v1-base](https://github.com/Lee-W/dvc_example/tree/v1-base). You can `git checkout` to the tag to follow along.

```sh
$ git clone https://github.com/Lee-W/dvc_example/ --branch v1-base
$ cd dvc_example
$ tree
.
├── LICENSE
├── Pipfile
├── Pipfile.lock
├── digit_recognizer
│   ├── __init__.py
│   └── digit_recognizer.py
├── docs
│   └── README.md
├── mkdocs.yml
├── output
└── tasks.py
```

To set up the development environment, you'll need [pipenv](https://pipenv.pypa.io/en/latest/) and [invoke](https://www.pyinvoke.org/). If you run into an error when running `pipenv install`, you can run `export SYSTEM_VERSION_COMPAT=1` before it. It's an open issue ([Issue with NumPy, macOS 11 Big Sur, Python 3.9.1 Does pipenv not use the latest pip? #4564](https://github.com/pypa/pipenv/issues/4564#issuecomment-756625303)) of pipenv as of now. Or, you can just run the following commands.

```sh
# install needed tools
pipx install pipenv invoke

# set up environments
invoke init-dev
```

We'll use [digit_recognizer/digit_recognizer.py](https://github.com/Lee-W/dvc_example/blob/v1-base/digit_recognizer/digit_recognizer.py) for training a model that can recognize handwritten digits.

```python
def main():
    X, y = load_data()
    X_train, X_test, y_train, y_test = process_data(X, y)
    model = train_model(X_train, y_train)
    predicted_y = model.predict(X_test)
    output_results(y_test, predicted_y)
    output_metrics(y_test, predicted_y)
```

### Install DVC into the virtual environment

```sh
pipenv install dvc
```

If you're to save data to remote storage, you might need to install extra dependencies.
(e.g., `pipenv install dvc[s3]`)

* Supported types
    * `[s3]`
    * `[azure]`
    * `[gdrive]`
    * `[gs]`
    * `[oss]`
    * `[ssh]`

Or, use `pipenv install dvc[all]` to install them all

Read [dvc remote](https://dvc.org/doc/command-reference/remote) for more information

### Initialize DVC

```sh
# initialize DVC configurations
$ pipenv run dvc init

# see what's created by DVC
$ tree .dvc

.dvc
├── config
└── plots
    ├── confusion.json
    ├── confusion_normalized.json
    ├── default.json
    ├── linear.json
    ├── scatter.json
    └── smooth.json

# track DVC configuration through git
$ git add .dvc

# git commit
$ pipenv run cz commit
```

### Add DVC remote
I'll use another local directory `../dvc_remote` as our remote storage. You can change it to s3 or other remote storage.

```sh
mkdir ../dvc_remote
dvc remote add --default local ../dvc_remote
```

Through `--default` flag, we can push/pull from `local` remote without specifying remote name.

Let see what's changed in `.dvc/config`.

```cfg
$ cat .dvc/config

[core]
remote = local
['remote "local"']
url = ../../dvc_remote
```

The url is `../../dvc_remote` instead of `../dvc_remote` because it's the relative path to `.dvc`. As we've not yet push anything to our pseudo remote, `../dvc_remote` is still empty.

### Track data through DVC

By this time, the data is loaded through [sklearn.datasets.load_digits](https://scikit-learn.org/0.24/modules/generated/sklearn.datasets.load_digits.html). We're going to change it to read from static file in `data/`.

```python
def load_data():
    # Load data
    digits = datasets.load_digits()
    ...
```

We can use the following script to output the digit data into `data/`. Note that it's a one-time use script. We won't add it into git.

```python
import os

import pandas as pd
from sklearn import datasets

os.mkdir("data")

digits = datasets.load_digits()

df = pd.DataFrame(digits.data)
df.to_csv("data/digit_data.csv", header=False, index=False)

df = pd.DataFrame(digits.target)
df.to_csv("data/digit_target.csv", header=False, index=False)
```

We'll need to make changes to `load_data` and `main` functions to read data from these files.

```python
def load_data(X_path, y_path):
    with open(X_path) as input_file:
        csv_reader = csv.reader(input_file, quoting=csv.QUOTE_NONNUMERIC)
        X = list(csv_reader)

    with open(y_path) as input_file:
        csv_reader = csv.reader(input_file, quoting=csv.QUOTE_NONNUMERIC)
        y = [row[0] for row in csv_reader]

    return X, y

......

def main():
    X, y = load_data("data/digit_data.csv", "data/digit_target.csv")
    ......
```

Run `pipenv run python digit_recognizer/digit_recognizer.py` to check whether everything works as we expected. If so, add these code changes into git.

Next, add `data/` to DVC.

```sh
$ pipenv run dvc add data

100% Add|████████████████|1/1 [00:00,  2.14file/s]

To track the changes with git, run:

git add data.dvc .gitignore
```

`dvc add` creates a `data.dvc` file to track `data/` and add it into `.gitignore` so that `data/` will only be tracked through DVC but not git.

```sh
# Add DVC files into git track
git add .gitignore data.dvc

# git commit
pipenv run cz commit
```

In `data.dvc`, we can see 2 files (`digit_data.csv` and `digit_target.csv`) are tracked.

```sh
$ cat data.dvc

outs:
- md5: b8d81f4964ecb86739c79c833fb491f3.dir
  size: 494728
  nfiles: 2
  path: data
```

Push these tracked data into DVC remote

```sh
dvc push
```

See what's changed in our repo storage `../dvc_remote`

```sh
$ tree ../dvc_remote

../dvc_remote
├── 02
│   └── b861b6dc8e08da6d66547860f69277
├── 8c
│   └── ba569595920d230ade453b150f372b
└── b8
    └── d81f4964ecb86739c79c833fb491f3.dir

3 directories, 3 files
```

The md5 value of our tracked data is `b8d81f4964ecb86739c79c833fb491f3.dir`. There's also a corresponding file in `../dvc_remote/b8/d81f4964ecb86739c79c833fb491f3.dir`.

```sh
$ cat ../dvc_remote/b8/d81f4964ecb86739c79c833fb491f3.dir

[{"md5": "02b861b6dc8e08da6d66547860f69277", "relpath": "digit_data.csv"}, {"md5": "8cba569595920d230ade453b150f372b", "relpath": "digit_target.csv"}]%
```

This file indicates where the actual data sources are stored in `../dvc_remote`.

In conclusion, if we want to know how data is stored through DVC,

1. find the md5 value in `*.dvc` in our project
2. find the path that matches this md5 value in our remote storage
3. use the md5 value specified in the previous step to find the data sources in our remote storage

But most of the time, we don't need to do so. We can leave the tracking work to DVC.

### Fetch data from DVC remote storage

```sh
# temporary delete our data locally
$ rm -rf data

# check whether DVC actually tracks our data
$ dvc status

data.dvc:
changed outs:
    deleted:            data

# bring our data back from remote storage
$ dvc checkout data

data
├── digit_data.csv
└── digit_target.csv
```

### Add data changes into DVC
To demonstrate how DVC track data changes, let's remove the last 2 rows from `data/digit_data.csv` and `data/digit_target.csv`.

```sh
# check what's changed
$ dvc status

data.dvc:
changed outs:
    modified:           data

# Add these changes to DVC and git
$ dvc add
$ git add data.dvc
# git commit
$ pipenv run cz commit

# Push these changes to our remote storage
$ dvc push
```

The md5 value has been changed, and the size of our data is smaller than our previous record, 494728.

```sh
$ cat data.dvc

outs:
- md5: a333e114a49194e823ab9a4fa9e33ee9.dir
  size: 494172
  nfiles: 2
  path: data
```

More files are added to `../dvc_remote` due to the data changes. You can follow the steps in the previous section to see what're actually store.

```sh
$ tree ../dvc_remote

../dvc_remote
├── 02
│   └── b861b6dc8e08da6d66547860f69277
├── 2a
│   └── 6cfa13365ac9b3af5146133aca6789
├── 8c
│   └── ba569595920d230ade453b150f372b
├── 94
│   └── 2481fce846fb9750b7b8023c80a5ef
├── a3
│   └── 33e114a49194e823ab9a4fa9e33ee9.dir
└── b8
    └── d81f4964ecb86739c79c833fb491f3.dir

6 directories, 6 files
```

Let's `git checkout` to the previous git commit to see what happens if we only revert the changes in `data.dvc`.

```sh
# or "git checkout v2-track-data"
git checkout HEAD~1
```

After running `wc -l data/digit_data.csv`, we'll still find 1795 rows instead of 1797 rows in the previous stage. That's because we need to run `dvc checkout` as well.

We might easily forget this step. Thus, DVC implements a git-hook that can trigger `dvc checkout` right after `git checkout`. You can install these git-hooks through `dvc install`. These hooks are added into `.git/hooks`. If you want to know the detail of what's added, read [dvc install](https://dvc.org/doc/command-reference/install).

Test these steps again. There should be an additional line after running `git checkout`. This is the output message of `dvc checkout`.

```text
M       data/
```

Push our code to a remote git repository

```sh
git remote add origin <REMOTE GIT REPO>
git push origin main
```

### Fetch code and data changes from remote
We've already pushed all the code and data changes to remote. Let's see how we could reproduce in another environment.

```sh
# check what's in our repo
$ dvc list <REMOTE GIT REPO>

.dvcignore
.github
.gitignore
LICENSE
Pipfile
Pipfile.lock
data
data.dvc
digit_recognizer
docs
mkdocs.yml
output
tasks.py
```

Although git does not track `data/`, we can still list it through DVC.

Because we use relative path `../dvc_remote` as DVC remote storage, we need to create the new project in the same layer as `dvc_example`. We'll clone the project into `../dvc_example_on_another_machine`.

```sh
# Clone repo git repo
$ git clone <YOUR REMOTE GIT REPO> ../dvc_example_on_another_machine
$ cd ../dvc_example_on_another_machine
$ tree .

.
├── LICENSE
├── Pipfile
├── Pipfile.lock
├── data.dvc
├── digit_recognizer
│   ├── __init__.py
│   └── digit_recognizer.py
├── docs
│   └── README.md
├── mkdocs.yml
├── output
└── tasks.py

3 directories, 9 files
```

As you can see, `data/` has not yet been added to the project. We can now pull data from our DVC remote storage.

```sh
# pull data from default DVC remote storage
$ dvc pull

A   data/
1 file added and 2 files fetched

# `data` has now been added to the project
$ tree .

.
├── LICENSE
├── Pipfile
├── Pipfile.lock
├── data
│   ├── digit_data.csv
│   └── digit_target.csv
├── data.dvc
├── digit_recognizer
│   ├── __init__.py
│   └── digit_recognizer.py
├── docs
│   └── README.md
├── mkdocs.yml
├── output
└── tasks.py

4 directories, 11 files
```

That's all for data versioning in DVC. In the next post, We'll continue on versioning a data pipeline, tracking parameters and metrics. We won't need `dvc_example_on_another_machine` for the following steps. Feel free to remove it and change directory back to `dvc_example`.

## Reference
* [DVC](https://dvc.org/)
* [CS 329S: Machine Learning Systems Design](https://stanford-cs329s.github.io/syllabus.html)
