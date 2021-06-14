Title: DVC - Run experiments
Date: 2021-06-16 12:57
Category: Tech
Tags: Data, Version Control, DVC
Slug: dvc-run-experiments
Authors: Lee-W
Series: Data Version Control Tutorial
Status: draft


### Run with different parameter - Experiment


```sh
pipenv run dvc exp run --set-param train.gamma=0.01
pipenv run dvc exp run --set-param train.gamma=0.001
pipenv run dvc exp run --set-param train.gamma=0.0001
```

```sh
$ dvc exp list

main:
exp-1cfca
exp-c2522
```


```sh
dvc exp show
```

![](media/16231470608474.jpg)


What are experiments? custom git ref with a single commit based on HEAD

try `git log --all` or `tig -all`

you'll see revs like `refs/exps/57/943d433dc86efe0eefd27ebbdad554a7e5f829/exp-c2522`

you can push or pull these experiments through `dvc exp push` and `dvc exp pull`

..

By default, dvc exp show only shows experiments since the last commit

it updates `dvc.lock` and `params.yaml`



now we can choose one
```sh
$ dvc exp apply [hash]
$ git add dvc.lock params.yaml prc.json roc.json scores.json
$ git commit -a -m "Preserve best random forest experiment"
```

```sh
dvc exp gc --workspace
```

### parallel
https://dvc.org/doc/command-reference/exp/run#queueing-and-parallel-execution

```sh
$ pipenv run dvc exp run --queue --set-param train.gamma=0.01
$ pipenv run dvc exp run --queue --set-param train.gamma=0.001
$ pipenv run dvc exp run --queue --set-param train.gamma=0.0001

Queued experiment '39df7af' for future execution.
Queued experiment '04edfbd' for future execution.
Queued experiment 'e86dd9b' for future execution.
```

```sh
 dvc exp run --run-all -jobs 4
```

it's still not a stable feature


One gocha is that only tracked files and directories will be included in --queue/temp experiments

### Other experiment feature
* checkpoint

> To track successive steps in a longer experiment, you can write your code so it registers checkpoints with DVC at runtime. This allows you, for example, to track the progress in deep learning techniques such as evolving neural networks.
