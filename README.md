# HyperGraph Link Prediction

This repo contains codes for link prediction task for hypergraphs. 
Some codes reproduced from [this repo](https://github.com/granelle/www20-higher-order).


## Getting Started
### Prerequisites
_Python version_

![Python](https://img.shields.io/badge/Python-3.7.9-green.svg?style=plastic)

#### Clone Repo

```bash
    $ git clone https://github.com/alibtasdemir/Hypergraph-Tailed.git
    $ cd Hypergraph-Tailed
 ```

#### Datasets

Used dataset is under [this page](https://www.cs.cornell.edu/~arb/data/):

* **coauth-DBLP:** co-authorship on DBLP papers.
* **coauth-MAG-Geology:** co-authorship on Geology papers.
* **coauth-MAG-History:** co-authorship on History papers.
* **tags-math-sx:** sets of tags applied to questions on math.stackexchange.com.
* **tags-ask-ubuntu:** sets of tags applied to questions on askubuntu.com.
* **threads-math-sx:** sets of users asking and answering questions on threads on math.stackexchange.com.
* **threads-ask-ubuntu:** sets of users asking and answering questions on threads on askubuntu.com.
* **NDC-substances:** sets of substances making up drugs.
* **NDC-classes:** sets of classifications applied to drugs.
* **DAWN:** sets of drugs used by patients recorded in emergency room visits.
* **congress-bills:** sets of congresspersons cosponsoring bills.
* **email-Eu:** sets of email addresses on emails.
* **email-Enron:** sets of email addresses on emails.
* **contact-high-school:** groups of people in contact at a high school.
* **contact-primary-school:** groups of people in contact at a primary school.

Download and unzip datafiles in `[RAW_DATA_FOLDER]`

```
[RAW_DATA_FOLDER]
 ├── coauth-DBLP
 ├── coauth-MAG-Geology
 ├── coauth-MAG-History
 ├── ...
 └── contact-primary-school
```

#### 1. Preprocess data files

To preprocess data files use:
```
python run.py --task preproc --logdir [LOG_DIR] --rawdir [RAW_DATA_FOLDER] --outdir [PROC_DATA_FOLDER]
```

#### 2. Generate HyperGraph Objects and Save with Pickle

```
python run.py --task create --hgdir [HG_FOLDER] --graphdir [PROC_DATA_FOLDER] --graphname [GRAPH NAME] --hedgesize [HYPEREDGE SIZE] --sampling [NEG SAMPLING TYPE] --imb [IMBALANCE]
```

#### Options
* `--graphname`: The name of a graph to create pickle object. Pass `""`(empty string) to use all graphs inside `PROC_DATA_FOLDER`. Default: `email-Enron`
* `--hedgesize`: The target hyperedge size for link prediction. Choices are `[0, 4, 5]`. Use `0` to use both `4, 5` hyperedge sizes. Default: `0`
* `--sampling`: Negative sampling technique. Choices are `[clique, star, tailed]`. Use `""` (empty string) to use all techniques. Default: `""`
* `--imb`: Class imbalance ratio (`NEG/POS`). Choices are `[0, 1, 2, 5, 10]`. Use `0` to use all. Default: `0`


#### 3. Train/Test

To train/test the networks:
```
python run.py --task predict --hgdir [HG_FOLDER] --resultsdir [RESULTS_FOLDER] --graphname [GRAPH NAME] --hedgesize [HYPEREDGE SIZE] --sampling [NEG SAMPLING TYPE] --imb [IMBALANCE] --feature [FEATURE]
```

#### Options
* Check previous section for `--graphname` `--hedgesize` `--sampling` `--imb`
* `--feature`: Name of the feature that will be used for representation of nodes/hyperedges. Choices are `[gm, hm, am, cn, jc, aa]`. Use `""` (empty string) to use all. Default: `""`
  * `gm` : Geometric Mean - [Wikipedia](https://en.wikipedia.org/wiki/Geometric_mean)
  * `hm` : Harmonic Mean - [Wikipedia](https://en.wikipedia.org/wiki/Harmonic_mean)
  * `am` : Arithmetic Mean - [Wikipedia](https://en.wikipedia.org/wiki/Arithmetic_mean)
  * `cn` : Common Neighbors - [NetworkX](https://networkx.org/documentation/stable/reference/generated/networkx.classes.function.common_neighbors.html)
  * `jc` : Jaccard Coefficient - [Wikipedia](https://en.wikipedia.org/wiki/Jaccard_index)
  * `aa` : Adamic-Adar - [Wikipedia](https://en.wikipedia.org/wiki/Adamic%E2%80%93Adar_index)

## Contact
Open an issue for any inquiries.
You may also have contact with [alibaran@tasdemir.us](mailto:alibaran@tasdemir.us)