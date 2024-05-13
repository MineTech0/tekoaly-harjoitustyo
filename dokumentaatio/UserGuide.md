# User guide

## Setup

### Installation

Move to the project directory:

```bash
cd src
```

To install the required dependencies, run the following command:

```bash
poetry install
```

## Training the models

To train the model you have to create `data` directory in the `src` directory. The `data` directory should contain two subdirectories: `training` and `validation`. Each of these subdirectories should contain further folders named after the dance styles they represent, which serve as labels for the songs. All songs should be stored in `.mp3` format.

To preprocess the data, run the following command:

```bash
poetry run invoke preprocess
```

This will convert the songs to mel-spectrograms and save them in `.npz` format for efficiency and ease of access during model training.

The already preprocessed data is not included in the repository, so you have to preprocess the data before training the models.

### Peer model

The peer model is trained in the `peer_model` in file `peer_model.ipnyb` run the cells to train the model. Select the correct kernel to run the cells use the installed poetry environment.

### Own model

The own model is trained in the `own_model` directory. The training script is `training.py`. To train the model, run the following command:

```bash
poetry run invoke train
```

You can also use the `own_model.ipynb` to train the model.

## Running the web application

To start the web application, run the following command:

```bash
poetry run invoke start
```

Once the application is running, you can access it at `http://localhost:5000`. You can upload a song, and the application will predict the associated dance style. 

If you want to use your own trained model, you have to replace the `own_model.pkl` file and peer model `peer_model.keras` file in the `web-app/services/models` directory.

**The load method creates 18GB pickle file so it's not included in repo. You can download the zipped pickle file from [here](https://helsinkifi-my.sharepoint.com/:u:/g/personal/niilokur_ad_helsinki_fi/EVfZm8Qs1ExDmbV47-hwz9oBd5UILtEQJIDiWKMsfOpTGw?e=JCIgcS)**
