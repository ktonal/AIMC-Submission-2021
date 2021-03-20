import os
import pandas as pd
import re
from concurrent.futures import ThreadPoolExecutor

from mimikit.models.freqnet import FreqNet
from mimikit.models.sample_rnn import SampleRNN

from mimikit import NeptuneConnector

NC = NeptuneConnector(user="k-tonal",
                      setup={"project": "experiment-Stimme"})


def get_experiments_by_id():
    proj = NC.get_project("project")
    exps = proj.get_experiments()
    return {e.id: e for e in exps}


def _download_states(nc, eid, destination):
    try:
        nc.download_experiment(eid, destination=destination, artifacts="states")
    except Exception as e:
        print("no states found in", eid, e)


def download_states(destination, *exp_ids):
    """
    downloads the states of the experiment ids passed as *args

    Parameters
    ----------
    destination : where to download the states
    exp_ids: ids of the experiment to download the states from

    Examples
    --------
    # when in a colab notebook
    >>> download_states("/content/", "EXS-24", "EXS-28")
    """
    nc = NeptuneConnector(user="k-tonal", setup={})
    args = []
    for eid in exp_ids:
        nc.setup[eid] = "experiment-Stimme/" + eid
        args += [(nc, eid, destination)]
    with ThreadPoolExecutor(max_workers=len(args)) as executor:
        executor.map(_download_states, *zip(*args))


def list_downloaded_checkpoints(root_dir):
    epochs = {}
    for path in os.listdir(root_dir):
        if "EXS" in path and os.path.isdir(os.path.join(root_dir, path)):
            epochs[path] = FreqNet.list_available_epochs(os.path.join(root_dir, path))

    return [(eid, ep) for eid, eps in epochs.items() for ep in eps]


def load_exp_id_at_epochs(root_dir, *ids_and_epochs):
    """
    loads checkpoints located in `root_dir` for the experiment ids and epochs passed as *args.
    note that the dbs for the models are built if they don't exist yet in root dir and are name exp-id.h5.

    Parameters
    ----------
    root_dir: where the states are located
    ids_and_epochs: tuple of (str, int) e.g. ("EXS-21", 25)

    Returns
    -------
    list of loaded models

    Examples
    --------
    # when in a colab notebook
    >>> models = load_exp_id_at_epochs("/content/", ("EXS-23", 25), ("EXS-23", 50))
    """
    model_classes = dict(SampleRNN=SampleRNN, FreqNet=FreqNet)
    models = []
    for eid, ep in ids_and_epochs:
        ckpt_path = os.path.join(root_dir, eid, "states", "epoch=%i.ckpt" % ep)
        hparams = FreqNet.load_checkpoint_hparams(ckpt_path)
        cls, exp_files = model_classes[hparams["model_class"]], hparams["files"]
        db_path = os.path.join(root_dir, eid + ".h5")
        if cls is FreqNet:
            if not os.path.exists(db_path):
                db = cls.db_class.make(db_path, files=["./data/" + file for file in exp_files],
                                       n_fft=hparams["n_fft"], hop_length=hparams["hop_length"],
                                       sr=hparams.get("sr", 22050))
            else:
                db = cls.db_class(db_path)
            models += [FreqNet.load_from_checkpoint(os.path.join(root_dir, eid, "states", "epoch=%i.ckpt" % ep), db=db)]
        elif cls is SampleRNN:
            if not os.path.exists(db_path):
                db = cls.db_class.make(db_path, files=["./data/" + file for file in exp_files],
                                       q_levels=hparams["q_levels"],
                                       sr=hparams.get("sr", 16000),
                                       emphasis=hparams.get("emphasis", 0.))
            else:
                db = cls.db_class(db_path)
            models += [SampleRNN.load_from_checkpoint(os.path.join(root_dir, eid, "states", "epoch=%i.ckpt" % ep), db=db)]

    return models


def get_project_summary(columns={"id", "loss", "model_class", "files", "running_time", "created"}):
    project = NC.get_project("project")
    df = project.get_leaderboard()
    df = df.rename(columns={name: re.sub(r"(channel_|parameter_)", "", name) for name in df.columns})
    df = df.drop(columns=[col for col in df.columns if col not in columns])
    if "running_time" in columns:
        df["running_time"] = pd.to_datetime(df.running_time, unit='s').dt.strftime("%Hh-%Mm-%Ss")
    if "created" in columns:
        df["created"] = df.created.dt.strftime('%B %d, %Y')
    df = df.where(pd.notnull(df), "-")

    return df