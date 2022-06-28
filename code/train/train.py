# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
import os
from sklearn.model_selection import train_test_split
from transformers import T5Tokenizer, T5ForConditionalGeneration, PreTrainedTokenizer
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from pytorch_lightning.callbacks.progress import TQDMProgressBar
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from os import environ
import logging

FORMAT = "%(asctime)s:%(name)s:%(levelname)s - %(message)s"
# Use filename="file.log" as a param to logging to log to a file
logging.basicConfig(format=FORMAT, level=logging.INFO)

MAX_EPOCHS = 50
EARLY_STOPPING_PATIENCE_EPOCHS = 5

NUM_GPUS = environ["NUM_GPUS"]
DATA_SOURCE = environ["DATA_SOURCE"]
TRANSFORMERS_CACHE_PATH = environ["TRANSFORMERS_CACHE_PATH"]
OUTPUT_PATH = environ["OUTPUT_PATH"]
LIGHTNING_LOGS_PATH = environ["LIGHTNING_LOGS_PATH"]

T5_MODEL_NAME = 't5-small'


class PyTorchDataModule(Dataset):
    """  PyTorch Dataset class  """

    def __init__(
        self,
        data: pd.DataFrame,
        tokenizer: PreTrainedTokenizer,
        source_max_token_len: int = 512,
        target_max_token_len: int = 512,
    ):
        """
        initiates a PyTorch Dataset Module for input data
        Args:
            data (pd.DataFrame): input pandas dataframe. Dataframe must have 2 column --> "source_text" and "target_text"
            tokenizer (PreTrainedTokenizer): a PreTrainedTokenizer (T5Tokenizer, MT5Tokenizer, or ByT5Tokenizer)
            source_max_token_len (int, optional): max token length of source text. Defaults to 512.
            target_max_token_len (int, optional): max token length of target text. Defaults to 512.
        """
        self.tokenizer = tokenizer
        self.data = data
        self.source_max_token_len = source_max_token_len
        self.target_max_token_len = target_max_token_len

    def __len__(self):
        """ returns length of data """
        return len(self.data)

    def __getitem__(self, index: int):
        """ returns dictionary of input tensors to feed into T5/MT5 model"""

        data_row = self.data.iloc[index]
        source_text = data_row["source_text"]

        source_text_encoding = self.tokenizer(
            source_text,
            max_length=self.source_max_token_len,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            add_special_tokens=True,
            return_tensors="pt",
        )

        target_text_encoding = self.tokenizer(
            data_row["target_text"],
            max_length=self.target_max_token_len,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            add_special_tokens=True,
            return_tensors="pt",
        )

        labels = target_text_encoding["input_ids"]
        labels[
            labels == 0
        ] = -100  # to make sure we have correct labels for T5 text generation

        return dict(
            source_text_input_ids=source_text_encoding["input_ids"].flatten(),
            source_text_attention_mask=source_text_encoding["attention_mask"].flatten(),
            labels=labels.flatten(),
            labels_attention_mask=target_text_encoding["attention_mask"].flatten(),
        )


class LightningDataModule(pl.LightningDataModule):
    """ PyTorch Lightning data class """

    def __init__(
        self,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        tokenizer: PreTrainedTokenizer,
        batch_size: int = 4,
        source_max_token_len: int = 512,
        target_max_token_len: int = 512,
        num_workers: int = 2,
    ):
        """
        initiates a PyTorch Lightning Data Module
        Args:
            train_df (pd.DataFrame): training dataframe. Dataframe must contain 2 columns --> "source_text" & "target_text"
            test_df (pd.DataFrame): validation dataframe. Dataframe must contain 2 columns --> "source_text" & "target_text"
            tokenizer (PreTrainedTokenizer): PreTrainedTokenizer (T5Tokenizer, MT5Tokenizer, or ByT5Tokenizer)
            batch_size (int, optional): batch size. Defaults to 4.
            source_max_token_len (int, optional): max token length of source text. Defaults to 512.
            target_max_token_len (int, optional): max token length of target text. Defaults to 512.
        """
        super().__init__()

        self.train_df = train_df
        self.test_df = test_df
        self.batch_size = batch_size
        self.tokenizer = tokenizer
        self.source_max_token_len = source_max_token_len
        self.target_max_token_len = target_max_token_len
        self.num_workers = num_workers

    def setup(self, stage=None):
        self.train_dataset = PyTorchDataModule(
            self.train_df,
            self.tokenizer,
            self.source_max_token_len,
            self.target_max_token_len,
        )
        self.test_dataset = PyTorchDataModule(
            self.test_df,
            self.tokenizer,
            self.source_max_token_len,
            self.target_max_token_len,
        )

    def train_dataloader(self):
        """ training dataloader """
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )

    def test_dataloader(self):
        """ test dataloader """
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        """ validation dataloader """
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )


class LightningModel(pl.LightningModule):
    """ PyTorch Lightning Model class"""

    def __init__(
        self,
        tokenizer,
        model,
        outputdir: str = OUTPUT_PATH,
        save_only_last_epoch: bool = False,
    ):
        """
        initiates a PyTorch Lightning Model
        Args:
            tokenizer : T5/MT5/ByT5 tokenizer
            model : T5/MT5/ByT5 model
            outputdir (str, optional): output directory to save model checkpoints. Defaults to OUTPUT_PATH.
            save_only_last_epoch (bool, optional): If True, save just the last epoch else models are saved for every epoch
        """
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.outputdir = outputdir
        self.average_training_loss = None
        self.average_validation_loss = None
        self.save_only_last_epoch = save_only_last_epoch

    def forward(self, input_ids, attention_mask, decoder_attention_mask, labels=None):
        """ forward step """
        output = self.model(
            input_ids,
            attention_mask=attention_mask,
            labels=labels,
            decoder_attention_mask=decoder_attention_mask,
        )

        return output.loss, output.logits

    def training_step(self, batch, batch_size):
        """ training step """
        input_ids = batch["source_text_input_ids"]
        attention_mask = batch["source_text_attention_mask"]
        labels = batch["labels"]
        labels_attention_mask = batch["labels_attention_mask"]

        loss, outputs = self(
            input_ids=input_ids,
            attention_mask=attention_mask,
            decoder_attention_mask=labels_attention_mask,
            labels=labels,
        )

        self.log(
            "train_loss", loss, prog_bar=True, logger=True, on_epoch=True, on_step=True
        )
        return loss

    def validation_step(self, batch, batch_size):
        """ validation step """
        input_ids = batch["source_text_input_ids"]
        attention_mask = batch["source_text_attention_mask"]
        labels = batch["labels"]
        labels_attention_mask = batch["labels_attention_mask"]

        loss, outputs = self(
            input_ids=input_ids,
            attention_mask=attention_mask,
            decoder_attention_mask=labels_attention_mask,
            labels=labels,
        )

        self.log(
            "val_loss", loss, prog_bar=True, logger=True, on_epoch=True, on_step=True
        )
        return loss

    def test_step(self, batch, batch_size):
        """ test step """
        input_ids = batch["source_text_input_ids"]
        attention_mask = batch["source_text_attention_mask"]
        labels = batch["labels"]
        labels_attention_mask = batch["labels_attention_mask"]

        loss, outputs = self(
            input_ids=input_ids,
            attention_mask=attention_mask,
            decoder_attention_mask=labels_attention_mask,
            labels=labels,
        )

        self.log("test_loss", loss, prog_bar=True, logger=True)
        return loss

    def configure_optimizers(self):
        """ configure optimizers """
        return AdamW(self.parameters(), lr=0.0001)

    def training_epoch_end(self, training_step_outputs):
        """ save tokenizer and model on epoch end """
        self.average_training_loss = np.round(
            torch.mean(torch.stack([x["loss"] for x in training_step_outputs])).item(),
            4,
        )
        path = f"{self.outputdir}/model.ckpt"
        if self.save_only_last_epoch:
            if self.current_epoch == self.trainer.max_epochs - 1:
                self.tokenizer.save_pretrained(path)
                self.model.save_pretrained(path)
        else:
            self.tokenizer.save_pretrained(path)
            self.model.save_pretrained(path)

    def validation_epoch_end(self, validation_step_outputs):
        _loss = [x.cpu() for x in validation_step_outputs]
        self.average_validation_loss = np.round(
            torch.mean(torch.stack(_loss)).item(),
            4,
        ) 



class TrainSKInterface:
    def __init__(self) -> None:
        self.df_attr_brandname = None
        self.df_attr_material = None
        self.df_attr_train = None
        self.df_attr_test = None
        self.df_val = None
        self.files_path = DATA_SOURCE
        
        try:
            if environ["NUM_GPUS"] != '0':
                self.use_gpu = True
            else:
                self.use_gpu = False
        except KeyError as e:
            self.use_gpu = False
    
    def read_dataset(self) -> None:    
        f_train_test = self.files_path + '/' + 'products_train_test.csv'
        #df = pd.read_csv(f_train_test, sep=';', names=['PRODUCT', 'ATTR_NAME', 'ATTR_VAL'])
        df = pd.read_csv(f_train_test, sep=';')

        df_attr_brandname = df[df['ATTR_NAME'] == 'Brand Name']
        df_attr_brandname = df_attr_brandname.drop(['ATTR_NAME'], axis=1)
        df_attr_brandname = df_attr_brandname.rename(columns={"PRODUCT": "source_text", "ATTR_VAL": "target_text"})
        df_attr_brandname = df_attr_brandname.astype({"source_text": "string", "target_text": "string"})
        df_attr_brandname = df_attr_brandname.dropna()
        df_attr_brandname['source_text'] = "recommend brand name: " + df_attr_brandname['source_text']
        
        self.df_attr_brandname = df_attr_brandname

        df_attr_material = df[df['ATTR_NAME'] == 'Material']
        df_attr_material = df_attr_material.drop(['ATTR_NAME'], axis=1)
        df_attr_material = df_attr_material.rename(columns={"PRODUCT": "source_text", "ATTR_VAL": "target_text"})
        df_attr_material = df_attr_material.astype({"source_text": "string", "target_text": "string"})
        df_attr_material = df_attr_material.dropna()
        df_attr_material['source_text'] = "recommend material: " + df_attr_material['source_text']

        self.df_attr_material = df_attr_material
     
        return None


    def split_dataset(self) -> None:
        df_attr_brandname_train, df_attr_brandname_test = train_test_split(self.df_attr_brandname, test_size=0.2)
        df_attr_material_train, df_attr_material_test = train_test_split(self.df_attr_material, test_size=0.2)
   
        self.df_attr_train = pd.concat([df_attr_brandname_train, df_attr_material_train])
        self.df_attr_test = pd.concat([df_attr_brandname_test, df_attr_material_test])

        return None


    def train_model(self) -> None:
        logging.info('Init tokenizer')    
        tokenizer = T5Tokenizer.from_pretrained(T5_MODEL_NAME, cache_dir=TRANSFORMERS_CACHE_PATH)

        model = T5ForConditionalGeneration.from_pretrained(
            T5_MODEL_NAME, return_dict=True, cache_dir=TRANSFORMERS_CACHE_PATH
        )

        logging.info('Init LightningDataModule')
        datamodule = LightningDataModule(
            self.df_attr_train,
            self.df_attr_test,
            tokenizer,
            batch_size=32,
            source_max_token_len=64,
            target_max_token_len=8,
            num_workers=4,
        )

        logging.info('Init LightningModel')
        T5Model = LightningModel(
            tokenizer=tokenizer,
            model=model,
            outputdir=OUTPUT_PATH,
            save_only_last_epoch=False,
        )

        logging.info('Init callbacks')
        callbacks = [TQDMProgressBar(refresh_rate=5)]

        early_stop_callback = EarlyStopping(
            monitor="val_loss",
            min_delta=0.00,
            patience=EARLY_STOPPING_PATIENCE_EPOCHS,
            verbose=True,
            mode="min",
        )
        callbacks.append(early_stop_callback)
        
        logging.info('Init trainer')
        trainer = pl.Trainer(
            logger=True,
            default_root_dir=LIGHTNING_LOGS_PATH,
            callbacks=callbacks,
            max_epochs=MAX_EPOCHS,
            gpus=NUM_GPUS,
            precision=32,
            log_every_n_steps=1,
            #num_sanity_val_steps=0,     # to prevent hangup on Windows
        )

        logging.info('Starting model fit')
        trainer.fit(T5Model, datamodule)

        return None


    def run_workflow(self) -> None:
        self.read_dataset()
        self.split_dataset()
        self.train_model()

        return None


if __name__ == "__main__":
    train_obj = TrainSKInterface()
    train_obj.run_workflow()
