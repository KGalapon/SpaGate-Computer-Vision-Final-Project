import pandas as pd
from pathlib import Path


class logger(): #writes one row per epoch to results/<folder>/train_log.csv

    def __init__(self, folder_name, results_dir):

        self.folder_path = Path(folder_name) #name of this run's folder
        self.results_dir = results_dir #the folder where all runs are stored
        self.log_path = self.results_dir / self.folder_path #this run's folder
        self.log_path.mkdir(parents=True, exist_ok=True)

        self.state_df = pd.DataFrame(
            {
                'epoch': [],
                'train_loss': [],
                's/epoch': [],
                'lr': [],
                'batch_size': [],
                'vram_use': [],
                'ram_use': []
            }
        ) #one row is added every epoch, there is no validation set so no val columns

    def save(self):
        self.state_df.to_csv(self.log_path / Path('train_log.csv'))

    def update(self, update_dict):
        update_df = pd.DataFrame(update_dict) #one row, every value in the dict is a one-item list

        if len(self.state_df) == 0:
            self.state_df = update_df #first epoch, nothing to join to yet
        else:
            self.state_df = pd.concat([self.state_df, update_df], axis=0, ignore_index=True) #ignore_index so the row numbers count up
        self.save()

    def save_test(self, test_metrics):
        #test_metrics is a dataframe with one row per test split
        test_metrics.to_csv(self.log_path / Path('test_metrics.csv'), index=False)