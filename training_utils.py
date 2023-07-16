import os

class SaveDirs:
    base_path = 'artifacts'
    
    @classmethod
    def add_dir(cls, new_dir_name):
        os.makedirs(f'{cls.base_path}/{new_dir_name}', exist_ok = True)
        
    @classmethod
    def create_train_dirs(cls):
        '''
        Call creation of directories for saving vocab, models and loss curves
        '''
        cls.add_dir('saved_vocab')
        cls.add_dir('saved_models')
        cls.add_dir('loss_curves')