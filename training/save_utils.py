import os

class SaveDirs:
    base_path = 'artifacts'
    
    @classmethod
    def add_dir(cls, name, include_base_path=False):
        if include_base_path: 
            name = f'{cls.base_path}/{name}'
        os.makedirs(name, exist_ok = True)
        
    @classmethod
    def create_dirs(cls, dirs):
        '''
        Create directories from a list of directory names
        '''
        for dir in dirs:
            cls.add_dir(dir, include_base_path=True)