import os
import shutil



class DictX(dict):
    """
    Taken From https://dev.to/0xbf/use-dot-syntax-to-access-dictionary-key-python-tips-10ec
    """

    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError as k:
            raise AttributeError(k)

    def __setattr__(self, key, value):
        self[key] = value

    def __delattr__(self, key):
        try:
            del self[key]
        except KeyError as k:
            raise AttributeError(k)

    def __repr__(self):
        return '<DictX ' + dict.__repr__(self) + '>'


class AssetManager:

	def __init__(self, base_dir):
		self.__base_dir = base_dir

		self.__preprocess_dir = os.path.join(self.__base_dir, 'preprocess')
		if not os.path.exists(self.__preprocess_dir):
			os.makedirs(self.__preprocess_dir, exist_ok=True)

		self.__models_dir = os.path.join(self.__base_dir, 'models')
		if not os.path.exists(self.__models_dir):
			os.mkdir(self.__models_dir)

		self.__tensorboard_dir = os.path.join(self.__base_dir, 'tensorboard')
		if not os.path.exists(self.__tensorboard_dir):
			os.mkdir(self.__tensorboard_dir)

	def get_preprocess_file_path(self, data_name):
		return os.path.join(self.__preprocess_dir, data_name + '.npz')

	def get_model_dir(self, model_name):
		return os.path.join(self.__models_dir, model_name)

	def recreate_model_dir(self, model_name, keep_prev=False):
		model_dir = self.get_model_dir(model_name)

		if keep_prev:
			self.__create_dir(model_dir)
		else:
			self.__recreate_dir(model_dir)
		return model_dir

	def get_tensorboard_dir(self, model_name):
		return os.path.join(self.__tensorboard_dir, model_name)

	def recreate_tensorboard_dir(self, model_name, keep_prev=False):
		tensorboard_dir = self.get_tensorboard_dir(model_name)

		if keep_prev:
			self.__create_dir(tensorboard_dir)
		else:
			self.__recreate_dir(tensorboard_dir)
		return tensorboard_dir

	@staticmethod
	def __recreate_dir(path):
		if os.path.exists(path):
			raise ValueError(f'{path} directory already exists!')
			# shutil.rmtree(path)

		os.makedirs(path)

	@staticmethod
	def __create_dir(path):
		os.makedirs(path, exist_ok=True)
