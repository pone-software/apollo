import logging
import os
import pickle
from abc import abstractmethod, ABC
from typing import TypeVar, Type, List, Any, Union


def create_folder(path):
    """
    Creates a folder if it does not exist

    Args:
        path: Pathlike parameter explaining which path should be created as a folder

    Returns:
        Nothing

    """
    is_exist = os.path.exists(path)

    if is_exist:
        logging.warning('Folder %s already exists', path)
    else:
        os.makedirs(path)


JSONSerializableType = TypeVar('JSONSerializableType', bound='JSONSerializable')


class JSONSerializable(ABC):
    """
    Interface to make a class serializable to and from json
    """

    @classmethod
    @abstractmethod
    def from_json(cls, json: Union[dict, list]) -> Type[JSONSerializableType]:
        """
        Load a class from a json like data type

        Args:
            json: json to be parsed

        Returns:
            Object of Class

        """
        raise NotImplementedError('from dict not implemented')

    @abstractmethod
    def as_json(self) -> Union[dict, list]:
        """
        converts the class into a json writable and readable data structure

        Returns:
            List or dict containing all the serialized information of the class

        """
        raise NotImplementedError('as dict not implemented')


FolderSavableType = TypeVar('FolderSavableType', bound='FolderSavable')


class FolderSavable(ABC):
    """
    Tool class that enables enumerable class to be saved to a folder using pickle
    """

    @abstractmethod
    def __len__(self):
        """
        Returns the length of the class values

        Returns:
            Length of the Class Values

        """
        raise NotImplementedError('Method __len__ not implemented')

    @abstractmethod
    def __getitem__(self, key) -> Type[FolderSavableType]:
        """
        Returns an object containing only the selected items of the enumerable object

        Args:
            key: Slice or integer of item to select

        Returns:
            Object with the selected items

        """
        raise NotImplementedError('Method __getitem__ not implemented')

    def to_folder(self, path, batch_size: int = 100, filename: str = 'part_{index}.pickle'):
        """
        Saves object batched to the passed folder

        Args:
            path: path where the object should be saved
            batch_size: number of sub-objects in each batch
            filename: how the individual batch files should be named. Must contain '{index}'.

        Returns:

        """
        logging.info('Starting to pickle to folder %s with batch_size %s', path, batch_size)

        is_exist = os.path.exists(path)

        if is_exist:
            logging.warning('Folder %s already exists', path)
        else:
            os.makedirs(path)

        number_of_items = len(self)

        if not number_of_items:
            logging.warning('Just pickled an empty object.')

        loop_index = start_index = 0

        while start_index <= number_of_items:
            current_object = self[start_index:start_index + batch_size]
            current_object.to_pickle(os.path.join(path, filename.format(index=loop_index)))
            loop_index += 1
            start_index = loop_index * batch_size

    def to_pickle(self, filename):
        """
        Saves object to a single pickle file

        Args:
            filename: which filename should be created

        Returns:

        """
        file = open(filename, 'wb')
        pickle.dump(self, file)
        file.close()


FolderLoadableType = TypeVar('FolderLoadableType', bound='FolderLoadable')


class FolderLoadable(ABC):
    """
    Tool class that enables enumerable object to be loaded from a folder
    """

    @classmethod
    def from_pickles(cls: Type[FolderLoadableType], filenames: List[str], **kwargs) -> Type[FolderLoadableType]:
        """
        Loads object from list of filenames.

        Args:
            filenames: List of filenames to be loaded
            **kwargs: arguments to be passed down to the object

        Returns:
            Enumerable object containing the information of all passed files

        """
        if len(filenames) == 0:
            logging.warning('Imported empty objects')
            return cls(**kwargs)

        final_result = None
        for filename in filenames:
            with open(filename, 'rb') as f:
                result = pickle.load(f)

            if not isinstance(result, cls):
                result = cls._load_result(result, **kwargs)

            if final_result is None:
                final_result = result
            else:
                final_result = final_result + result

            logging.info('File %s loaded', f)

        return final_result

    @classmethod
    def _load_result(cls: Type[FolderLoadableType], result: Any, **kwargs) -> Type[FolderLoadableType]:
        """
        helper function to enable subclasses to define an individual way of handling non-class imports via pickle

        Args:
            result: data of the pickle file import
            **kwargs: arguments to be used in this function

        Returns:
            object of type of the class

        """
        raise ValueError('Method load_result not implemented and type not matching')

    @abstractmethod
    def __add__(self, other: Type[FolderLoadableType]) -> Type[FolderLoadableType]:
        """
        Adds two enumerable objects together and creates a third one. All is done by reference

        Args:
            other: Object to be added to initial one

        Returns:
            new object containing data from both previous objects

        """
        raise NotImplementedError('Method concat not implemented')

    @classmethod
    def from_folder(cls: Type[FolderLoadableType], folder, **kwargs) -> Type[FolderLoadableType]:
        """
        Loads pickles from all pickle files in one folder

        Args:
            folder: folder to load pickles from
            **kwargs: arguments to be passed down during creation

        Returns:
            Object of the class to be loaded

        """
        filenames = []
        logging.info('Start to load folder %s', folder)
        for file in os.listdir(folder):
            if file.endswith(".pickle"):
                filenames.append(os.path.join(folder, file))

        imported_object = cls.from_pickles(filenames, **kwargs)
        logging.info('Finish to load folder %s', folder)

        return imported_object

    @classmethod
    def from_pickle(cls: Type[FolderLoadableType], filename, **kwargs) -> FolderLoadableType:
        """
        Imports an individual pickle file

        Args:
            filename: file to be loaded
            **kwargs: arguments to be passed down during creation

        Returns:
            Object of the class to be loaded

        """
        logging.info('Start to load file %s', filename)

        imported_object = cls.from_pickles([filename], **kwargs)
        logging.info('Finish to load file %s', filename)
        return imported_object
