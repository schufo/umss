import torch

# ------------------------------------------------------------------------------
# Base model class
# ------------------------------------------------------------------------------


class _Model(torch.nn.Module):

    """
    Base class for all models
    """

    def __init__(self):
        super(_Model, self).__init__()

    @classmethod
    def from_config(cls, config: dict):
        """ All models should have this class method """
        raise NotImplementedError

# ------------------------------------------------------------------------------
# Helper class to load models more easily
# ------------------------------------------------------------------------------


class ModelLoader:
    """
    Helper class used to instantiate the desired model from a string
    """

    @staticmethod
    def get_model(model_name):
        """
        Returns an instance of the desired model

        :param model_name: the model name as a string (insensitive to snake_case
        or UpperCamelCase)
        """
        resolved_name = ModelLoader._resolve_name(model_name)
        for subclass in _Model.__subclasses__():
            if subclass.__name__ == resolved_name:
                return subclass
        raise NameError(
            "The model name {} was not found.".format(resolved_name))

    @staticmethod
    def _resolve_name(name: str):
        """
        Converts all snake_case names to UpperCamelCase and does nothing if name
        is already in the right case
        """
        components = name.split('_')
        if len(components) > 1:
            return ''.join(x.title() for x in components)
        return name