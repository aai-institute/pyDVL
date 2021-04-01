from accsr.config import ConfigProviderBase, DefaultDataConfiguration


class __Configuration(DefaultDataConfiguration):
    pass


class ConfigProvider(ConfigProviderBase[__Configuration]):
    pass


_config_provider = ConfigProvider()


def get_config(reload=False) -> __Configuration:
    """
    :param reload: if True, the configuration will be reloaded from the json files
    :return: the configuration instance
    """
    return _config_provider.get_config(reload=reload)
