from .farm import FarmPlugin
from .mine import MinePlugin

_PLUGINS = [
    FarmPlugin(),
    MinePlugin(),
]

def get_plugin(scene_type: str):
    for plugin in _PLUGINS:
        if scene_type in plugin.scene_types():
            return plugin

    return FarmPlugin()