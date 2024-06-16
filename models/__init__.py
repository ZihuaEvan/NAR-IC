from models.NAIC import NAIC
from models.Trans import Trans
__factory = {
    'NAIC': NAIC,
    'Trans': Trans,
}

def names():
    return sorted(__factory.keys())

def create(name, *args, **kwargs):
    if name not in __factory:
        raise KeyError("Unknown caption model:", name)
    return __factory[name](*args, **kwargs)