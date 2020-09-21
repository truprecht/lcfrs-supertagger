from collections import namedtuple
from warnings import warn

class MissingParameterWarning(Warning):
    pass
class UnknownParameterWarning(Warning):
    pass
class MissingRequiredParameter(Exception):
    def __init__(self, mplist):
        super(MissingRequiredParameter, self).__init__(f"Missing parameter(s): {', '.join(mp)}.")

class Parameters:
    def __init__(self, **params):
        for _, (ptype, defvalue) in params.items():
            assert type(ptype) is type
            assert type(defvalue) is ptype or defvalue is None
        self._params = params
        self._param_t = namedtuple(f"Parameters", " ".join(params.keys()))

    @classmethod
    def merge(cls, param1, param2, *params):
        p = { **param1._params, **param2._params }
        for paramx in params:
            p = { **p, **paramx._params }
        return cls(**p)

    def default(self):
        defvalues = (v for _, v in self._params.values())
        return self._param_t(*defvalues)

    def __call__(self, **kv):
        unknown_params = [param for param in kv if not param in self._params]
        if unknown_params:
            warn(
                UnknownParameterWarning(f'Ignoring values for unknow parameters: {", ".join(unknown_params)}.'),
                stacklevel=2)

        missing_params = [param for param in self._params if not param in kv]
        missing_required_params = [mp for mp in missing_params if self._params[mp][1] is None]

        if missing_required_params:
            raise MissingRequiredParameter(missing_required_params)

        if missing_params:
            warn(
                MissingParameterWarning(f'Using default values for missing parameters: {", ".join(missing_params)}.'),
                stacklevel=2)

        values = ((vtype(kv[param]) if param in kv else defval) \
                    for param, (vtype, defval) in self._params.items())
        return self._param_t(*values)