"""
A minimal stand-in for Streamlit so app.py can be executed in a test without
a browser. It records what the app renders and answers widgets with scripted values.
"""

from unittest.mock import MagicMock


class StopApp(Exception):
    pass


class SessionState(dict):
    __getattr__ = dict.get

    def __setattr__(self, key, value):
        self[key] = value


class FakeStreamlit:
    def __init__(self, selection_index=0, button=True, industry="All",
                 genres=(), how_many=10):
        self.session_state = SessionState()
        self.calls = []                 # (function, first argument) log
        self.script = dict(selection_index=selection_index, button=button,
                           industry=industry, genres=list(genres),
                           how_many=how_many)
        self.sidebar = self

    # -- decorators / control flow
    def cache_resource(self, *args, **kwargs):
        if args and callable(args[0]):
            return args[0]
        return lambda fn: fn

    def stop(self):
        raise StopApp

    # -- widgets with scripted answers
    def selectbox(self, label, options, index=0, **kwargs):
        self.calls.append(("selectbox", label))
        options = list(options)
        i = self.script["selection_index"]
        return None if i is None else options[i]

    def button(self, label, disabled=False, **kwargs):
        self.calls.append(("button", label))
        return self.script["button"] and not disabled

    def radio(self, label, options, **kwargs):
        return self.script["industry"]

    def multiselect(self, label, options, **kwargs):
        return self.script["genres"]

    def select_slider(self, label, options, value=None, **kwargs):
        return self.script["how_many"]

    # -- layout
    def columns(self, spec, **kwargs):
        count = len(spec) if isinstance(spec, (list, tuple)) else spec
        return [MagicMock() for _ in range(count)]

    def expander(self, label, **kwargs):
        self.calls.append(("expander", label))
        return MagicMock()

    def spinner(self, *args, **kwargs):
        return MagicMock()

    # -- everything else is just recorded
    def __getattr__(self, name):
        def recorder(*args, **kwargs):
            self.calls.append((name, args[0] if args else None))
            return MagicMock()
        return recorder

    def rendered(self, name):
        return [arg for fn, arg in self.calls if fn == name]
