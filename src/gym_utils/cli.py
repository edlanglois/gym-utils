"""Command line / script helpers."""
import argparse

__all__ = [
    'DictLookupAction',
]


class DictLookupAction(argparse.Action):
    """Argparse action that allows only keys from a given dictionary."""

    def __init__(self,
                 option_strings,
                 dest,
                 nargs=None,
                 default=None,
                 choices=None,
                 required=False,
                 help=None,
                 metavar=None):
        if choices is None:
            raise ValueError('Must set choices to the lookup dict.')
        self.dict = choices
        super().__init__(
            option_strings,
            dest,
            nargs=nargs,
            default=(self.dict[default] if default is not None else None),
            choices=self.dict.keys(),
            required=required,
            help=help,
            metavar=metavar)

    def __call__(self, parser, namespace, values, option_string=None):
        if self.nargs in (None, '?'):
            mapped_values = self.dict[values]
        else:
            mapped_values = [self.dict[v] for v in values]
        setattr(namespace, self.dest, mapped_values)
