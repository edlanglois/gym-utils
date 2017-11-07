#!/usr/bin/env python
"""Run an RL algorithm on an OpenAI gym environment."""
import argparse
import importlib
import itertools
import os.path
import tempfile
import time

import gym
from gym_utils import agents


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


def parse_args():
    """Parse command-line arguments.

    Returns:
        An `argparse.Namespace` object containing the parsed arguments.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0]
                                     if __doc__ else None)
    parser.add_argument(
        '-e', '--env', default='CartPole-v0', help='Environment ID.')
    parser.add_argument(
        '-a',
        '--agent',
        action=DictLookupAction,
        default='random',
        choices={
            'random': agents.RandomAgent,
        },
        help='Agent')
    parser.add_argument(
        '--max-steps',
        type=int,
        metavar='N',
        help='Run for at most %(metavar)s steps.')
    parser.add_argument(
        '--render', action='store_true', help='Render the environment.')
    parser.add_argument(
        '-I',
        '--import',
        type=str,
        nargs='+',
        dest='import_',
        metavar='MODULE',
        help='Import the named modules before creating the environment.')
    parser.add_argument(
        '--log',
        action='store_true',
        help='If true, store logs about the run.')
    parser.add_argument(
        '--log-dir',
        default=os.path.join(tempfile.gettempdir(),
                             os.path.splitext(os.path.basename(__file__))[0]),
        help='Base logging directory. (default: %(default)s)')

    return parser.parse_args()


def main():
    args = parse_args()
    if args.import_:
        for module_name in args.import_:
            importlib.import_module(module_name)

    env = gym.make(args.env)
    if args.log:
        env = gym.wrappers.Monitor(env, args.log_dir)
    agent = args.agent(env)

    observation = env.reset()
    for i in itertools.count():
        action = agent.act(observation)
        observation, reward, done, _ = env.step(action)
        agent.update(observation, reward, done)
        if args.render:
            env.render()
        if done:
            observation = env.reset()


if __name__ == '__main__':
    main()
