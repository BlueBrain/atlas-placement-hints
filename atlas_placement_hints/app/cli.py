"""The atlas-placement-hints command line launcher"""

import logging

import click

from atlas_placement_hints.app.placement_hints import app as cli_app
from atlas_placement_hints.version import VERSION

L = logging.getLogger(__name__)


def cli():
    """The main CLI entry point"""
    logging.basicConfig(level=logging.INFO)
    app = cli_app
    app = click.version_option(VERSION)(app)
    app()
