"""TheTalker: streaming-serving benchmark client and deploy configs for open TTS models.

Public surface:
    thetalker.metrics        RTFx / TTFA / continuity math, one implementation
    thetalker.client         the load-generating client (``main()`` is the CLI)
    thetalker.deploy_configs deploy-config lookup behind the ``thetalker-deploy`` command
"""

__version__ = "0.1.0"

__all__ = ["__version__"]
