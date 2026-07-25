"""Public reproduction-pack exporters for the research atlas."""

__all__ = ["export_public_pack"]


def __getattr__(name: str):
    if name == "export_public_pack":
        from lol_kills.export.public_pack import export_public_pack

        return export_public_pack
    raise AttributeError(name)
