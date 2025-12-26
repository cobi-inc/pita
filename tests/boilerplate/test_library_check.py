import pita

def test_library_installed():
    """
    Test that the pita library is installed and can be imported.
    Also checks that the version attribute is available.
    """
    assert pita is not None
    assert hasattr(pita, "__version__")
    assert isinstance(pita.__version__, str)
    print(f"Pita library version: {pita.__version__}")
