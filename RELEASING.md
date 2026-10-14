# Publishing DiskMELTS to PyPI

The package metadata is in `pyproject.toml`. The wheel bundles the production
checkpoints and continuum reference tables under `src/diskmelts/`; the large
`Model_grids/`, `Pretrain_grid/`, and top-level `Trained_model/` directories
are not part of the distribution.

1. Create a [PyPI account](https://pypi.org/account/register/) and verify its
   email address. Confirm that the name `diskmelts` is available on PyPI. If
   it is taken, change `project.name` in `pyproject.toml` and the installation
   commands in the README and installation guide.
2. Set a new, unused `project.version` in `pyproject.toml` for every release.
   Commit and push the tested source. PyPI does not permit replacing an
   uploaded distribution file.
3. Build and inspect the distributions from a clean checkout:

   ```bash
   python -m pip install --upgrade build twine
   python -m build
   python -m twine check dist/*
   ```

   Inspect the wheel and source archive before upload, especially when model
   files change. They should contain the packaged checkpoints and continuum
   tables, but not the slab grids or private data.
4. Test the wheel in a fresh environment **outside** the source checkout:

   ```bash
   python -m venv /tmp/diskmelts-release-test
   /tmp/diskmelts-release-test/bin/python -m pip install dist/diskmelts-*.whl
   cd /tmp
   /tmp/diskmelts-release-test/bin/python -c "import diskmelts; print(diskmelts.load_fitting_models('C2H2').keys())"
   ```

   On Windows, use the virtual environment's `Scripts/python.exe` path.
5. To rehearse the upload, use [TestPyPI](https://packaging.python.org/en/latest/guides/using-testpypi/)
   with a separate TestPyPI account and token: `python -m twine upload
   --repository testpypi dist/*`.
6. Publish to PyPI with `python -m twine upload dist/*` and a PyPI API token.
   Then verify `python -m pip install diskmelts` in a fresh environment from
   outside the checkout. Tag the released commit (for example, `v2.1`).

For automated releases, PyPI's
[Trusted Publishing](https://docs.pypi.org/trusted-publishers/)
can connect a GitHub Actions release workflow to PyPI without storing a long
lived token. Register the repository owner, repository name, and exact workflow
filename in PyPI before using that route.
