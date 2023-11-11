# DTU-Special-Course

## Tasks
- Python inspect to import functions into jupyter
- Jupyter book (https://jupyterbook.org/en/stable/publish/gh-pages.html)
- pytest (https://docs.pytest.org/en/7.4.x/)


### Notes:
*- To build the Jupyter book use:* jupyter-book build DTU\ Special\ Course/ --path-output DTU\ Special\ Course/jupyter-book/ --config DTU\ Special\ Course/jupyter-book/_config.yml
*- To push the changes to Github:* ghp-import -n -p -f _build/html

- Use scoped-style
- pyproject.toml (dependencies)


python setup.py build_ext --inplace