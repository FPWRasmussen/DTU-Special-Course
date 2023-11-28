[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/FPWRasmussen/DTU-Special-Course/main)
# Wind Energy Analysis Tools

This repository contains a collection of Jupyter notebooks developed as a part of the special course "Noise and Visual Impact Analysis in Wind Energy" at DTU (Technical University of Denmark). The notebooks are designed to assist in wind turbine impact assessments, specifically addressing shadow flickering, noise calculations (ISO 9613-2), and generating visual impact assessments using the Google Street View API.

## Disclaimer

This project is a student initiative, and as such, the author does not assume any liability or provide any warranty for the quality of the product. Users are encouraged to contribute to further development under the terms of the [GNU Affero General Public License v3.0](LICENSE) license agreement.

## Usage

To use the notebooks, follow the steps outlined below: (TODO)

## Acknowledgments

Special thanks to the course instructors at the DTU Wind department for their guidance and support during the development of these tools.


## Tasks
- Python inspect to import functions into jupyter
- Jupyter book (https://jupyterbook.org/en/stable/publish/gh-pages.html)
- pytest (https://docs.pytest.org/en/7.4.x/)


## Notes:
*- To build the Jupyter book use:* jupyter-book build DTU\ Special\ Course/ --path-output DTU\ Special\ Course/jupyter-book/ --config DTU\ Special\ Course/jupyter-book/_config.yml (--builder pdfhtml) (DROPBOX)

*- To push the changes to Github:* ghp-import -n -p -f _build/html (jupyter-book)

- Use scoped-style
- pyproject.toml (dependencies)

python setup.py build_ext --inplace
