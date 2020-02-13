# Sample code for converting `.ipynb` files to msdocs markdown using `nbconvert` library

This translates metadata from notebooks into correct metadata format for msdocs. Additional metadata fields should be added to `.ipynb` files to populate `.md` metadata. In this sample, only a couple fields are populated for demonstration.

`custom_preprocess.py` contains a custom override for preprocess class, currently just removing image tags and the copyright cell. This can be extended to perform additional checks expected by msdocs that aren't enforced for notebooks (e.g. `en-us` locale in url's, product name blacklists, etc.)

Final result is full conversion to `model-register-and-deploy.md`. See `env.yaml` for required dependencies.
