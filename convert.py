import nbformat
from nbconvert import MarkdownExporter
from traitlets.config import Config
from custom_preprocess import CustomPreprocessor


def parse_metadata(nb: nbformat.NotebookNode):
    """
    Any metadata field can be added to a notebook and parsed as follows. 
    These are a few examples of existing fields
    """
    metadata_map = {}
    nb_metadata = nb.metadata

    metadata_map["ms.author"] = nb_metadata["authors"][0]["name"]
    metadata_map["title"] = nb_metadata["task"]

    if nb_metadata.get("description") is None:
        metadata_map["description"] = ""
    else:
        metadata_map["description"] = nb_metadata["description"]
    return metadata_map


def build_append_md_metadata(nb_metadata: dict, nb_as_md):
    meta_string = "---\n" \
    "title: " + nb_metadata.get("title") + "\n" \
    "titleSuffix: Azure Machine Learning\n" \
    "description: " + nb_metadata.get("description") + "\n" \
    "services: machine-learning\n" \
    "ms.service: machine-learning\n" \
    "ms.subservice: core\n" \
    "ms.topic: conceptual\n" \
    "ms.author: " + nb_metadata.get("ms.author") + "\n" \
    "author: \n" \
    "ms.reviewer: " + nb_metadata.get("ms.author") + "\n" \
    "ms.date: 02/13/2020 \n" \
    "---\n\n"

    content = nb_as_md[0]
    content = meta_string + content
    copy = (content, nb_as_md[1])
    return copy


# load nb and extract metadata
nb = nbformat.read("model-register-and-deploy.ipynb", as_version=4)
nb_metadata = parse_metadata(nb)

# process and convert to .md
config = Config()
config.MarkdownExporter.preprocessors = [CustomPreprocessor]
custom_exporter = MarkdownExporter(config=config)
nb_as_md = custom_exporter.from_notebook_node(nb)

# append msdocs metadata
nb_as_md = build_append_md_metadata(nb_metadata, nb_as_md)

with open("model-register-and-deploy.md", 'w') as f:    
    f.write(nb_as_md[0])
