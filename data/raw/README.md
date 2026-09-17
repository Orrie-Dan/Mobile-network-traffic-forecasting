# Raw Data

The raw Milan mobile telecommunications dataset from Harvard Dataverse is **not stored in this repository**.

## Why data is external

The complete dataset contains 62 files and is approximately **19.4 GB**. This exceeds practical limits for version control on GitHub, so raw files must remain outside the repository.

## Where the data lives

Raw ZIP files are stored externally in **Google Drive** and accessed for processing through **Google Colab**.

## Guidelines

- Do not commit raw dataset files to Git.
- Do not place large extracted files under this directory in a tracked state.
- Do not rename or modify the original dataset files.
- Use `data/processed/` for cleaned or aggregated outputs derived from the raw data.
- Use `data/samples/` only for small excerpts needed for development or testing.
