# Contributing

Thank you for your interest in contributing to this project.

## Reporting issues

If you find a bug or have a feature request, please open a
[GitHub issue](https://github.com/Amy-Hsu/Automated-Cloud-Removal-and-Seamless-Satellite-Image-Mosaicking-Pipeline/issues).
Include as much detail as possible: error messages, input data description,
Python/GDAL versions, and the step that failed.

## Submitting changes

1. Fork the repository and create a feature branch.
2. Make your changes; keep commits focused and descriptive.
3. Test your changes with your own imagery if possible.
4. Open a pull request with a clear description of what you changed and why.

## Code style

- Python code should be compatible with Python 3.9+.
- Use descriptive variable names and include docstrings for new functions.
- Keep the four-step pipeline structure intact — changes that break
  downstream steps need careful coordination.

## Scope

This pipeline was designed for orthorectified optical satellite imagery.
Contributions that extend it to other sensors or regions are welcome, as long
as they remain backward-compatible with the existing configuration format.
