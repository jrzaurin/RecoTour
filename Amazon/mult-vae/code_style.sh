# # Black code style
black . models utils tests
# flake8 standards
flake8 . --max-complexity=10 --max-line-length=127 --ignore=W503,C901,E203
# mypy
mypy prepare_data.py --ignore-missing-imports