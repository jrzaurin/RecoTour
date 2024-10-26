# sort imports
isort --quiet . rec_zoo
# Black code style
black . rec_zoo
# flake8 standards
flake8 . --max-complexity=10 --max-line-length=127 --ignore=E203,E266,E501,E722,E721,F401,F403,F405,W503,C901,F811
# mypy
mypy rec_zoo --ignore-missing-imports --no-strict-optional
