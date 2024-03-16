asdf install python 3.10.8
asdf local python 3.10.8

python -m venv venv
source venv/bin/activate
pip install -r requirements.txt \
  --trusted-host mirrors.tools.huawei.com \
  -i http://mirrors.tools.huawei.com/pypi/simple
pip install -r dev-requirements.txt \
  --trusted-host mirrors.tools.huawei.com \
  -i http://mirrors.tools.huawei.com/pypi/simple
pre-commit install
