python3 -m venv .env
.env/bin/pip install -r requirements.txt

curl https://elan.lean-lang.org/elan-init.sh -sSf | sh
1
source $HOME/.elan/env
elan toolchain install 4.7.0-rc2
lake update
lake exe cache get
lake build
lake build repl
