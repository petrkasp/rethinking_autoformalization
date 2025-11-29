python3 -m venv .env
.env/bin/pip install -r requirements.txt

curl https://elan.lean-lang.org/elan-init.sh -sSf | sh
source $HOME/.elan/env
elan toolchain install 4.8.0
lake update
lake exe cache get
lake build
lake build repl

git clone https://huggingface.co/internlm/internlm2-math-plus-20b

git clone https://huggingface.co/purewhite42/bm25_f
git clone https://huggingface.co/purewhite42/bm25_f_if
git clone https://huggingface.co/purewhite42/dependency_retriever_f
git clone https://huggingface.co/purewhite42/dependency_retriever_f_if

git clone https://huggingface.co/purewhite42/rautoformalizer_nora_internlm
git clone https://huggingface.co/purewhite42/rautoformalizer_ra_internlm
git clone https://huggingface.co/purewhite42/rautoformalizer_gtra_deepseek
