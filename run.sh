export TMPDIR=$SCRATCHDIR
trap 'clean_scratch' TERM EXIT

module add python/3.10.4-gcc-8.3.0-ovkjwzd
