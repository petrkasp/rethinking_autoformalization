import asyncio
from time import time

from common.constants import INIT_WAIT_TIME
from common.repl import REPL

REPL_ROOT = ".lake/packages/REPL"
MATHLIB_ROOT = "./"

TEST_HEADER = "import Mathlib\n\nopen Complex Filter Function Metric Finset\nopen scoped BigOperators Topology\n\n\n"

def main():
    start = time()

    async def idk():
        repl = REPL(
            repl_root=REPL_ROOT,
            project_root=MATHLIB_ROOT,
            timeout=60*60 # 1 hour
        )
        repl._run_interactive()
        await asyncio.sleep(INIT_WAIT_TIME)
        init_env = await repl.run_cmd_async(TEST_HEADER)
        print("REPL Succesfully loaded for the first time! Took:", time() - start)

    asyncio.run(idk())

if __name__ == "__main__":
    main()
