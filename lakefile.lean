import Lake
open Lake DSL

package «rautoformalizer» where
  -- add package configuration options here

@[default_target]
lean_exe «rautoformalizer» where
  root := `Main

require mathlib from git
  "https://github.com/leanprover-community/mathlib4.git" @ "v4.8.0"

require REPL from git
  "https://github.com/leanprover-community/repl.git" @ "v4.8.0"

require «proofNet-lean4» from git
  "https://github.com/rahul3613/ProofNet-lean4.git" @ "60efffb605ee07bf723db4fb8058129a7c8a89bb"

require ConNF from git
  "https://github.com/leanprover-community/con-nf.git" @ "16041ae6ea8b9a2ca79952afc7b927ccea18697b"

-- removed doc gen, couldn't get it to work with v4.8.0 and hopefully it's not needed for anything
