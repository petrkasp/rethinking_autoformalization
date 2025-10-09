import Lake
open Lake DSL

package «my-project» where
  -- add package configuration options here

lean_lib «MyProject» where
  -- add library configuration options here

@[default_target]
lean_exe «my-project» where
  root := `Main

require mathlib from git
  "https://github.com/leanprover-community/mathlib4.git" @ "59fdb6b04d7d16825a54483d550d9572ff473abf"

require REPL from git
  "https://github.com/leanprover-community/repl.git" @ "2ab7948163863ee222891653ac98941fe4f20e87"

require «proofNet-lean4» from git
  "https://github.com/rahul3613/ProofNet-lean4.git" @ "60efffb605ee07bf723db4fb8058129a7c8a89bb"

require ConNF from git
  "https://github.com/leanprover-community/con-nf.git" @ "16041ae6ea8b9a2ca79952afc7b927ccea18697b"

require «doc-gen4» from git
  "https://github.com/leanprover/doc-gen4.git" @ "780bbec107cba79d18ec55ac2be3907a77f27f98"
